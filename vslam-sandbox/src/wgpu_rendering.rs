use kas_wgpu::wgpu;
use std::{cell::RefCell, mem::size_of, rc::Rc};
use wgpu::{include_spirv, Buffer, ShaderModule};
use wgpu::{util::DeviceExt, BindingResource};

use bytemuck::{Pod, Zeroable};
use rand::Rng;

use cgmath::*;
use kas::prelude::*;
use kas::{draw::Pass, geom::Rect};
use kas_wgpu::draw::{CustomPipe, CustomPipeBuilder, CustomWindow};

struct Shaders {
    vertex: ShaderModule,
    fragment: ShaderModule,
}

impl Shaders {
    fn new(device: &wgpu::Device) -> Self {
        let vertex = device.create_shader_module(include_spirv!(
            "../shaders/point_cloud_viewer/shader.vert.spv"
        ));

        let fragment = device.create_shader_module(include_spirv!(
            "../shaders/point_cloud_viewer/shader.frag.spv"
        ));

        Shaders { vertex, fragment }
    }
}

pub const OPENGL_TO_WGPU_MATRIX: Matrix4<f32> = Matrix4::new(
    1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5, 1.0,
);

fn get_uniform_as_u8_vec(mat: &Matrix4<f32>, rect: &Rect, window_size: &Size) -> Vec<u8> {
    let matrix_ref: &[f32; 16] = mat.as_ref();
    let matrix_as_byte_array: &[u8] = bytemuck::cast_slice(matrix_ref);

    let offset = vec![rect.pos.0 as f32, rect.pos.1 as f32];
    let size = vec![rect.size.0 as f32, rect.size.1 as f32];
    let window_size = vec![window_size.0 as f32, window_size.1 as f32];

    let mut output: Vec<u8> = vec![];

    output.extend(bytemuck::cast_slice(matrix_as_byte_array));
    output.extend(bytemuck::cast_slice(&offset[..]));
    output.extend(bytemuck::cast_slice(&size[..]));
    output.extend(bytemuck::cast_slice(&window_size[..]));

    output
}

fn get_matrix(aspect_ratio: f32, cam_pos_x: f32) -> Matrix4<f32> {
    let aspect_ratio = if aspect_ratio <= 0.0f32 {
        print!(
            "aspect ratio value is not correct: {}. Taking 1.0 instead",
            aspect_ratio
        );
        1.0f32
    } else {
        aspect_ratio
    };

    assert!(aspect_ratio > 0.0f32);

    let perspective: PerspectiveFov<f32> = PerspectiveFov::<f32> {
        fovy: Rad::<f32>::from(Deg::<f32>(90.0)),
        aspect: aspect_ratio,
        near: 0.01,
        far: 1000.0,
    };

    let projection_matrix = Matrix4::<f32>::from(perspective.to_perspective());

    let distance = 3.0f32;

    let transformation_matrix = Matrix4::look_at(
        Point3::new(
            (cam_pos_x as f32).cos() * distance,
            0.0 as f32,
            (cam_pos_x as f32).sin() * distance,
        ),
        Point3::new(0f32, 0.0, 0.0),
        Vector3::unit_y(),
    );

    OPENGL_TO_WGPU_MATRIX * projection_matrix * transformation_matrix
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct Vertex {
    _pos: [f32; 3],
    _color: [f32; 4],
}

unsafe impl Zeroable for Vertex {}
unsafe impl Pod for Vertex {}

pub struct PipeBuilder;

impl CustomPipeBuilder for PipeBuilder {
    type Pipe = Pipe;

    fn build(
        &mut self,
        device: &wgpu::Device,
        tex_format: wgpu::TextureFormat,
        depth_format: wgpu::TextureFormat,
    ) -> Self::Pipe {
        let shaders = Shaders::new(device);

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&uniform_bind_group_layout],
            push_constant_ranges: &[],
        });

        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex_stage: wgpu::ProgrammableStageDescriptor {
                module: &shaders.vertex,
                entry_point: "main",
            },
            fragment_stage: Some(wgpu::ProgrammableStageDescriptor {
                module: &shaders.fragment,
                entry_point: "main",
            }),
            rasterization_state: None,
            primitive_topology: wgpu::PrimitiveTopology::PointList,
            color_states: &[wgpu::ColorStateDescriptor {
                format: tex_format,
                color_blend: wgpu::BlendDescriptor::REPLACE,
                alpha_blend: wgpu::BlendDescriptor::REPLACE,
                write_mask: wgpu::ColorWrite::ALL,
            }],
            depth_stencil_state: Some(wgpu::DepthStencilStateDescriptor {
                format: depth_format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Always,
                stencil: wgpu::StencilStateDescriptor {
                    front: wgpu::StencilStateFaceDescriptor::IGNORE,
                    back: wgpu::StencilStateFaceDescriptor::IGNORE,
                    read_mask: 0,
                    write_mask: 0,
                },
            }),
            vertex_state: wgpu::VertexStateDescriptor {
                index_format: wgpu::IndexFormat::Uint16,
                vertex_buffers: &[wgpu::VertexBufferDescriptor {
                    stride: size_of::<Vertex>() as wgpu::BufferAddress,
                    step_mode: wgpu::InputStepMode::Vertex,
                    attributes: &wgpu::vertex_attr_array![0 => Float3, 1 => Float4],
                }],
            },
            sample_count: 1,
            sample_mask: !0,
            alpha_to_coverage_enabled: false,
        });

        Pipe {
            render_pipeline,
            window_size: Rc::new(RefCell::new(Size(100, 100))),
        }
    }
}

pub struct Pipe {
    render_pipeline: wgpu::RenderPipeline,
    window_size: Rc<RefCell<Size>>,
}

impl Pipe {
    fn get_window_size(&self) -> Size {
        self.window_size.borrow().clone()
    }
}

pub struct RenderPassObj {
    pub vertices: Rc<RefCell<Vec<Vertex>>>,
    pub buffer: Option<Buffer>,
    pub size: u32,
}

pub struct PipeWindow {
    bind_group: wgpu::BindGroup,
    uniform_buffer: Buffer,
    passes: Vec<RenderPassObj>,
    render_rect: Rect,
    vertices: Rc<RefCell<Vec<Vertex>>>,
    cam_pos_x: f32,
}

impl CustomPipe for Pipe {
    type Window = PipeWindow;

    fn new_window(&self, device: &wgpu::Device, size: Size) -> Self::Window {
        let render_rect = Rect::new(Coord(0, 0), size);

        let uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::VERTEX,
                    ty: wgpu::BindingType::UniformBuffer {
                        dynamic: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
                label: Some("uniform_bind_group_layout"),
            });

        let cam_pos_x = 4.0f32;

        let matrix = get_matrix(1.0f32, cam_pos_x);
        let uniforms_value = get_uniform_as_u8_vec(&matrix, &render_rect, &size);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&uniforms_value),
            usage: wgpu::BufferUsage::UNIFORM | wgpu::BufferUsage::COPY_DST,
        });

        let uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &uniform_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: BindingResource::Buffer(uniform_buffer.slice(..)),
            }],
            label: Some("uniform_bind_group"),
        });

        PipeWindow {
            bind_group: uniform_bind_group,
            passes: vec![],
            render_rect: render_rect,
            uniform_buffer: uniform_buffer,
            vertices: Rc::new(RefCell::new(vec![])),
            cam_pos_x: cam_pos_x,
        }
    }

    fn resize(
        &self,
        _window: &mut Self::Window,
        _device: &wgpu::Device,
        _encoder: &mut wgpu::CommandEncoder,
        size: Size,
    ) {
        {
            let mut mut_size = self.window_size.borrow_mut();
            *mut_size = size;
        }
    }

    fn update(
        &self,
        window: &mut Self::Window,
        device: &wgpu::Device,
        encoder: &mut wgpu::CommandEncoder,
    ) {
        let window_size = self.get_window_size();
        window.cam_pos_x += 0.1f32;

        let aspect_ratio = window.render_rect.size.0 as f32 / window.render_rect.size.1 as f32;
        let matrix = get_matrix(aspect_ratio, window.cam_pos_x);
        let render_rect = window.render_rect;

        let uniform_values = get_uniform_as_u8_vec(&matrix, &render_rect, &window_size);

        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(&uniform_values),
            usage: wgpu::BufferUsage::COPY_SRC,
        });

        encoder.copy_buffer_to_buffer(
            &uniform_buffer,
            0,
            &window.uniform_buffer,
            0,
            uniform_values.len() as u64,
        );

        for render_pass_obj in &mut window.passes {
            let vertices_len = render_pass_obj.vertices.borrow().len();
            if vertices_len > 0 {
                let vertices = render_pass_obj.vertices.borrow_mut();

                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: None,
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsage::VERTEX,
                });

                render_pass_obj.buffer = Some(buffer);
                render_pass_obj.size = vertices_len as u32;
            } else {
                render_pass_obj.buffer = None;
            }
        }
    }

    fn render_pass<'a>(
        &'a self,
        window: &'a mut Self::Window,
        _device: &wgpu::Device,
        pass: usize,
        render_pass: &mut wgpu::RenderPass<'a>,
    ) {
        if let Some(render_pass_obj) = window.passes.get(pass) {
            if let Some(buffer) = render_pass_obj.buffer.as_ref() {
                render_pass.set_pipeline(&self.render_pipeline);
                render_pass.set_bind_group(0, &window.bind_group, &[]);
                render_pass.set_vertex_buffer(0, buffer.slice(..));
                render_pass.draw(0..render_pass_obj.size, 0..1);
            }
        }
    }
}

impl CustomWindow for PipeWindow {
    type Param = ();

    fn invoke(&mut self, pass: Pass, rect: Rect, _param: Self::Param) {
        self.render_rect = rect;

        if self.passes.len() <= pass.pass() {
            self.passes.push(RenderPassObj {
                vertices: Rc::new(RefCell::new(vec![])),
                buffer: None,
                size: 0,
            });
        }

        let vertices_len = {
            let vertices_borrowed = self.vertices.borrow();
            vertices_borrowed.len()
        };

        if vertices_len == 0 {
            let mut vertices = vec![];
            let mut rng = rand::thread_rng();
            let max = 100;
            for i in 0..max {
                let index = i as f32 / max as f32 * 2.0f32 * std::f32::consts::PI;
                vertices.push(Vertex {
                    _pos: [index.cos(), index.sin(), 0.0],
                    _color: [
                        rng.gen_range(0.0, 1.0),
                        rng.gen_range(0.0, 1.0),
                        rng.gen_range(0.0, 1.0),
                        1.0,
                    ],
                });
            }

            *self.vertices.borrow_mut() = vertices;
        }

        self.passes[0].vertices = self.vertices.clone();
    }
}
