use glium::glutin;
use image::{DynamicImage, GenericImageView};

fn create_display(
    width: f32,
    height: f32,
    event_loop: &glutin::event_loop::EventLoop<()>,
) -> glium::Display {
    let window_builder = glutin::window::WindowBuilder::new()
        .with_resizable(true)
        .with_inner_size(glutin::dpi::LogicalSize { width, height })
        .with_title("Rust CV");

    let context_builder = glutin::ContextBuilder::new()
        .with_depth_buffer(0)
        .with_srgb(true)
        .with_stencil_buffer(0)
        .with_vsync(true);

    glium::Display::new(window_builder, context_builder, event_loop).unwrap()
}

fn read_image(
    egui: &mut egui_glium::EguiGlium,
    image: &DynamicImage,
) -> Result<(egui::TextureId, egui::emath::Vec2), std::io::Error> {
    // Convert the data to an array of epaint colors.
    let image_bytes = image.as_bytes();
    let mut pixels = Vec::with_capacity(image_bytes.len() / 4);
    for rgba in image_bytes.chunks_exact(4) {
        match rgba {
            &[r, g, b, a] => pixels.push(egui::Color32::from_rgba_unmultiplied(r, g, b, a)),
            _ => (),
        }
    }

    // Create a new texture and upload the pixels to it.
    let (_, painter) = egui.ctx_and_painter_mut();
    let id = painter.alloc_user_texture();
    painter.set_user_texture(
        id,
        (image.width() as usize, image.height() as usize),
        &pixels,
    );

    let size = egui::emath::Vec2::new(image.width() as f32, image.height() as f32);
    Ok((id, size))
}
pub fn imgshow(image: &DynamicImage) -> ! {
    let event_loop = glutin::event_loop::EventLoop::with_user_event();
    let display = create_display(
        (image.width() + 15) as f32,
        (image.height() + 15) as f32,
        &event_loop,
    );

    let mut egui = egui_glium::EguiGlium::new(&display);

    let (image_id, image_size) = read_image(&mut egui, &image).expect("Couldn't read image");

    event_loop.run(move |event, _, control_flow| {
        let mut redraw = || {
            egui.begin_frame(&display);

            let (_, shapes) = egui.end_frame(&display);
            {
                let mut target = display.draw();

                egui::CentralPanel::default().show(egui.ctx(), |ui| {
                    ui.image(image_id, image_size);
                });

                egui.paint(&display, &mut target, shapes);

                // draw things on top of egui here

                target.finish().unwrap();
            }
        };

        match event {
            // Platform-dependent event handlers to workaround a winit bug
            // See: https://github.com/rust-windowing/winit/issues/987
            // See: https://github.com/rust-windowing/winit/issues/1619
            glutin::event::Event::RedrawEventsCleared if cfg!(windows) => redraw(),
            glutin::event::Event::RedrawRequested(_) if !cfg!(windows) => redraw(),

            glutin::event::Event::WindowEvent { event, .. } => {
                if egui.is_quit_event(&event) {
                    *control_flow = glium::glutin::event_loop::ControlFlow::Exit;
                }

                egui.on_event(&event);

                display.gl_window().window().request_redraw(); // TODO: ask egui if the events warrants a repaint instead
            }

            _ => (),
        }
    })
}
