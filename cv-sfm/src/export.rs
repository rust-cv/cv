use cv_core::nalgebra::{Point3, Vector3};
use ply_rs::{
    ply::{
        Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
        ScalarType,
    },
    writer::Writer,
};
use std::io::Write;

const CAMERA_COLOR: [u8; 3] = [255, 0, 255];

pub struct ExportCamera {
    pub optical_center: Point3<f64>,
    pub up_direction: Vector3<f64>,
    pub forward_direction: Vector3<f64>,
    pub focal_length: f64,
}

pub fn export(
    mut writer: impl Write,
    points_and_colors: Vec<(Point3<f64>, [u8; 3])>,
    cameras: Vec<ExportCamera>,
    camera_faces: bool,
) {
    // crete a ply objet
    let mut ply = Ply::<DefaultElement>::new();
    ply.header.encoding = Encoding::Ascii;
    ply.header
        .comments
        .push("Exported from rust-cv/vslam-sandbox".to_string());

    // Define the vertex element, which will be used for face verticies and reconstruction points.
    let mut point_element = ElementDef::new("vertex".to_string());
    let p = PropertyDef::new("x".to_string(), PropertyType::Scalar(ScalarType::Double));
    point_element.properties.add(p);
    let p = PropertyDef::new("y".to_string(), PropertyType::Scalar(ScalarType::Double));
    point_element.properties.add(p);
    let p = PropertyDef::new("z".to_string(), PropertyType::Scalar(ScalarType::Double));
    point_element.properties.add(p);
    let p = PropertyDef::new("red".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("green".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    let p = PropertyDef::new("blue".to_string(), PropertyType::Scalar(ScalarType::UChar));
    point_element.properties.add(p);
    ply.header.elements.add(point_element);

    if camera_faces {
        // Define the face element which will be used for cameras.
        let mut face_element = ElementDef::new("face".to_string());
        let vertex_list = PropertyDef::new(
            "vertex_index".to_string(),
            PropertyType::List(ScalarType::UChar, ScalarType::Int),
        );
        face_element.properties.add(vertex_list);
        ply.header.elements.add(face_element);
    }

    let mut faces: Vec<DefaultElement> = vec![];
    let mut vertices: Vec<DefaultElement> = vec![];

    let mut add_vertex = |p: Point3<f64>, [r, g, b]: [u8; 3]| -> usize {
        let pos = vertices.len();
        let mut point = DefaultElement::new();
        point.insert("x".to_string(), Property::Double(p.x));
        point.insert("y".to_string(), Property::Double(p.y));
        point.insert("z".to_string(), Property::Double(p.z));
        point.insert("red".to_string(), Property::UChar(r));
        point.insert("green".to_string(), Property::UChar(g));
        point.insert("blue".to_string(), Property::UChar(b));
        vertices.push(point);
        pos
    };

    let mut add_triangle = |a: usize, b: usize, c: usize| -> usize {
        let pos = faces.len();
        let mut face = DefaultElement::new();
        face.insert(
            "vertex_index".to_string(),
            Property::ListInt(vec![a as i32, b as i32, c as i32]),
        );
        faces.push(face);
        pos
    };

    // Add cameras
    for ExportCamera {
        optical_center,
        up_direction,
        forward_direction,
        focal_length,
    } in cameras
    {
        let right_direction = forward_direction.cross(&up_direction);
        let center_point = add_vertex(optical_center, CAMERA_COLOR);
        let [up_right, up_left, down_left, down_right] =
            [(1, 1), (1, -1), (-1, -1), (-1, 1)].map(|(up, right)| {
                add_vertex(
                    optical_center
                        + forward_direction * focal_length
                        + up as f64 * up_direction * focal_length
                        + right as f64 * right_direction * focal_length,
                    CAMERA_COLOR,
                )
            });

        if camera_faces {
            add_triangle(center_point, down_right, up_right);
            add_triangle(center_point, up_right, up_left);
            add_triangle(center_point, up_left, down_left);
            add_triangle(center_point, down_left, down_right);
        }
    }

    // Add points
    for (p, c) in points_and_colors {
        add_vertex(p, c);
    }

    ply.payload.insert("vertex".to_string(), vertices);
    if camera_faces {
        ply.payload.insert("face".to_string(), faces);
    }

    // set up a writer
    let w = Writer::new();
    w.write_ply(&mut writer, &mut ply).unwrap();
}
