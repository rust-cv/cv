use cv::nalgebra::Point3;
use ply_rs::ply::{
    Addable, DefaultElement, ElementDef, Encoding, Ply, Property, PropertyDef, PropertyType,
    ScalarType,
};
use ply_rs::writer::Writer;
use std::io::Write;

pub fn export(mut writer: impl Write, points_and_colors: Vec<(Point3<f64>, [u8; 3])>) {
    // crete a ply objet
    let mut ply = Ply::<DefaultElement>::new();
    ply.header.encoding = Encoding::Ascii;
    ply.header
        .comments
        .push("Exported from rust-cv/vslam-sandbox".to_string());

    // Define the elements we want to write. In our case we write a 2D Point.
    // When writing, the `count` will be set automatically to the correct value by calling `make_consistent`
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

    // Add data
    let points: Vec<_> = points_and_colors
        .into_iter()
        .map(|(p, [r, g, b])| {
            let mut point = DefaultElement::new();
            point.insert("x".to_string(), Property::Double(p.x));
            point.insert("y".to_string(), Property::Double(p.y));
            point.insert("z".to_string(), Property::Double(p.z));
            point.insert("red".to_string(), Property::UChar(r));
            point.insert("green".to_string(), Property::UChar(g));
            point.insert("blue".to_string(), Property::UChar(b));
            point
        })
        .collect();

    ply.payload.insert("vertex".to_string(), points);

    // set up a writer
    let w = Writer::new();
    w.write_ply(&mut writer, &mut ply).unwrap();
}
