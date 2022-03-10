//! Parser for ply
//!
//! Documentation for ply:
//! https://brainder.org/tag/ply/
//!
//! Example of ply file - FreeSurfer surface (*.asc | suggested: *.srf )
//! ascii version of ... <-- Identification
//! Number of vertices --> 6 8 <-- Comment
//! # Vertex coordinates (x,y,z) - last param is *Is in patch (boolean; not used)
//! 1.0 0.0 0.0 0
//! 0.0 1.0 0.0 0
//! 0.0 1.0 0.0 0
//! -1.0 0.0 0.0 0
//! 0.0 -1.0 0.0 0
//! # Definition of faces from vertex indices (Zero indexed)
//! # Last param is same as last param for vertex coordinates
//! 0 1 2 0
//! 1 3 2 0
//! 4 0 2 0
//! 1 0 5 0
//! 3 1 5 0
//! 4 3 5 0
//! 0 4 5 0
//!
//! Example of ply file - FreeSurfer surface (*.asc | suggested: *.srf )
//!
//! #Vertex index (zero indexed) - Vertex coordinates (x,y,z), float - Scalar values
//! 000 1 0 0  0.59075
//! 001 0 1 0 -0.44027
//! 002 0 0 1 1.69289
//! 003 -1 0 0 -0.61236
//! 004 0 -1 0 0.82814
//! 005 0 0 -1 0.71058
//!
//! Facewise data (*.dpf)
//!  
//! #Vertex index (zero indexed) - Vertex indices that define faces (zero indexed), unsigned integers - Scalar values
//! 000 0 1 2 -0.064088
//! 001 1 3 2 1.087376
//! 002 3 4 2 0.781619
//! 003 4 0 2 2.133629
//! 004 1 0 5 -0.778786
//! 005 3 1 5 0.058316
//! 006 4 3 5 0.996754
//! 007 0 4 5 -1.141163
//!
//! VTK polygonal data (*.vtk)
//!
//! # vtk DataFile Version 2.0 <-- Identification
//! Octahedron circumradius=1 <-- Comment (max 256 chars)
//! # Metadata
//! ASCII
//! DATASET POLYDATA
//! --> POINTS 8 FLOAT
//! #Vertex coordinates (x,y,z)
//! 1.0 0.0 0.0
//! 0.0 1.0 0.0
//! 0.0 1.0 0.0
//! -1.0 0.0 0.0
//! 0.0 -1.0 0.0
//! POLYGONS 8 32 <-- Number of faces & numbeer of values in this section of the file
//! # First number is amount of vertices that define this line
//! # The other 3 are definitions of faces from vertex indices (zero indexed)
//! 3 0 1 2
//! 3 1 3 2
//! 3 3 4 2
//! 3 4 0 2
//! 3 1 0 5
//! 3 3 1 5
//! 3 4 3 5
//! 3 0 4 3
//!
//! Wavefront Object (*.obj)
//!
//! # The "v" identifies this element as a vertex
//! # The other values are vertex coordinates (x, y, z)
//! v 1.0 0.0 0.0
//! v 0.0 1.0 0.0
//! v 0.0 1.0 0.0
//! v -1.0 0.0 0.0
//! v 0.0 -1.0 0.0
//! # The "f" identifies this element as a face
//! # The other 3 values are definitions of faces from vertex indices (1 indexed!!)
//! f 1 2 3
//! f 2 4 3
//! f 4 5 3
//! f 5 1 3
//! f 2 1 6
//! f 4 2 6
//! f 5 4 6
//! f 1 5 6
//!
//! TODO: Maybe only keep this
//! Standard Polygon (*.ply) - with color attributes
//!
//! =============== header begins
//! ply
//! format ascii 1.0
//! element vertex 6 <-- Number of vertices
//! =============== Vertex coordinate type
//! property float x
//! property float y
//! property float z
//! =============== Data type for color property
//! property uchar red
//! property uchar green
//! property uchar blue
//! element face 8 <-- Number of faces
//! =============== How the vertex indices are listed to define faces
//! property list uchar int vertex_index
//! end_header
//! =============== Vertex coordinates (x, y, z) - RGB triplet to define vertex colors (0 to 255)
//! 1.0 0.0 0.0 255 0 0
//! 0.0 1.0 0.0 0 255 0
//! 0.0 1.0 1.0 0 0 255
//! -1.0 0.0 0.0 0 0 255
//! 0.0 -1.0 0.0 255 0 255
//! 0.0 0.0 -1.0 255 255 255
//! =============== First value is the number of vertices that define this face
//! =============== The rest are definitions of faces from vertex indices (Zero indexed)
//! 3 0 1 2
//! 3 1 3 2
//! 3 3 4 2
//! 3 4 0 2
//! 3 1 0 5
//! 3 3 1 5
//! 3 3 1 5
//! 3 4 3 5
//! 3 0 4 5

use std::fs;
use std::io;

#[derive(Debug)]
struct Header<'a> {
    file_format: &'a str,
    char_format: &'a str,
}

#[derive(Debug)]
struct VertexAmount {
    amount: u16,
}

#[derive(Debug)]
struct VertexCoordinateType;

#[derive(Debug)]
struct ColorPropType {
    amount: u16,
}

#[derive(Debug)]
struct VertexDefinition {
    info: String,
    collection: Vec<char>,
    vertex_index: i16,
}

#[derive(Debug)]
struct Vertex {
    x: f64,
    y: f64,
    z: f64,
}

#[derive(Debug)]
struct Color {
    r: u8,
    g: u8,
    b: u8,
}

#[derive(Debug)]
struct VertexWithColor {
    vertex: Vertex,
    color: Color,
}

type Point = [u8; 3];

#[derive(Debug)]
struct VertexFaceDefinition {
    num_of_faces: u8,
    point: Point,
}

#[derive(Debug)]
pub struct PlyColored<'a> {
    header: Header<'a>,
    vertex_amount: u8,
    vert_coord_type: VertexCoordinateType, // property float x...
    color_prop_type: ColorPropType,
    num_of_faces: u8,
    vert_info: String,
    vert_coordinates: Vec<VertexWithColor>,
    vert_face_defs: Vec<VertexFaceDefinition>,
}

// #[derive(Debug, Clone)]
pub struct Ply<'a> {
    header: Header<'a>,
    vertex_amount: u8,
    // vert_coord_type: VertexCoordinateType,
    num_of_faces: u8,
    // vert_info: String,
    vertices: Vec<Vertex>,
    vert_face_defs: Vec<VertexFaceDefinition>,
}

/// Go throug a ply file and parse the contents into a ply struct to use
/// FIXME: rustify code and add error handling
pub fn parse_ply(filename: String) {
    //
    let mut contents = fs::read_to_string(filename).expect("Failed to write PLY file");

    let contents: Vec<&str> = contents.split("\\n").collect();

    let header = Header {
        file_format: contents[0],
        char_format: contents[1],
    };
    let vertex_amount: u8 = contents[2].4;
    let color_props = 3;
    let num_of_faces = contents[6];
    let vertex_def: String = contents[7];
    // end header at 8
    let mut vertices: Vec<VertexWithColor> = Vec::new();
    // FIXME: May have to use slice to iter function
    for line in contents[9..9 + 6] {
        let vertex = Vertex {
            x: line[0],
            y: line[1],
            z: line[2],
        };
        let color = Color {
            r: line[3],
            g: line[4],
            b: line[5],
        };
        let vertex_coords_with_color = VertexWithColor { vertex, color };
        vertices.push(vertex_coords_with_color);
    }
    let mut vert_face_defs: Vec<VertexFaceDefinition> = Vec::new();
    for line in contents[15..] {
        let vert_face_def = VertexFaceDefinition {
            num_of_faces: line[0],
            point: [line[1], line[2], line[3]],
        };
        vert_face_defs.push(vert_face_def);
    }

    // Return a PLY struct containing all info we need
    PlyColored {
        header,
        vertex_amount,
        // Vertex coordinate type,
        color_prop_type,
        num_of_faces,
        // Vert info,
        vert_coordinates: vertices,
        vertices,
        vert_face_defs,
    }
}
