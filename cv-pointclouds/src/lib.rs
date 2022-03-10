//! Rust CV Point Clouds
//! Library for Rust CV Tooling
//!
//! Display Tool for 3D models written as PLY Files
//!
//!
//!
//!
//!

mod ply_parser;

pub use ply_parser;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {
        let result = 2 + 2;
        assert_eq!(result, 4);
    }
}
