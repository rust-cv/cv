use nalgebra::{Point2};


pub struct SIFTConfig {
    pub sigma: f64,
    pub num_intervals: usize,
    pub assumed_blur: f64,
    pub image_border_width: usize
}


impl Default for SIFTConfig {
    fn default() -> Self {
        SIFTConfig {
            sigma: 1.6,
            num_intervals: 3,
            assumed_blur: 0.5,
            image_border_width: 5
        }
    }
}

impl SIFTConfig {
    pub fn new() -> Self {
        SIFTConfig::default()
    }
}

pub struct KeyPoint {
    pub pt: Point2<f64>,
    pub octave: usize,
    pub size: f64,
    pub angle: f64,
    pub response: f64
}

pub struct Descriptor {
    pub vector: [f64; 128]
}