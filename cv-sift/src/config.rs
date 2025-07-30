

#[derive(Debug, Clone, Copy)]
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