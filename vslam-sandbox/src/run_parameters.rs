use std::str::FromStr;

#[derive(Debug)]
pub struct RunParameters {
    pub option_parameters: OptionParametersParsed,
    pub files_path: Vec<String>,
}

#[derive(Debug)]
pub struct OptionParameters {
    pub input_settings_file: String,
    pub input_focal_x: String,
    pub input_focal_y: String,
    pub input_focal_x_center: String,
    pub input_focal_y_center: String,
    pub input_focal_radial_distortion: String,
}

#[derive(Debug)]
pub struct OptionParametersParsed {
    pub input_settings_file: String,
    pub input_focal_x: f64,
    pub input_focal_y: f64,
    pub input_focal_x_center: f64,
    pub input_focal_y_center: f64,
    pub input_focal_radial_distortion: f64,
}

impl OptionParameters {
    pub fn try_parse(&self) -> Result<OptionParametersParsed, <f64 as FromStr>::Err> {
        Ok(OptionParametersParsed {
            input_settings_file: self.input_settings_file.clone(),
            input_focal_x: self.input_focal_x.parse()?,
            input_focal_y: self.input_focal_y.parse()?,
            input_focal_x_center: self.input_focal_x_center.parse()?,
            input_focal_y_center: self.input_focal_y_center.parse()?,
            input_focal_radial_distortion: self.input_focal_radial_distortion.parse()?,
        })
    }
}