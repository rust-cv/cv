use image::{DynamicImage, RgbaImage, Rgba, io::Reader as ImageReader};
use nalgebra::{DMatrix, Scalar};
use imgshow::imgshow;
use num::{Num, NumCast};
use std::io::ErrorKind;


#[derive(Debug, Clone)]
pub struct SimpleRGBAImage<T: Num + NumCast + Send + Sync + Copy + Scalar> {
    pub height: u32,
    pub width: u32,
    pub channels: [DMatrix<T>; 4]
}

impl<T: Num + NumCast + Clone + Send + Sync + Scalar + Copy> IntoIterator for SimpleRGBAImage<T> {
    type Item = [T; 4];
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self
        .red()
        .iter()
        .zip(
            self
            .green()
            .iter()
        ).zip(
            self
            .blue()
            .iter()
        ).zip(
            self
            .alpha()
            .iter()
        )
        .map(|(((&a, &b), &c), &d)| [a, b, c, d])
        .collect::<Vec<Self::Item>>()
        .into_iter()
    }
}

impl<'a, T: Num + NumCast + Clone + Send + Sync + Scalar + Copy> IntoIterator for &'a SimpleRGBAImage<T> {
    type Item = [T; 4];
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self
        .red()
        .iter()
        .zip(
            self
            .green()
            .iter()
        ).zip(
            self
            .blue()
            .iter()
        ).zip(
            self
            .alpha()
            .iter()
        )
        .map(|(((&a, &b), &c), &d)| [a, b, c, d])
        .collect::<Vec<Self::Item>>()
        .into_iter()
    }
}

impl<'a, T: Num + NumCast + Clone + Send + Sync + Scalar + Copy> SimpleRGBAImage<T> {

    pub fn open(path: &str) -> Result<SimpleRGBAImage<T>, ErrorKind>{
        
        let img = ImageReader::open(path);
        if img.is_err() {
            return Err(ErrorKind::NotFound);
        }
        let clean_img = img
        .unwrap()
        .decode()
        .unwrap()
        .to_rgba8();
        let d_img = DynamicImage::ImageRgba8(clean_img);
        Ok(SimpleRGBAImage::<T>::from_dynamic(&d_img))
    }

    pub fn show(&self) {
        let temp = self.as_view();
        imgshow(&temp).unwrap();
    }

    pub fn shape(&self) -> (u32, u32) {
        (self.height, self.width)
    }

    pub fn zeros_like((height, width): (u32, u32)) -> SimpleRGBAImage<T> {
        SimpleRGBAImage::<T>::from_slice(
            &vec![NumCast::from(0_f64).unwrap(); height as usize * width as usize * 4_usize],
            width,
            height
        )
    }

    pub fn save(&self, path: &str) -> image::ImageResult<()> {
        self.as_view().save(path)
    }

    pub fn to_u8(&self) -> SimpleRGBAImage<u8> {
        SimpleRGBAImage::<u8>::from_dynamic(&self.as_view())       
    }

    pub fn to_f64(&self) -> SimpleRGBAImage<f64> {
        SimpleRGBAImage::<f64>::from_dynamic(&self.as_view())       
    }

    pub fn is_same_as(&self, other: &SimpleRGBAImage<T>) -> bool {
        if !(self.height == other.height && self.width == other.width) {
            return false;
        }
        let self_as_u8 = self.to_u8();
        let other_as_u8 = other.to_u8();

        self_as_u8.red() == other_as_u8.red() &&
        self_as_u8.green() == other_as_u8.green() &&
        self_as_u8.blue() == other_as_u8.blue() &&
        self_as_u8.alpha() == other_as_u8.alpha()
    }

    pub fn from_slice(slice: &[T], width: u32, height: u32) -> SimpleRGBAImage<T> {
        
        let mut red : Vec<T> = Vec::new();
        let mut green: Vec<T> = Vec::new();
        let mut blue: Vec<T> = Vec::new();
        let mut alpha: Vec<T> = Vec::new();
        
        slice.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 4 {
                0 => {
                    red.push(NumCast::from(pixel).unwrap());
                },
                1 => {
                    green.push(NumCast::from(pixel).unwrap());
                },
                2 => {
                    blue.push(NumCast::from(pixel).unwrap());
                },
                3 => {
                    alpha.push(NumCast::from(pixel).unwrap());
                },
                _ => {
                    unreachable!();
                }
            }
        });

        let h = height as usize;
        let w = width as usize;

        let red_dmat = DMatrix::<T>::from_iterator(h, w, red.into_iter());
        let green_dmat = DMatrix::<T>::from_iterator(h, w, green.into_iter());
        let blue_dmat = DMatrix::<T>::from_iterator(h, w, blue.into_iter());
        let alpha_dmat = DMatrix::<T>::from_iterator(h, w, alpha.into_iter());

        Self {
            width,
            height,
            channels: [red_dmat, green_dmat, blue_dmat, alpha_dmat]
        }
    }

    pub fn as_view(&self) -> DynamicImage {
        
        let mut img = RgbaImage::new(self.width, self.height);

        for (idx, [x, y, z, a]) in self.into_iter().enumerate() {

            let rgba = Rgba::<u8>([
                NumCast::from(x).unwrap(),
                NumCast::from(y).unwrap(),
                NumCast::from(z).unwrap(),
                NumCast::from(a).unwrap()
            ]);
            img.put_pixel(
                idx as u32 % self.width as u32,
                idx as u32 / self.width as u32,
                rgba
            );
        }
        DynamicImage::ImageRgba8(img)
    }

    pub fn from_dynamic(image: &DynamicImage) -> SimpleRGBAImage<T> {
        let rgb = image.to_rgba8();
        let raw = rgb.clone().into_raw();

        let width = rgb.width() as u32;
        let height = rgb.height() as u32;
        
        let mut red : Vec<T> = Vec::new();
        let mut green: Vec<T> = Vec::new();
        let mut blue: Vec<T> = Vec::new();
        let mut alpha: Vec<T> = Vec::new();        


        raw.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 4 {
                0 => {
                    red.push(NumCast::from(pixel).unwrap());
                },
                1 => {
                    green.push(NumCast::from(pixel).unwrap());
                },
                2 => {
                    blue.push(NumCast::from(pixel).unwrap());
                },
                3 => {
                    alpha.push(NumCast::from(pixel).unwrap());
                }
                _ => {
                    unreachable!();
                }
            }
        });

        let h = height as usize;
        let w = width as usize;

        let red_dmat = DMatrix::from_iterator(h, w, red.into_iter());
        let green_dmat = DMatrix::from_iterator(h, w, green.into_iter());
        let blue_dmat = DMatrix::from_iterator(h, w, blue.into_iter());
        let alpha_dmat = DMatrix::from_iterator(h, w, alpha.into_iter());

        Self {
            width,
            height,
            channels: [red_dmat, green_dmat, blue_dmat, alpha_dmat]
        }
    }

    fn red(&'a self) -> &'a DMatrix<T> {
        &self.channels[0]
    }

    fn green(&'a self) -> &'a DMatrix<T> {
        &self.channels[1]
    }

    fn blue(&'a self) -> &'a DMatrix<T> {
        &self.channels[2]
    }

    fn alpha(&'a self) -> &'a DMatrix<T> {
        &self.channels[3]
    }

    fn red_mut(&'a mut self) -> &'a mut DMatrix<T> {
        &mut self.channels[0]
    }

    fn green_mut(&'a mut self) -> &'a mut DMatrix<T> {
        &mut self.channels[1]
    }

    fn blue_mut(&'a mut self) -> &'a mut DMatrix<T> {
        &mut self.channels[2]
    }

    fn alpha_mut(&'a mut self) -> &'a mut DMatrix<T> {
        &mut self.channels[3]
    }
}

pub trait ApplyAcrossChannels<T: Num + NumCast + Clone + Send + Sync + Copy + Scalar> {
    fn apply_channels(&self, f: &dyn Fn(&DMatrix<T>) -> DMatrix<T>) -> SimpleRGBAImage<T>;
    fn apply_channels_mut(&mut self, f: &mut dyn FnMut(&mut DMatrix<T>));
}

impl<'a, T: Num + NumCast + Clone + Send + Sync + Copy + Scalar> ApplyAcrossChannels<T> for SimpleRGBAImage<T> {
    fn apply_channels(&self, f: &dyn Fn(&DMatrix<T>) -> DMatrix<T>) -> SimpleRGBAImage<T> {
        
        let new_red = f(self.red());
        let new_green = f(self.green());
        let new_blue = f(self.blue());
        let new_alpha = f(self.alpha());

        let (new_height, new_width) = new_red.shape();

        Self {
            width: new_width as u32,
            height: new_height as u32,
            channels: [new_red, new_green, new_blue, new_alpha]
        }
    }

    fn apply_channels_mut(&mut self, f: &mut dyn FnMut(&mut DMatrix<T>)) {

        f(self.red_mut());
        f(self.green_mut());
        f(self.blue_mut());
        f(self.alpha_mut());

        let (new_height, new_width) = self.red().shape();
        self.height = new_height as u32;
        self.width = new_width as u32;
    }
}