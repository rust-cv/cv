use image::{DynamicImage, RgbImage, Rgb, GenericImageView, ColorType};
use nalgebra::{DMatrix};
use imgshow::imgshow;
use itertools::izip;
use imgproc_rs::image::{Image as ImgProcImage};
use imgproc_rs::convert::{u8_to_f64_scale as img_proc_u8_to_f64_scale};


#[derive(Debug, Clone)]
pub struct SimpleRGBImageU8 {
    pub height: u32,
    pub width: u32,
    pub channels: [DMatrix<u8>; 3]
}

#[derive(Debug, Clone)]
pub struct SimpleRGBImageF64 {
    pub height: u32,
    pub width: u32,
    pub channels: [DMatrix<f64>; 3]
}

pub trait Shape {
    fn shape(&self) -> (u32, u32);
}

impl Shape for SimpleRGBImageF64 {
    fn shape(&self) -> (u32, u32) {
        (self.height, self.width)
    }
}
impl Shape for SimpleRGBImageU8 {
    fn shape(&self) -> (u32, u32) {
        (self.height, self.width)
    }
}

// pub trait AcrossChannelOperations {
//     fn apply_mut<F: Fn()>
// }
// pub trait ApplyAcrossChannels<T> {
//     fn apply_mut<F: Fn(usize, usize, T) -> T>(&mut self, f: F);
//     fn apply<F: Fn(usize, usize, T) -> T>(&self, f: F) -> Self;
// }


// impl ApplyAcrossChannels<u8> for SimpleRGBImageU8 {
//     fn apply_mut<F : Fn(usize, usize, u8) -> u8>(
//         &mut self, f: F
//     ) {

//         for (i, j) in izip!(0..self.height as usize, 0..self.width as usize) {
//             for k in 0..3 {
//                 self.channels[k][(i, j)] = f(i, j, self.channels[k][(i, j)]);
//             }
//         }

//     }
//     fn apply<F : Fn(usize, usize, u8) -> u8>(&self, f: F) -> SimpleRGBImageU8 {

//         let red_cp = self.red().apply(f)

//         return SimpleRGBImageU8 {
//             height: self.height,
//             width: self.width,
//             channels: [
//                 red_dmat,
//                 green_dmat,
//                 blue_dmat
//             ]
//         }
//     }
// }


// impl ApplyAcrossChannels<u8> for SimpleRGBImageF64 {
//     fn apply_mut<F : Fn(&DMatrix<f64>) -> DMatrix<f64>>(
//         &mut self, f: F
//     ) {
//         self.channels[0] = f(&self.channels[0]);
//         self.channels[1] = f(&self.channels[1]);
//         self.channels[2] = f(&self.channels[2]);
//     }
//     fn apply<F : Fn(&DMatrix<f64>) -> DMatrix<f64>>(&self, f: F) -> Self {
//         Self {
//             height: self.height,
//             width: self.width,
//             channels: [
//                 f(&self.channels[0]),
//                 f(&self.channels[1]),
//                 f(&self.channels[2])
//             ]
//         }
//     }
// }


impl IntoIterator for SimpleRGBImageU8 {
    type Item = (u8, u8, u8);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter{
        izip!(
            self.channels[0].transpose().iter(),
            self.channels[1].transpose().iter(),
            self.channels[2].transpose().iter()
        )
        .map(|(&a, &b, &c)| (a, b, c))
        .collect::<Vec<(u8, u8, u8)>>()
        .into_iter()
    }
}

impl IntoIterator for SimpleRGBImageF64 {
    type Item = (f64, f64, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        izip!(
            self.channels[0].transpose().iter(),
            self.channels[1].transpose().iter(),
            self.channels[2].transpose().iter()
        )
        .map(|(&a, &b, &c)| (a, b, c))
        .collect::<Vec<(f64, f64, f64)>>()
        .into_iter()
    }
}


impl<'a> IntoIterator for &'a SimpleRGBImageU8 {
    type Item = (u8, u8, u8);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        izip!(
            self.red().iter(),
            self.green().iter(),
            self.blue().iter()
        )
        .map(|(&a, &b, &c)| (a, b, c))
        .collect::<Vec<(u8, u8, u8)>>()
        .into_iter()
    }
}

impl<'a> IntoIterator for &'a SimpleRGBImageF64 {
    type Item = (f64, f64, f64);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        izip!(
            self.red().iter(),
            self.green().iter(),
            self.blue().iter()
        )
        .map(|(&a, &b, &c)| (a, b, c))
        .collect::<Vec<(f64, f64, f64)>>()
        .into_iter()
    }
}


impl<'a> SimpleRGBImageU8 {

    pub fn show(&self) {
        let temp = self.as_view();
        imgshow(&temp).unwrap();
    }

    pub fn from_slice(slice: &[u8], width: u32, height: u32) -> SimpleRGBImageU8 {
        
        let mut red : Vec<u8> = Vec::new();
        let mut green: Vec<u8> = Vec::new();
        let mut blue: Vec<u8> = Vec::new();
        
        slice.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 3 {
                0 => {
                    red.push(pixel);
                },
                1 => {
                    green.push(pixel);
                },
                2 => {
                    blue.push(pixel);
                },
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

        Self {
            width: width,
            height: height,
            channels: [red_dmat, green_dmat, blue_dmat]
        }
    }

    pub fn as_imgproc_image_u8(&self) -> ImgProcImage<u8> {

        let d_img = self.as_view();
        let (width, height) = d_img.dimensions();
        let (channels, alpha) = (3, false);

        ImgProcImage::<u8>::from_slice(width, height, channels, alpha, d_img.as_bytes())
    }

    pub fn as_imgproc_image_f64(&self) -> ImgProcImage<f64> {

        let d_img = self.as_view();
        let (width, height) = d_img.dimensions();
        let (channels, alpha) = (3, false);

        let raw_u8 = ImgProcImage::<u8>::from_slice(width, height, channels, alpha, d_img.as_bytes());

        img_proc_u8_to_f64_scale(&raw_u8, 255)
    }

    pub fn as_view(&self) -> DynamicImage {
        let mut img = RgbImage::new(self.width, self.height);

        for (idx, (x, y, z)) in self.into_iter().enumerate() {
            let rgb = Rgb([x, y, z]);
            img.put_pixel(
                idx as u32 % self.width as u32,
                idx as u32 / self.width as u32,
                rgb
            );
        }
        DynamicImage::ImageRgb8(img)
    }

    pub fn from_dynamic(image: &DynamicImage) -> SimpleRGBImageU8 {
        let rgb = image.to_rgb8();
        let raw = rgb.clone().into_raw();

        let width = rgb.width() as u32;
        let height = rgb.height() as u32;
        
        let mut red : Vec<u8> = Vec::new();
        let mut green: Vec<u8> = Vec::new();
        let mut blue: Vec<u8> = Vec::new();
        
        raw.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 3 {
                0 => {
                    red.push(pixel);
                },
                1 => {
                    green.push(pixel);
                },
                2 => {
                    blue.push(pixel);
                },
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

        Self {
            width: width,
            height: height,
            channels: [red_dmat, green_dmat, blue_dmat]
        }
    }

    fn red(&'a self) -> &'a DMatrix<u8> {
        &self.channels[0]
    }

    fn green(&'a self) -> &'a DMatrix<u8> {
        &self.channels[1]
    }

    fn blue(&'a self) -> &'a DMatrix<u8> {
        &self.channels[2]
    }

}


impl<'a> SimpleRGBImageF64 {


    pub fn show(&self) {
        let temp = self.as_view();
        imgshow(&temp).unwrap();
    }


    pub fn from_slice(slice: &[f64], width: u32, height: u32) -> SimpleRGBImageF64 {
        
        let mut red : Vec<f64> = Vec::new();
        let mut green: Vec<f64> = Vec::new();
        let mut blue: Vec<f64> = Vec::new();
        
        slice.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 3 {
                0 => {
                    red.push(pixel);
                },
                1 => {
                    green.push(pixel);
                },
                2 => {
                    blue.push(pixel);
                },
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

        Self {
            width: width,
            height: height,
            channels: [red_dmat, green_dmat, blue_dmat]
        }
    }

    pub fn as_view(&self) -> DynamicImage {
        let mut img = RgbImage::new(self.width, self.height);

        for (idx, (x, y, z)) in self.into_iter().enumerate() {
            let rgb = Rgb([
                (x * 255.99).round() as u8,
                (y * 255.99).round() as u8,
                (z * 255.99).round() as u8
            ]);
            img.put_pixel(
                idx as u32 % self.width as u32,
                idx as u32 / self.width as u32,
                rgb
            );
        }
        DynamicImage::ImageRgb8(img)
    }

    pub fn from_dynamic(image: &'a DynamicImage) -> SimpleRGBImageF64 {
        let rgb = image.to_rgb8();
        let raw = rgb.clone().into_raw();

        let width = rgb.width() as u32;
        let height = rgb.height() as u32;
        
        let mut red : Vec<f64> = Vec::new();
        let mut green: Vec<f64> = Vec::new();
        let mut blue: Vec<f64> = Vec::new();
        
        raw.iter().enumerate().for_each(|(idx, &pixel)| {
            match idx % 3 {
                0 => {
                    red.push(pixel as f64 / 255.0);
                },
                1 => {
                    green.push(pixel as f64 / 255.0);
                },
                2 => {
                    blue.push(pixel as f64 / 255.0);
                },
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

        Self {
            width: width,
            height: height,
            channels: [red_dmat, green_dmat, blue_dmat]
        }
    }
    pub fn red(&'a self) -> &'a DMatrix<f64> {
        &self.channels[0]
    }

    pub fn green(&'a self) -> &'a DMatrix<f64> {
        &self.channels[1]
    }

    pub fn blue(&'a self) -> &'a DMatrix<f64> {
        &self.channels[2]
    }
}

// trait Op {
//     pub fn apply(&self, image: &mut SimpleRGBImage<f32>) -> SimpleRGBImage<f32>;

// }




// #[derive(Debug)]
// pub struct SimpleRGBImageU8 {
//     pub height: u32,
//     pub width: u32,
//     pub channels: [DMatrix<u8>; 3]
// }

// #[derive(Debug)]
// pub struct SimpleRGBImageF64 {
//     pub height: u32,
//     pub width: u32,
//     pub channels: [DMatrix<f64>; 3]
// }


// // trait SimpleRGBImage {
// //     fn new(width: u32, height: u32) -> Self;
// //     fn from_dynamic(image: &DynamicImage) -> Self;
// //     fn shape(&self) -> (u32, u32);
// // }


// // impl SimpleRGBImage {
// //     pub fn shape(&self) -> (u32, u32) {
// //         (self.width, self.height)
// //     }
// // }


// impl SimpleRGBImageU8 {
//     pub fn new(width: u32, height: u32) -> SimpleRGBImageU8 {
//         Self {
//             width: width,
//             height: height,
//             channels: [
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0
//                 ),
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0
//                 )
//                 ,
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0
//                 )]
//         }
//     }

//     pub fn from_dynamic(image: &DynamicImage) -> SimpleRGBImageU8 {
//         let rgb = image.to_rgb8();
//         let raw = rgb.clone().into_raw();

//         let width = rgb.width() as u32;
//         let height = rgb.height() as u32;
        
//         let mut red : Vec<u8> = Vec::new();
//         let mut green: Vec<u8> = Vec::new();
//         let mut blue: Vec<u8> = Vec::new();
        
//         raw.iter().enumerate().for_each(|(idx, &pixel)| {
//             match idx % 3 {
//                 0 => {
//                     red.push(pixel);
//                 },
//                 1 => {
//                     green.push(pixel);
//                 },
//                 2 => {
//                     blue.push(pixel);
//                 },
//                 _ => {
//                     unreachable!();
//                 }
//             }
//         });

//         let h = height as usize;
//         let w = width as usize;

//         // DMatrix constructs column-first, but we need to inject row-first.
//         // So we transpose.

//         let red_dmat = DMatrix::from_iterator(h, w, red.into_iter()).transpose();
//         let green_dmat = DMatrix::from_iterator(h, w, green.into_iter()).transpose();
//         let blue_dmat = DMatrix::from_iterator(h, w, blue.into_iter()).transpose();

//         Self {
//             width: width,
//             height: height,
//             channels: [red_dmat, green_dmat, blue_dmat]
//         }
//     }
//     pub fn shape(&self) -> (u32, u32) {
//         (self.width, self.height)
//     }
// }

// impl SimpleRGBImageF64 {
//     pub fn new(width: u32, height: u32) -> SimpleRGBImageF64 {
//         Self {
//             width: width,
//             height: height,
//             channels: [
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0.0
//                 ),
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0.0
//                 )
//                 ,
//                 DMatrix::from_fn(
//                     height as usize,
//                     width as usize,
//                     |_, _| 0.0
//                 )
//             ]
//         }
//     }

//     pub fn from_dynamic(image: &DynamicImage) -> SimpleRGBImageF64 {
//         let rgb = image.to_rgb8();
//         let raw = rgb.clone().into_raw();

//         let width = rgb.width() as u32;
//         let height = rgb.height() as u32;
        
//         let mut red : Vec<f64> = Vec::new();
//         let mut green: Vec<f64> = Vec::new();
//         let mut blue: Vec<f64> = Vec::new();
        
//         raw.iter().enumerate().for_each(|(idx, &pixel)| {
//             match idx % 3 {
//                 0 => {
//                     red.push((pixel as f64) / 255.0);
//                 },
//                 1 => {
//                     green.push((pixel as f64) / 255.0);
//                 },
//                 2 => {
//                     blue.push((pixel as f64) / 255.0);
//                 },
//                 _ => {
//                     unreachable!();
//                 }
//             }
//         });

//         let h = height as usize;
//         let w = width as usize;

//         // DMatrix constructs column-first, but we need to inject row-first.
//         // So we transpose.

//         let red_dmat = DMatrix::from_iterator(h, w, red.into_iter()).transpose();
//         let green_dmat = DMatrix::from_iterator(h, w, green.into_iter()).transpose();
//         let blue_dmat = DMatrix::from_iterator(h, w, blue.into_iter()).transpose();

//         Self {
//             width: width,
//             height: height,
//             channels: [red_dmat, green_dmat, blue_dmat]
//         }
//     }
//     pub fn shape(&self) -> (u32, u32) {
//         (self.width, self.height)
//     }

//     pub fn show(&self) {

//     }

// }


// pub fn generate_base_image() {
//     let image = SimpleRGBImage::<u8>::new(100, 100);
//     println!("Img: {:?}", image);
// }