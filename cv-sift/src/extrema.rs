use crate::sift::{KeyPoint, Descriptor};
use crate::image::{SimpleRGBAImage};

use num::{Num, NumCast};
use nalgebra::{Scalar, Matrix3, DMatrix};
use nalgebra::base::coordinates::{M3x3};
use itertools::Itertools;
use std::convert::TryInto;
use std::fmt;


#[derive(Clone, Eq)]
pub struct PixelSquare<T> where T: Scalar + Copy + PartialOrd + Num + NumCast {
    pub(crate) _inner: M3x3<T>,
    center_x: usize,
    center_y: usize,
    _inner_dmat: DMatrix<T>
}

impl<T> PartialEq for PixelSquare<T>
where T: Scalar + Copy + PartialOrd + Num + NumCast 
{
    fn eq(&self, other: &Self) -> bool {
        self.center_x.eq(&other.center_x) &&
        self.center_y.eq(&other.center_y) &&
        self._inner.eq(&other._inner)
    }
}


impl<T> fmt::Debug for PixelSquare<T>
where T: Scalar + Copy + PartialOrd + Num + NumCast
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PixelSquare")
        .field("m11", &self._inner.m11)
        .field("m12", &self._inner.m12)
        .field("m13", &self._inner.m13)
        .field("m21", &self._inner.m21)
        .field("m22", &self._inner.m22)
        .field("m23", &self._inner.m23)
        .field("m31", &self._inner.m21)
        .field("m32", &self._inner.m22)
        .field("m33", &self._inner.m23)
        .field("center_x", &self.center_x)
        .field("center_y", &self.center_y)
        .finish()
    }
}

impl<T> PartialOrd for PixelSquare<T>
where T: Scalar + Copy + PartialOrd + Num + NumCast
{
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.max().partial_cmp(&other.max())
    }
}

impl<T> PixelSquare<T> where T: Scalar + Copy + PartialOrd + Num + NumCast {

    pub fn new(img: DMatrix<T>, center_x: usize, center_y: usize) -> Self {

        let _inner = M3x3 {
            m11: *img.get((center_x - 1, center_y - 1)).unwrap(),
            m12: *img.get((center_x - 1, center_y)).unwrap(),
            m13: *img.get((center_x - 1, center_y + 1)).unwrap(),
            m21: *img.get((center_x, center_y - 1)).unwrap(),
            m22: *img.get((center_x, center_y)).unwrap(),
            m23: *img.get((center_x, center_y + 1)).unwrap(),
            m31: *img.get((center_x + 1, center_y - 1)).unwrap(),
            m32: *img.get((center_x + 1, center_y)).unwrap(),
            m33: *img.get((center_x + 1, center_y + 1)).unwrap(),
        };

        let _inner_dmat = img.clone();

        Self {
            _inner,
            center_y,
            center_x,
            _inner_dmat
        }
    }

    pub fn max(&self) -> T {
        let val = [
            self._inner.m11,
            self._inner.m12,
            self._inner.m13,
            self._inner.m21,
            self._inner.m22,
            self._inner.m23,
            self._inner.m31,
            self._inner.m32,
            self._inner.m33,
        ]
        .iter()
        .map(|x| x.to_f64().unwrap())
        .fold(-1./0., f64::max);

        NumCast::from(val).unwrap()
    }

    pub fn get(&self) -> T {
        self._inner.m22
    }

    pub fn step_up(&mut self) -> Result<(), std::io::ErrorKind> {

        if self.center_x <= 1 {
            return Err(std::io::ErrorKind::InvalidData);
        }

        self._inner.m31 = self._inner.m21;
        self._inner.m32 = self._inner.m22;
        self._inner.m33 = self._inner.m23;

        self._inner.m21 = self._inner.m11;
        self._inner.m22 = self._inner.m12;
        self._inner.m23 = self._inner.m13;


        match (
            self._inner_dmat.get((self.center_x - 2, self.center_y - 1)),
            self._inner_dmat.get((self.center_x - 2, self.center_y)),
            self._inner_dmat.get((self.center_x - 2, self.center_y + 1)),
        ) {
            (Some(top_left), Some(top_middle), Some(top_right)) => {
                self._inner.m11 = *top_left;
                self._inner.m12 = *top_middle;
                self._inner.m13 = *top_right;
                self.center_x -= 1;

                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }

    pub fn step_down(&mut self) -> Result<(), std::io::ErrorKind> {
        if self.center_x >= self._inner_dmat.shape().0 - 2 {
            return Err(std::io::ErrorKind::InvalidData);
        }

        self._inner.m11 = self._inner.m21;
        self._inner.m12 = self._inner.m22;
        self._inner.m13 = self._inner.m23;

        self._inner.m21 = self._inner.m31;
        self._inner.m22 = self._inner.m32;
        self._inner.m23 = self._inner.m33;


        match (
            self._inner_dmat.get((self.center_x + 2, self.center_y - 1)),
            self._inner_dmat.get((self.center_x + 2, self.center_y)),
            self._inner_dmat.get((self.center_x + 2, self.center_y + 1)),
        ) {
            (Some(bottom_left), Some(bottom_middle), Some(bottom_right)) => {
                self._inner.m31 = *bottom_left;
                self._inner.m32 = *bottom_middle;
                self._inner.m33 = *bottom_right;
                self.center_x += 1;

                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }

    pub fn step_right(&mut self) -> Result<(), std::io::ErrorKind> {

        if self.center_y >= ((self._inner_dmat.shape().1) - 2) {
            return Err(std::io::ErrorKind::InvalidData);
        }

        self._inner.m11 = self._inner.m12;
        self._inner.m21 = self._inner.m22;
        self._inner.m31 = self._inner.m32;

        self._inner.m12 = self._inner.m13;
        self._inner.m22 = self._inner.m23;
        self._inner.m32 = self._inner.m33;


        match (
            self._inner_dmat.get((self.center_x - 1, self.center_y + 2)),
            self._inner_dmat.get((self.center_x, self.center_y + 2)),
            self._inner_dmat.get((self.center_x + 1, self.center_y + 2)),
        ) {
            (Some(right_top), Some(right_middle), Some(right_bottom)) => {
                self._inner.m13 = *right_top;
                self._inner.m23 = *right_middle;
                self._inner.m33 = *right_bottom;
                self.center_y += 1;

                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
        
    }
    pub fn step_left(&mut self) -> Result<(), std::io::ErrorKind> {
        if self.center_y <= 1 {
            return Err(std::io::ErrorKind::InvalidData);
        }

        self._inner.m13 = self._inner.m12;
        self._inner.m23 = self._inner.m22;
        self._inner.m33 = self._inner.m32;

        self._inner.m12 = self._inner.m11;
        self._inner.m22 = self._inner.m21;
        self._inner.m32 = self._inner.m31;


        match (
            self._inner_dmat.get((self.center_x - 1, self.center_y - 2)),
            self._inner_dmat.get((self.center_x, self.center_y - 2)),
            self._inner_dmat.get((self.center_x + 1, self.center_y - 2)),
        ) {
            (Some(left_top), Some(left_middle), Some(left_bottom)) => {
                self._inner.m11 = *left_top;
                self._inner.m21 = *left_middle;
                self._inner.m31 = *left_bottom;
                self.center_y -= 1;

                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }
}




#[derive(Clone, Debug)]
pub struct PixelCube<T> where T: Num + NumCast + Sync + Send + Copy + Scalar + PartialOrd + Num + NumCast {
    pub top: PixelSquare<T>,
    pub middle: PixelSquare<T>,
    pub bottom: PixelSquare<T>,
    rows: usize,
    cols: usize
}


impl<T> PixelCube<T> where T: Num + NumCast + Sync + Send + Copy + Scalar + PartialOrd + Num + NumCast {
    pub fn new(imgs: &[DMatrix<T>; 3]) -> Self {

        let height = imgs.iter().map(|img| img.shape().0).min().unwrap();
        let width = imgs.iter().map(|img| img.shape().1).min().unwrap();

        Self {
            top: PixelSquare::<T>::new(imgs[0].clone(), 1, 1),
            middle: PixelSquare::<T>::new(imgs[1].clone(), 1, 1),
            bottom: PixelSquare::<T>::new(imgs[2].clone(), 1, 1),
            rows: height as usize,
            cols: width as usize,
        }
    }
    pub fn step_right(&mut self) -> Result<(), std::io::ErrorKind> {
        match (self.top.step_right(), self.middle.step_right(), self.bottom.step_right()) {
            (Ok(()), Ok(()), Ok(())) => {
                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }
    pub fn step_left(&mut self) -> Result<(), std::io::ErrorKind> {
        match (self.top.step_left(), self.middle.step_left(), self.bottom.step_left()) {
            (Ok(()), Ok(()), Ok(())) => {
                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }
    pub fn step_down(&mut self) -> Result<(), std::io::ErrorKind> {
        match (self.top.step_down(), self.middle.step_down(), self.bottom.step_down()) {
            (Ok(()), Ok(()), Ok(())) => {
                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }
    pub fn step_up(&mut self) -> Result<(), std::io::ErrorKind> {
        match (self.top.step_up(), self.middle.step_up(), self.bottom.step_up()) {
            (Ok(()), Ok(()), Ok(())) => {
                return Ok(())
            }
            _ => return Err(std::io::ErrorKind::InvalidData)
        }
    }

    pub fn is_extremum(&self) -> bool {
        let tx = self.top.max();
        let mx = self.middle.max();
        let bx = self.bottom.max();

        mx >= tx && mx >= bx && mx == self.middle.get()
    }
    pub fn get(&self) -> T {
        self.middle.get()
    }
}


pub struct PixelCubeIterator<T>
where T: Num + NumCast + Sync + Send + Copy + Scalar + PartialOrd + Num + NumCast
{
    rows: usize,
    cols: usize,
    pub(crate) _inner: PixelCube<T>,
    cx: usize,
    cy: usize
}


impl<T> PixelCubeIterator<T>
where T: Num + NumCast + Sync + Send + Copy + Scalar + PartialOrd + Num + NumCast
{
    pub fn new(imgs: &[DMatrix<T>; 3]) -> Self {

        let rows = imgs.iter().map(|img| img.shape().0).min().unwrap();
        let cols = imgs.iter().map(|img| img.shape().1).min().unwrap();

        Self {
            rows,
            cols,
            _inner: PixelCube::<T>::new(imgs),
            cx: 1,
            cy: 1
        }
    }
}


impl<T> Iterator for PixelCubeIterator<T>
where T: Num + NumCast + Sync + Send + Copy + Scalar + PartialOrd + Num + NumCast
{
    type Item = PixelCube<T>;


    fn next(&mut self) -> Option<Self::Item> {
        if self.cx == self.rows - 2 && self.cy == self.cols - 2 {
            return None;
        }
        if self.cy < self.cols - 2 {
            self._inner.step_right();
            self.cy += 1;
            return Some(self._inner.clone());
        // TODO: Maybe have a .goto(cx, cy) operation on the cube?   
        } else if self.cy == self.cols - 2 {
            self._inner.step_down();
            self.cx += 1;
            self.cy = 0;
        }
    }

}



// #[derive(Debug, Clone)]
// pub struct PixelCubeIterator<'a> {
//     pub triple: GrayTriple<'a>,
//     pub height: usize,
//     pub width: usize,
//     pub border_width: usize,
//     pub max_size: usize,
//     row_idx: usize,
//     col_idx: usize,
//     row_low: usize,
//     col_low: usize,
//     row_high: usize,
//     col_high: usize,
//     ptrs: Option<[*mut f64; 27]>
// }

// impl<'a> PixelCubeIterator<'a> {

//     const directions: [i8; 3] = [-1, 0, 1];

//     fn initialize_ptrs(&mut self) {
//         let (cx, cy):(usize, usize) = (self.border_width, self.border_width);
//         let mut data_slices: Vec<*mut f64> = vec![];
//         for slice in [self.triple.front, self.triple.middle, self.triple.back] {
//             let mut layer: Vec<*mut f64> = vec![];
//             for (&dx, &dy) in GrayTriple::directions.iter().cartesian_product(GrayTriple::directions.iter()){
//                 let x = (cx as i32 + dx as i32) as usize;
//                 let y = (cy as i32 + dy as i32) as usize;
//                 data_slices.push(&slice[(x, y)] as *mut f64);
//             }
//         }
//         assert_eq!(data_slices.len(), 27);
//         self.ptrs = Some(data_slices.try_into().unwrap());
//     }

//     fn reset_ptrs(&mut self) {

//     }

//     fn step_ptrs(&mut self) {
//         for ptr in self.ptrs.iter_mut() {
//             println!("{:?}", ptr);
//         }
//     }

//     pub fn step(&mut self) -> Option<()>{
//         self.col_idx += 1;


//         if self.col_idx >= self.col_high {
//             self.col_idx = self.col_low;
//             self.row_idx += 1;
//             self.reset_ptrs();
//         }

//         if self.row_idx >= self.row_high {
//             return None;
//         }
//         Some(())
//     }
// }

// impl<'a> Iterator for PixelCubeIterator<'a> {
//     type Item = PixelCube;

//     fn next(&mut self) -> Option<Self::Item> {
//         if self.row_idx == self.border_width {
//             self.initialize_ptrs();
//         }


//         None
//     }

// }

// #[derive(Debug, Clone, Copy)]
// pub struct GrayTriple<'a> {
//     pub front: &'a DMatrix<f64>,
//     pub middle: &'a DMatrix<f64>,
//     pub back: &'a DMatrix<f64>,
//     pub height: usize,
//     pub width: usize,
//     pub border_width: usize
// }

// impl<'a> GrayTriple<'a>{
//     pub fn new(
//         front: &'a DMatrix<f64>,
//         middle: &'a DMatrix<f64>,
//         back: &'a DMatrix<f64>,
//         height: usize,
//         width: usize,
//         border_width: usize
//     ) -> Self {
//         Self {
//             front,
//             middle,
//             back,
//             height,
//             width,
//             border_width
//         }
//     }

//     const directions: [i8; 3] = [-1, 0, 1];

//     // pub fn at(&self, i: usize, j: usize)-> PixelCube {

//     //     let mut data_slices: Vec<[f64; 9]> = vec![];

//     //     for slice in [self.front, self.middle, self.back] {
//     //         let mut found_slice: Vec<f64> = vec![];
//     //         for (&dx, &dy) in GrayTriple::directions.iter().cartesian_product(GrayTriple::directions.iter()){
//     //             let x = (i as i32 + dx as i32) as usize;
//     //             let y = (j as i32 + dy as i32) as usize;
//     //             found_slice.push(slice[(x, y)]);
//     //         }
//     //         data_slices.push(found_slice.try_into().unwrap());
//     //     }

//     //     PixelCube::from_slices(&data_slices[0], &data_slices[1], &data_slices[2])
//     // }
// }

// impl<'a> IntoIterator for GrayTriple<'a> {
//     type Item = PixelCube;
//     type IntoIter = PixelCubeIterator<'a>;

//     fn into_iter(self) -> Self::IntoIter {
//         PixelCubeIterator {
//             triple: self,
//             row_idx: 0,
//             col_idx: 0,
//             border_width: self.border_width,
//             width: self.width,
//             height: self.height,
//             max_size: (self.height - 2 * self.border_width) as usize * (self.width - 2 * self.border_width) as usize,
//             row_low: self.border_width,
//             col_low: self.border_width,
//             row_high: self.height - self.border_width,
//             col_high: self.width - self.border_width,
//             ptrs: None
//         }.into_iter()
//     }
// }
// impl<'a> Iterator for ImageTriple<'a> {
//     type Item = PixelCube<'a>;

//     fn next(&mut self) -> Option<Self::Item> {
//         None
//     }
        
// }

// impl ImageTriple<'_, f64> {
//     pub fn from_image(image: SimpleRGBAImage<f64>) -> ImageTriple<f64> {

//     }
// }