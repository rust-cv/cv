use levenberg_marquardt::LeastSquaresProblem;
use cv_core::nalgebra::dimension::U4;

/// Currently a total of 16 points with X and Y angular error.
/// Each point has X, Y, and Z.
/// There is one pose which has 5 real degrees of freedom.
impl LeastSquaresProblem<f64, U32, U>