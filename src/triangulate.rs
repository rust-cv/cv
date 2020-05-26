use cv_core::nalgebra::{IsometryMatrix3, Matrix3x4, Matrix4, Point3};
use cv_core::{Bearing, CameraPose, UnscaledRelativeCameraPose};
use cv_pinhole::NormalizedKeyPoint;

/// This solves for the 3d point that minimizes the reprojection error
pub fn make_triangulate_least_square_reprojection_error<B, I>(
    epsilon: f64,
    max_iterations: usize,
) -> impl Fn(I) -> Option<Point3<f64>>
where
    B: Bearing,
    I: Iterator<Item = (CameraPose, B)>,
{
    move |pairs| {
        let mut a: Matrix4<f64> = cv_core::nalgebra::zero();

        for (pose, bearing) in pairs {
            // Get the normalized bearing.
            let bearing = bearing.bearing().into_inner();
            // Get the pose as a 3x4 matrix.
            let rot = pose.rotation.matrix();
            let trans = pose.translation.vector;
            let pose = Matrix3x4::<f64>::from_columns(&[
                rot.column(0),
                rot.column(1),
                rot.column(2),
                trans.column(0),
            ]);
            // Set up the least squares problem.
            let term = pose - bearing * bearing.transpose() * pose;
            a += term.transpose() * term;
        }

        let se = a.try_symmetric_eigen(epsilon, max_iterations)?;

        se.eigenvalues
            .iter()
            .enumerate()
            .min_by_key(|&(_, &n)| float_ord::FloatOrd(n))
            .map(|(ix, _)| se.eigenvectors.column(ix).into_owned())
            .map(|h| (h.xyz() / h.w).into())
    }
}

/// This solves for the 3d point that minimizes the reprojection error
///
/// This uses some defaults to simplify usage.
pub fn triangulate_least_square_reprojection_error<B>(
    pairs: impl Iterator<Item = (CameraPose, B)>,
) -> Option<Point3<f64>>
where
    B: Bearing,
{
    make_triangulate_least_square_reprojection_error(1e-9, 100)(pairs)
}

pub fn triangulator(
    pose: UnscaledRelativeCameraPose,
    a: NormalizedKeyPoint,
    b: NormalizedKeyPoint,
) -> Option<Point3<f64>> {
    triangulate_least_square_reprojection_error(
        std::iter::once((CameraPose(IsometryMatrix3::identity()), a))
            .chain(std::iter::once((CameraPose((pose.0).0), b))),
    )
}
