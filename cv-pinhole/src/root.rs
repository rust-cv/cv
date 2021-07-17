use cv_core::nalgebra::{Matrix2, Vector2};

/// Find function root
///
/// # Method
///
/// The default implementation uses a Newton-Bisection hybrid method based on [^1]
/// that is guaranteed to converge to almost machine precision.
///
/// # Resources
///
/// [^1]: Numerical Recipes 2nd edition. p. 365
///
/// <https://github.com/osveliz/numerical-veliz/blob/master/src/rootfinding/NewtSafe.adb>
///
/// # Panics
///
/// Panics when $f(a) ⋅ f(b) > 0$.
///
pub(crate) fn root<F>(func: F, a: f64, b: f64) -> f64
where
    F: Fn(f64) -> (f64, f64),
{
    let (mut xl, mut xh) = (a, b);
    let (fl, _) = func(xl);
    if fl == 0.0 {
        return xl;
    }
    let (fh, _) = func(xh);
    if fh == 0.0 {
        return xh;
    }
    if fl * fh > 0.0 {
        panic!("Inverse outside of bracket [a, b].");
    }
    if fl > 0.0 {
        std::mem::swap(&mut xl, &mut xh);
    }
    let mut rts = 0.5 * (xl + xh);
    let mut dxold = (xl - xh).abs();
    let mut dx = dxold;
    let (mut f, mut df) = func(rts);
    loop {
        if (((rts - xh) * df - f) * ((rts - xl) * df - f) > 0.0)
            || (2.0 * f.abs() > (dxold * df).abs())
        {
            // Bisection step
            dxold = dx;
            dx = 0.5 * (xh - xl);
            rts = xl + dx;
            if xl == rts || xh == rts {
                return rts;
            }
        } else {
            // Newton step
            dxold = dx;
            dx = f / df;
            let tmp = rts;
            rts -= dx;
            if tmp == rts {
                return rts;
            }
        }
        let (nf, ndf) = func(rts);
        f = nf;
        df = ndf;
        if f < 0.0 {
            xl = rts;
        } else {
            xh = rts;
        }
    }
}

pub(crate) fn newton2<F>(func: F, initial: Vector2<f64>) -> Option<Vector2<f64>>
where
    F: Fn(Vector2<f64>) -> (Vector2<f64>, Matrix2<f64>),
{
    // Newton-Raphson iteration
    const MAX_ITER: usize = 10;
    let mut x = initial;
    let mut last_delta_norm = f64::MAX;
    for iter in 0.. {
        let (f, j) = func(x);
        let delta = j.lu().solve(&f).unwrap();
        let delta_norm = delta.norm_squared();
        x -= delta;
        if delta_norm < x.norm_squared() * f64::EPSILON * f64::EPSILON {
            // Converged to epsilon
            break;
        }
        if delta_norm >= last_delta_norm {
            // No progress
            if delta_norm < 100.0 * x.norm_squared() * f64::EPSILON * f64::EPSILON {
                // Still useful
                break;
            } else {
                // Divergence
                return None;
            }
        }
        last_delta_norm = delta_norm;
        if iter >= MAX_ITER {
            // No convergence
            return None;
        }
    }
    Some(x)
}
