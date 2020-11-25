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
