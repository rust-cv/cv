This document describes the proof describing the derivation of `cv_core::geom::reproject_along_translation`.

This function finds a translation from a 3d point that minimizes the reprojection error.

Here is a key and a diagram:

- `t` the translation vector
- `b` the `from` point
- `a` the `to` epipolar point
- `O` the optical center
- `@` the virtual image plane

```
     t<---b
         /
        /
@@@a@@@/@@@@@
   |  /
   | /
   |/
   O
```

Note that the direction of vector `t` is not relevant. The translation will be scaled to reduce the reprojection error, even if that means it reverses direction.

The way this is solved is by starting with the definition of the reprojection error in this case, and to get to that we must define the reprojection.

First, let `s` scale the translation.

The reprojection `p` can be defined as follows:

```
c = b + s * t
p = c.xy / c.z
```

This is because `c` is the 3d point after the translation and we can reproject it back to the virtual image plane by dividing the vector by its `z` component to set `z` to `1.0`. Since the `z` component can be discarded to retrieve the normalized image coordinate once it is on the virtual image plane, we don't take this intermediate step and simply divide the `x` and `y` components by `z` to achieve the same result.

Next, we must compute the `x` and `y` residual. Let `r` be the residual error vector for `x` and `y`.

```
r = a - p
```

This is simply defined as the distance from `p` to `a`. We want to minimize the norm of this vector, and ideally set it to zero.

```
r = <0, 0>
```

If we assume that the residual error can be reduced to zero (which may not be possible), then we can solve for `s` where `r = <0, 0>`. In the following we will derive `s` and add all priors:

```
c = b + s * t
p = c.xy / c.z
r = a - p
r = <0, 0>

a - p = <0, 0>
<a.x - p.x, a.y - p.y> = <0, 0>

a.x - p.x = 0
a.y - p.y = 0

p.x = (b.x + s * t.x) / (b.z + s * t.z)
p.y = (b.y + s * t.y) / (b.z + s * t.z)

0 = a.x - (b.x + s * t.x) / (b.z + s * t.z)
0 = a.y - (b.y + s * t.y) / (b.z + s * t.z)

a.x = (b.x + s * t.x) / (b.z + s * t.z)
a.y = (b.y + s * t.y) / (b.z + s * t.z)

(b.z + s * t.z) * a.x = (b.x + s * t.x)
(b.z + s * t.z) * a.y = (b.y + s * t.y)

(b.z + s * t.z) = (b.x + s * t.x) / a.x
(b.z + s * t.z) = (b.y + s * t.y) / a.y

(b.z + s * t.z) = (b.x + s * t.x) / a.x = (b.y + s * t.y) / a.y

(b.x + s * t.x) / a.x = (b.y + s * t.y) / a.y

a.y * (b.x + s * t.x) = a.x * (b.y + s * t.y)

a.y * b.x + a.y * s * t.x = a.x * b.y + a.x * s * t.y

a.y * b.x - a.x * b.y = a.x * s * t.y - a.y * s * t.x

a.y * b.x - a.x * b.y = s * (a.x * t.y - a.y * t.x)

s = (a.y * b.x - a.x * b.y) / (a.x * t.y - a.y * t.x)
```

Note that this process assumes nothing about the length of the translation vector `t`. It can be scaled regardless of its original length. Also note how, during the process of deriving `s`, `b.z` was eliminated. This is why the function does not take a `z` component. It is important to note that `b.xy` must come from a 3d point in camera space and not from a keypoint. The translation is what makes it unecessary to have the depth component.
