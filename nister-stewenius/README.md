# nister-stewenius

Essential matrix estimation from 5 normalized image coordinate correspondences from the paper "Recent developments on direct relative orientation"

This crate implements the `Estimator` trait from the `sample-consensus` crate. This allows integration with sample consensus algorithms built on that crate. The model returned by this crate is an essential matrix which is estimated from 5 points. On each estimation, up to `10` solutions may be returned. It is recommended to use the `arrsac` crate on your data with the estimator in this crate to get the best essential matrix for your data.

## Testing

Note that when `cargo test` is ran, since this crate builds by default with `no_std`, it spits out some errors when trying to run the doc tests:
```
error: no global memory allocator found but one is required; link to std or add `#[global_allocator]` to a static item that implements the GlobalAlloc trait.

error: aborting due to previous error
```

This error does not cause the unit tests to fail, nor does cargo return an error code. It can be annoying, but simply scroll up to see the unit test results. This happens due to https://github.com/rust-lang/rust/issues/54010.
