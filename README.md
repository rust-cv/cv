# nister-stewenius

Essential matrix estimation from 5 normalized image coordinate correspondences from the paper "Recent developments on direct relative orientation"

This crate implements the `Estimator` trait from the `sample-consensus` crate. This allows integration with sample consensus algorithms built on that crate. The model returned by this crate is an essential matrix which is estimated from 5 points. On each estimation, up to `10` solutions may be returned. It is recommended to use the `arrsac` crate on your data with the estimator in this crate to get the best essential matrix for your data.

## Credit

The credit for implementation of this algorithm mostly goes to Chris Sweeney (cmsweeney@cs.ucsb.edu). Although he did not write this crate, he is added as an author because his code from TheiaSfM was translated to Rust. The original copywrite headers are retained and this project is specially put under the BSD license.
