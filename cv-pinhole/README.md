# cv-pinhole

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/cv-pinhole.svg
[cl]: https://crates.io/crates/cv-pinhole/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/cv-pinhole/badge.svg
[dl]: https://docs.rs/cv-pinhole/

[lo]: https://tokei.rs/b1/github/rust-cv/cv-pinhole?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/cv-pinhole/workflows/unit%20tests/badge.svg
[bll]: https://github.com/rust-cv/cv-pinhole/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/cv-pinhole/workflows/no-std/badge.svg

Pinhole camera model for Rust CV

This crate seamlessly plugs into `cv-core` and provides pinhole camera models with and without distortion correction.
It can be used to convert image coordinates into real 3d direction vectors (called bearings) pointing towards where
the light came from that hit that pixel. It can also be used to convert backwards from the 3d back to the 2d
using the `uncalibrate` method from the `cv_core::CameraModel` trait.
