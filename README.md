# cv-core

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![no_std][bnl]

[ci]: https://img.shields.io/crates/v/cv-core.svg
[cl]: https://crates.io/crates/cv-core/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/cv-core/badge.svg
[dl]: https://docs.rs/cv-core/

[lo]: https://tokei.rs/b1/github/rust-cv/cv-core?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/cv-core/workflows/unit%20tests/badge.svg
[bll]: https://github.com/rust-cv/cv-core/workflows/lints/badge.svg
[bnl]: https://github.com/rust-cv/cv-core/workflows/no-std/badge.svg

This library provides common abstractions and types for computer vision (CV) in Rust.
All the crates in the rust-cv ecosystem that have or depend on CV types depend on this crate.
This includes things like camera model traits, bearings, poses, keypoints, etc. The crate is designed to
be very small so that it adds negligable build time. It pulls in some dependencies
that will probably be brought in by writing computer vision code normally.
The core concept is that all CV crates can work together with each other by using the
abstractions and types specified in this crate.

The crate is designed to work with `#![no_std]`, even without an allocator. `libm` is used
for all math algorithms that aren't present in `std`. Any code that doesn't need to be shared
across all CV crates should not belong in this repository. If there is a good reason to put
code that some crates may need into `cv-core`, it should be gated behind a feature.
