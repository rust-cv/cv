# cv-core

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl] ![LoC][lo]

[ci]: https://img.shields.io/crates/v/cv-core.svg
[cl]: https://crates.io/crates/cv-core/

[li]: https://img.shields.io/crates/l/specs.svg?maxAge=2592000

[di]: https://docs.rs/cv-core/badge.svg
[dl]: https://docs.rs/cv-core/

[lo]: https://tokei.rs/b1/github/rust-cv/cv-core?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam



Rust computer vision core crate

This crate contains core computer vision primitives that can be used in the Rust ecosystem
so that crates can talk to each other without glue logic. It is designed to maximize type
safety by encoding invariants in the type system such as original vs normalized keypoints.
