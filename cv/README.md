# cv

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl]

[ci]: https://img.shields.io/crates/v/cv.svg
[cl]: https://crates.io/crates/cv/

[li]: https://img.shields.io/badge/License-MIT-yellow.svg

[di]: https://docs.rs/cv/badge.svg
[dl]: https://docs.rs/cv/

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

Batteries-included pure-Rust computer vision crate

Unlike other crates in the rust-cv ecosystem, this crate enables all features by-default as part of its batteries-included promise. The features can be turned off if desired by setting `default-features = false` for the package in Cargo.toml. However, it is recommended that if you only want specific rust-cv components that you add those components individually to better control dependency bloat and build times. This crate is useful for experimentation, but it is recommended that before you publish a crate or deploy a binary that you use the component crates and not `cv`. The `cv` crate itself provides no functionality, but only provides a useful organization of computer vision components.

This crate is `no_std` capable, but you must turn off some of the default enabled features to achieve this.
