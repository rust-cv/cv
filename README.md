# cv

Batteries-included pure-Rust computer vision crate

Unlike other crates in the rust-cv ecosystem, this crate enables all features by-default as part of its batteries-included promise. The features can be turned off if desired by setting `default-features = false` for the package in Cargo.toml. However, it is recommended that if you only want specific rust-cv components that you add those components individually to better control dependency bloat and build times. This crate is useful for experimentation, but it is recommended that before you publish a crate or deploy a binary that you use the component crates and not `cv`. The `cv` crate itself provides no functionality, but only provides a useful organization of computer vision components.

This crate is `no_std` capable, but you must turn off some of the default enabled features to achieve this.
