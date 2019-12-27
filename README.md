# cv-core
Rust computer vision core crate

This crate contains core computer vision primitives that can be used in the Rust ecosystem
so that crates can talk to each other without glue logic. It is designed to maximize type
safety by encoding invariants in the type system such as original vs normalized keypoints.
