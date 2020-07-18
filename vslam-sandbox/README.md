# vslam-sandbox

A sandbox for integrating upstream vslam algorithms

## Goal

This sandbox should allow using and benchmarking the available vSLAM algorithms using the `vslam` crate as an abstraction to allow swapping out different components.

## Building

Please prepend `RUSTFLAGS="-C target-cpu=native"` to your cargo commands to run this with
native optimizations. Rust can perform some autovectorization via LLVM, but it needs to be
told that its okay that it only runs on your system. There are also some dependencies that
explicitly use the best available SIMD instructions when they are available, which they
aren't by default. The above environment variable will fix that and allow it to use AVX-512
or AVX2 depending on what is available.
