# akaze

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] ![MIT/Apache][li] [![docs.rs][di]][dl]

[ci]: https://img.shields.io/crates/v/akaze.svg
[cl]: https://crates.io/crates/akaze/

[li]: https://img.shields.io/badge/License-MIT-yellow.svg

[di]: https://docs.rs/akaze/badge.svg
[dl]: https://docs.rs/akaze/

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

AKAZE feature extraction algorithm for computer vision

Implementation of AKAZE based on the one by indianajohn. He gave me permission to copy this here and work from that, as his job conflicts with maintainership. The crate is greatly changed from the original.

See `tests/estimate_pose.rs` for a demonstration on how to use this crate.

This crate adds several optimizations (using ndarray) to the original implementation and integrates directly into the rust-cv ecosystem for ease-of-use. This crate does not currently use threading to speed anything up, but it might be added as a Cargo feature in the future.

The original implementation can be found here: <https://github.com/pablofdezalc/akaze>

The previous rust implementation can be found here: <https://github.com/indianajohn/akaze-rust>

The main site for the algorithm is normally [here](http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html), but it was down, so I found another link to the paper: <http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf>
