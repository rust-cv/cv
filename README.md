# akaze

Implementation of AKAZE based on the one by indianajohn. He gave me permission to copy this here and work from that.

This crate adds several optimizations (using ndarray) to the original implementation and integrates directly into the rust-cv ecosystem for ease-of-use. This crate does not currently use threading to speed anything up, but it might be added as a Cargo feature in the future.

The original implementation can be found here: https://github.com/pablofdezalc/akaze

The previous rust implementation can be found here: https://github.com/indianajohn/akaze-rust

The main site for the algorithm is normally [here](http://www.robesafe.com/personal/pablo.alcantarilla/kaze.html), but it was down, so I found another link to the paper: http://tulipp.eu/wp-content/uploads/2019/03/2017_TUD_HEART_kalms.pdf
