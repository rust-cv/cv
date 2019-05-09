use serde::{Serialize, Deserialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
enum Features {
    Akaze(vslam_akaze::Features),
}
