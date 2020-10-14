# Rust CV

[![Discord][dci]][dcl] [![Crates.io][ci]][cl] [![docs.rs][di]][dl] ![LoC][lo] ![Tests][btl] ![Lints][bll] ![Build][bbl]

[ci]: https://img.shields.io/crates/v/cv.svg
[cl]: https://crates.io/crates/cv/

[di]: https://docs.rs/cv/badge.svg
[dl]: https://docs.rs/cv/

[lo]: https://tokei.rs/b1/github/rust-cv/cv?category=code

[dci]: https://img.shields.io/discord/550706294311485440.svg?logo=discord&colorB=7289DA
[dcl]: https://discord.gg/d32jaam

[btl]: https://github.com/rust-cv/cv/workflows/unit%20tests/badge.svg
[bll]: https://github.com/rust-cv/cv/workflows/lints/badge.svg
[bbl]: https://github.com/rust-cv/cv/workflows/build/badge.svg

Rust CV is a project to implement computer vision algorithms, abstractions, and systems in Rust. `#[no_std]` is supported where possible.

## Documentation

Each crate has its own documentation, but the easiest way to check all of the documentation at once is to look at [the docs for the `cv` batteries-included crate](https://docs.rs/cv/).

Check out our tutorial book [here](https://rust-cv.github.io/tutorial/)! The book source for the tutorials can be found in the `tutorial` directory of the repository. The example code used in the tutorial can be found in the `tutorial-code` directory. The resources for tutorials can be found in the [site `res` directory](https://github.com/rust-cv/rust-cv.github.io/tree/master/res).

## About

This repository contains all computer vision crates for Rust CV in a mono-repo, including utilities as well as libraries. When updating libraries, all the crates in this repository should build for a PR to be accepted. Rust CV also maintains some other crates that are related to Computer Vision as well, which are located in the GitHub organization, not in this repository.

Each crate has its own associated license. Rust CV is comprised of different open source licenses, mostly MIT. See the crate directories (or their [crates.io](https://crates.io/) entries) for their individual licenses.

Each library was originally its own separate repository before being incorporated into the mono repo. The old repositories that are now in this repo are all archived, but still exist to find tagged versions, assocated commits, and issues. All new PRs should be made to this repository.

## What is computer vision

Many people are familiar with covolutional neural networks and machine learning (ML) in computer vision, but computer vision is much more than that. Computer vision broadly encompases image processing, photogrammetry, and pattern recognition. Machine learning can be used in all of these domains (e.g. denoisers, depth map prediction, and face detection), but it is not required. Almost all of the algorithms in this repository are not based on machine learning, but that does not mean you cannot use machine learning with these tools. Please take a look at https://www.arewelearningyet.com/ for Rust ML tools. We may expand into ML more in the future for tasks at which ML outperforms statistical algorithms.

## Goals

One of the first things that Rust CV focused on was algorithms in the domain of photogrammetry. Today, Rust now has enough photogrammetry algorithms to perform SfM and visual SLAM. Weakness still exists within image processing and pattern recognition domains.

Here are some of the domains of computer vision that Rust CV intends to persue along with examples of the domain (not all algorithms below live within the Rust CV organization, and some of these may exist and are unknown to us; some things may have changed since this was last updated):

* [ ] Image processing ([Wikipedia](https://en.wikipedia.org/wiki/Digital_image_processing))
  * [ ] Diffusion & blur
    * [x] [Gaussian blur](https://docs.rs/imageproc/0.20.0/imageproc/filter/fn.gaussian_blur_f32.html) ([Wikipedia](https://en.wikipedia.org/wiki/Gaussian_blur))
    * [ ] Fast Explicit Diffusion (FED) (implementation exists [within `akaze` crate](https://github.com/rust-cv/akaze/blob/a25ff0448d95600f10c69acb7e4f7d95045c1293/src/fed_tau.rs))
  * [ ] Contrast enhancement
    * [ ] Normalization ([Wikipedia](https://en.wikipedia.org/wiki/Normalization_(image_processing)))
    * [x] [Histogram equalization](https://docs.rs/imageproc/0.20.0/imageproc/contrast/fn.equalize_histogram.html) ([Wikipedia](https://en.wikipedia.org/wiki/Histogram_equalization))
  * [x] Edge detection ([Wikipedia](https://en.wikipedia.org/wiki/Edge_detection)) & gradient extraction ([Wikipedia](https://en.wikipedia.org/wiki/Image_derivatives))
    * [x] [Scharr filters](https://docs.rs/imageproc/0.20.0/imageproc/gradients/static.HORIZONTAL_SCHARR.html) (exists [within `akaze` crate](https://github.com/rust-cv/akaze/blob/a25ff0448d95600f10c69acb7e4f7d95045c1293/src/derivatives.rs))
    * [x] [Sobel filter](https://docs.rs/imageproc/0.20.0/imageproc/gradients/fn.horizontal_sobel.html) ([Wikipedia](https://en.wikipedia.org/wiki/Sobel_operator))
    * [x] [Canny edge detector](https://docs.rs/imageproc/0.20.0/imageproc/edges/fn.canny.html) ([Wikipedia](https://en.wikipedia.org/wiki/Canny_edge_detector))
  * [x] Perceptual hash ([Wikipedia](https://en.wikipedia.org/wiki/Perceptual_hashing))
    * [x] [aHash, dHash, and pHash](https://crates.io/crates/img_hash)
* [ ] Photogrammetry
  * [ ] Feature extraction ([Wikipedia](https://en.wikipedia.org/wiki/Feature_detection_(computer_vision)))
    * [x] [AKAZE](https://docs.rs/akaze/0.3.1/akaze/struct.Akaze.html)
    * [x] [FAST](https://docs.rs/imageproc/0.20.0/imageproc/corners/index.html) ([Wikipedia](https://en.wikipedia.org/wiki/Features_from_accelerated_segment_test))
    * [ ] ORB ([Wikipedia](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF))
    * [ ] SIFT ([Wikipedia](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform))
  * [ ] [Camera models and calibration](https://docs.rs/cv-core/0.10.0/cv_core/trait.CameraModel.html) (both from and to image coordinates from bearings)
    * [ ] Pinhole Camera ([Wikipedia](https://en.wikipedia.org/wiki/Pinhole_camera))
      * [x] [Skew, focals, and principle point](https://docs.rs/cv-pinhole/0.1.1/cv_pinhole/struct.CameraIntrinsics.html)
      * [ ] Kn radial distortion ([Wikipedia](https://en.wikipedia.org/wiki/Distortion_(optics)#Radial_distortion))
        * [x] [K1 radial distortion](https://docs.rs/cv-pinhole/0.1.1/cv_pinhole/struct.CameraIntrinsicsK1Distortion.html)
        * [ ] K1-K6 radial distortion
    * [ ] Fisheye Camera ([Wikipedia](https://en.wikipedia.org/wiki/Fisheye_lens))
      * [ ] Skew, focals, and principle point
      * [ ] K1-K4 fisheye distortion (same as OpenCV)
    * [ ] Equirectangular ([Wikipedia](https://en.wikipedia.org/wiki/Equirectangular_projection))
  * [ ] Matching ([Wikipedia](https://en.wikipedia.org/wiki/Point_feature_matching))
    * [x] Descriptor matching strategies
      * [x] [Brute force](https://docs.rs/space/0.10.3/space/fn.linear_knn.html) (for camera traking with binary features)
      * [x] [HNSW](https://docs.rs/hnsw/0.6.1/hnsw/struct.HNSW.html) (for loop closure)
    * [ ] Filtering strategies
      * [ ] Symmetric matching (exists [within cv-reconstruction](https://github.com/rust-cv/cv/blob/58444de1cb174622ac34cc705ab9142e081f412c/cv-reconstruction/src/lib.rs#L1337))
      * [ ] Lowe's ratio test matching
  * [ ] Geometric verification (utilized abstractions in [sample-consensus](https://docs.rs/sample-consensus/0.2.0/sample_consensus/))
    * [ ] [Consensus algorithms](https://docs.rs/sample-consensus/0.2.0/sample_consensus/trait.Consensus.html)
      * [x] [ARRSAC](https://docs.rs/arrsac/0.3.0/arrsac/struct.Arrsac.html)
      * [ ] AC-RANSAC
      * [ ] RANSAC ([Wikipedia](https://en.wikipedia.org/wiki/Random_sample_consensus))
    * [ ] [Estimation algorithms](https://docs.rs/sample-consensus/0.2.0/sample_consensus/trait.Estimator.html)
      * [x] P3P ([Wikipedia](https://en.wikipedia.org/wiki/Perspective-n-Point#P3P))
        * [x] [Lambda Twist](https://docs.rs/lambda-twist/0.2.0/lambda_twist/struct.LambdaTwist.html)
      * [x] Motion estimation ([Wikipedia](https://en.wikipedia.org/wiki/Motion_estimation))
        * [x] [Eight Point](https://docs.rs/eight-point/0.4.0/eight_point/struct.EightPoint.html) ([Wikipedia](https://en.wikipedia.org/wiki/Eight-point_algorithm))
        * [ ] [Nister-Stewenius](https://github.com/rust-cv/nister-stewenius/) (basically done, but not packaged up)
    * [ ] [Models](https://docs.rs/sample-consensus/0.2.0/sample_consensus/trait.Model.html)
      * [x] [Essential matrix](https://docs.rs/cv-core/0.10.0/cv_core/struct.EssentialMatrix.html) ([Wikipedia](https://en.wikipedia.org/wiki/Essential_matrix))
        * [x] With residual for [feature matches](https://docs.rs/cv-core/0.10.0/cv_core/struct.FeatureMatch.html)
      * [x] [Pose of world relative to camera](https://docs.rs/cv-core/0.10.0/cv_core/struct.WorldPose.html) ([Wikipedia](https://en.wikipedia.org/wiki/3D_pose_estimation))
        * [x] With residual for [feature to world matches](https://docs.rs/cv-core/0.10.0/cv_core/struct.FeatureWorldMatch.html)
      * [x] [Relative pose of camera](https://docs.rs/cv-core/0.10.0/cv_core/struct.RelativeCameraPose.html) ([Wikipedia](https://en.wikipedia.org/wiki/3D_pose_estimation))
        * [ ] With residual for [feature matches](https://docs.rs/cv-core/0.10.0/cv_core/struct.FeatureMatch.html)
      * [ ] Homography matrix ([Wikipedia](https://en.wikipedia.org/wiki/Homography_(computer_vision)))
        * [ ] With residual for [feature matches](https://docs.rs/cv-core/0.10.0/cv_core/struct.FeatureMatch.html)
      * [ ] Trifocal Tensor ([Wikipedia](https://en.wikipedia.org/wiki/Trifocal_tensor))
        * [ ] With residual for three-feature matches (not currently in cv-core, as there is no trifocal tensor yet)
    * [ ] [PnP](https://github.com/rust-cv/pnp) (estimation, outlier filtering, and optimization) (incomplete)
  * [ ] Image registration ([Wikipedia](https://en.wikipedia.org/wiki/Image_registration))
  * [ ] Real-time depth-map estimation (for direct visual odometry algorithms that require it) ([Wikipedia](https://en.wikipedia.org/wiki/Depth_map))
  * [ ] Visual concept detection (used for loop closure)
    * [ ] Bag of Visual Words (BoW, see [Wikpedia article](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision))
    * [ ] Second order occurence pooling (as per the paper "Higher-order Occurrence Pooling for Bags-of-Words: Visual Concept Detection")
    * [ ] Fisher vector encoding
    * [ ] Learned place recognition (also see pattern recognition domain below)
  * [ ] Reconstruction ([Wikipedia](https://en.wikipedia.org/wiki/3D_reconstruction))
    * [ ] Visibility graph ([Wikipedia](https://en.wikipedia.org/wiki/Visibility_graph))
    * [ ] Graph optimization
    * [ ] Loop closure ([Wikipedia](https://en.wikipedia.org/wiki/Simultaneous_localization_and_mapping#Loop_closure))
    * [ ] Exporting ([point cloud Wikipedia](https://en.wikipedia.org/wiki/Point_cloud))
      * [ ] To NVM file
      * [ ] To PLY file ([Wikipedia](https://en.wikipedia.org/wiki/PLY_(file_format)))
  * [ ] Post-reconstruction depth-map estimation (for reconstruction post-processing) ([Wikipedia](https://en.wikipedia.org/wiki/Depth_map))
  * [ ] Densification
    * [ ] Using more extracted features
      * [ ] Using curvature maximas (as per "VITAMIN-E: VIsual Tracking And MappINg with Extremely Dense Feature Points")
    * [ ] Using patch-match ([Wikipedia](https://en.wikipedia.org/wiki/PatchMatch)) and depth-map
    * [ ] Using depth-map only with edge detection
  * [ ] Meshing ([Wikipedia](https://en.wikipedia.org/wiki/Point_cloud#Conversion_to_3D_surfaces))
    * [ ] Delaunay triangulation
      * [ ] Filtered with NLTGV minimization (as per "VITAMIN-E: VIsual Tracking And MappINg with Extremely Dense Feature Points")
    * [ ] Poisson surface reconstruction ([Wikipedia](https://en.wikipedia.org/wiki/Poisson%27s_equation#Surface_reconstruction))
    * [ ] Surface refinement
    * [ ] Texturing ([related Wikipedia](https://en.wikipedia.org/wiki/Texture_mapping))
* [ ] Pattern recognition
  * [x] k-NN search
    * [x] [Brute force](https://docs.rs/space/0.10.3/space/fn.linear_knn.html)
    * [x] [HNSW](https://docs.rs/hnsw/0.6.1/hnsw/struct.HNSW.html)
    * [x] [FLANN](https://docs.rs/flann/0.1.0/flann/)
  * [ ] Face recognition ([Wikipedia](https://en.wikipedia.org/wiki/Facial_recognition_system))
  * [ ] Articulated body pose estimation ([Wikipedia](https://en.wikipedia.org/wiki/Articulated_body_pose_estimation))
  * [ ] Object recognition ([Wikipedia](https://en.wikipedia.org/wiki/Outline_of_object_recognition))
  * [ ] Place recognition (can assist in MVG)
  * [ ] Segmentation ([Wikipedia](https://en.wikipedia.org/wiki/Image_segmentation))
    * [ ] Semantic segmentation mapping (see [this blog post](https://www.jeremyjordan.me/semantic-segmentation/))

To support computer vision tooling, the following will be implemented:

* [ ] Point clouds ([Wikipedia](https://en.wikipedia.org/wiki/Point_cloud))
  * [ ] Display tool
    * [ ] LoD refinement ([Wikipedia](https://en.wikipedia.org/wiki/Level_of_detail))
      * [ ] Potree file format ([GitHub](https://github.com/potree/potree/))
    * [ ] PLY files ([Wikipedia](https://en.wikipedia.org/wiki/PLY_(file_format)))

## Credits

[TheiaSfM](https://github.com/sweeneychris/TheiaSfM) and all of its authors can be thanked as their abstractions are direct inspiration for this crate. In some cases, the names of some abstractions may be borrowed directly if they are consistent. You can find the TheiaSfM documentation [here](http://www.theia-sfm.org/api.html).

"[Past, Present, and Future of Simultaneous Localization And Mapping: Towards the Robust-Perception Age](https://arxiv.org/pdf/1606.05830.pdf)" is an excellent paper that compiles information about modern SLAM algorithms and papers.
