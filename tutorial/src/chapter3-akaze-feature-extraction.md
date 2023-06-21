# Feature extraction

In this chapter, we will be writing our second Rust-CV program. Our goal will be to run the AKAZE extractor and display the result.

## What is an image feature?

Features are comprised of a location in an image (called a "keypoint") and some data that helps characterize visual information about the feature (called a "descriptor"). We typically try to find features on images which can be matched to each other most easily. We want each feature to be visually discriminative and distinct so that we can find similar-looking features in other images without getting false-positives.

For the purposes of Multiple-View Geometry (MVG), which encompasses Structure from Motion (SfM) and visual Simultaneous Localization and Mapping (vSLAM), we typically wish to erase certain information from the descriptor of the feature. Specifically, we want features to match so long as they correspond to the same "landmark". A landmark is a visually distinct 3d point. Since the position and orientation (known as "pose") of the camera will be different in different images, the symmetric perspective distortion, in-plane rotation, lighting, and scale of the feature might be different between different frames. For simplicity, all of these terms mean that a feature looks different when you look at it from different perspectives and at different ISO, exposure, or lighting conditions. Due to this, we typically want to erase this variable information as much as possible so that our algorithms see two features of the same landmark as the same despite these differences.

If you want a more precise and accurate description of features, reading the [OpenCV documentation about features](https://docs.opencv.org/master/df/d54/tutorial_py_features_meaning.html) or Wikipedia [feature](https://en.wikipedia.org/wiki/Feature_(computer_vision)) or [descriptor](https://en.wikipedia.org/wiki/Visual_descriptor) pages is recommended.

## What is AKAZE and how does it work?

In this tutorial, we use the feature extraction algoirthm [AKAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf). AKAZE attempts to be invariant to rotation, lighting, and scale. It does not attempt to be invariant towards skew, which can happen if we are looking at a surface from a large angle of incidence. AKAZE occupies a "sweet spot" among feature extraction algorithms. For one, AKAZE feature detection can pick up the interior of corners, corners, points, and blobs. AKAZE's feature matching is very nearly as robust as SIFT, the gold standard for robustness in this class of algorithm, and it often exceeds the performance of SIFT. In terms of speed, AKAZE is significantly faster than SIFT to compute, and it can be sped up substantially with the use of GPUs or [even FPGAs](http://tulipp.eu/wp-content/uploads/2019/03/2017_TUD_HEART_kalms.pdf). However, the other gold standard in this class of algorithms is [ORB](https://en.wikipedia.org/wiki/Oriented_FAST_and_rotated_BRIEF) (and the closely related [FREAK](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.446.5816&rep=rep1&type=pdf), which can perform better than ORB). This algorithm targets speed, and it is roughly 5x faster than AKAZE, although this is implementation-dependent. One downside is that ORB is significantly less robust than AKAZE. Due to these considerations, AKAZE can, given the correct algorithms and computing power, meet the needs of real-time operation while also having high quality output.

AKAZE picks up features by looking at the approximation of the determinant of the hessian over the luminosity to perform a [second partial derivative test](https://en.wikipedia.org/wiki/Second_partial_derivative_test). This determinant is called the response in this context. Any local maxima in the response greater than a threshold is detected as a keypoint, and sub-pixel positioning is performed to extract the precise location of the keypoint. The threshold is typically above 0 and less than or equal to 0.01. This is done at several different scales, and at each of those scales the image is selectively blurred and occasionally shrunk. In the process of extracting the determinant of the hessian, it extracts first and second order gradients of the luminosity across X and Y in the image. By using the scale and rotation of the feature, we determine a scale and orientation at which to sample the descriptor from. The descriptor is extracted by making a series of binary comparisons in the luminosity, the first order gradients of the luminosity, and the second order gradients of the luminosity. Each binary comparison results in a single bit, and that bit is literally stored as a bit on the computer. In total, 486 comparisons are performed, thus AKAZE has a 486-bit "binary feature descriptor". For convenience, this can be padded with zeros to become 512-bit.

## Running the program

Make sure you are in the Rust CV mono-repo and run:

```bash
cargo run --release --bin chapter3-akaze-feature-extraction
```

If all went well you should have a window and see this:

![Akaze result](https://rust-cv.github.io/res/tutorial-images/akaze-result.png)

## The code

### Open the image

```rust
    let src_image = image::open("res/0000000000.png").expect("failed to open image file");
```

We saw this in [chapter 2](./chapter2-first-program.md). This will open the image. Make sure you run this from the correct location.

### Create an AKAZE feature extractor

```rust
    let akaze = Akaze::default();
```

For the purposes of this tutorial, we will just use the default AKAZE settings. You can modify the settings at this point by changing `Akaze::default()` to `Akaze::sparse()` or `Akaze::dense()`. It also has other settings you can modify as well.

### Extract the AKAZE features

```rust
    let (key_points, _descriptors) = akaze.extract(&src_image);
```

This line extacts the features from the image. In this case, we will not be using the descriptor data, so those are discarded with the `_`.

### Draw crosses and show image

```rust
    for KeyPoint { point: (x, y), .. } in key_points {
        drawing::draw_cross_mut(
            &mut image_canvas,
            Rgba([0, 255, 255, 128]),
            x as i32,
            y as i32,
        );
    }
```

Almost all of the rest of the code is the same as what we saw in [chapter 2](./chapter2-first-program.md). However, the above snippet is slightly different. Rather than randomly generating points, we are now using the X and Y components of the keypoints AKAZE extracted. The output image actually shows the keypoints of the features AKAZE found.

## End

This is the end of this chapter.

