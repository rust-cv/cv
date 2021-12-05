## Doing descriptors extractions

In this tutorial we will be doing our second Rust-CV program. Our goal will be to run the Akaze extractor and display the result.

By oversimplifying quite a lot, features descriptors are points in an image along data that characterize them, that are in interest to us. Usually, we try to found points that have specific patterns. The idea is that each descriptor is quite unique so when comparing pictures if we found the same descriptors on 2 different pictures we can be pretty confident that we are seeing the same object.

If you want a more precise and accurate description of features, reading the [OpenCV documentation about features](https://docs.opencv.org/master/df/d54/tutorial_py_features_meaning.html) or Wikipedia [feature](https://en.wikipedia.org/wiki/Feature_(computer_vision)) or [descriptor](https://en.wikipedia.org/wiki/Visual_descriptor) pages is recommended.

### Project setup

As for the previous step, we will create a new binary project: ``cargo new --bin chapter2-first-program``

This time we will use the following as dependencies
```toml
[dependencies]
cv = { git = "https://github.com/rust-cv/cv.git" }
imgshow = { git = "https://github.com/rust-cv/cv.git" }
image = "0.23.7"
imageproc = "0.21.0"
```

### Opening the image

We open the image the same way we have done in chapter 2.

```rs
let src_image = 
    image::open("../res/0000000000.png").expect("failed to open image file");
```

### Running the Akaze feature extractor

To run the Akaze feature extractor we start by declaring a threshold. 

```rs
let threshold = 0.001f64;
```

This value act as a sensitivity adjustment for the feature detection. The higher the value the "stronger" the features are. This will also mean we will find less features. On the other hand if we set a very low threshold we will have a lot of values but features of poor quality. Features on smooth surfaces would start to be detected and this will not be helpful later on if we try do matching because of too many false positive.

Then, we create a `Akaze` struct which we initialize with the threshold previously declared. 

```rs
let akaze = Akaze::new(threshold);
```

Finally we call the `extract` method which gives us back features and their associated descriptors.

```rs
let (key_points, _descriptor) = akaze.extract(&src_image);
```

Then we add the features on the image and display it. The code is quite similar as what we have done in chapter 2.

```rs
let mut image = drawing::Blend(src_image.to_rgba());
for KeyPoint { point: (x, y), .. } in key_points {
    drawing::draw_cross_mut(&mut image, Rgba([0, 255, 255, 128]), x as i32, y as i32);
}
let out_imgage = DynamicImage::ImageRgba8(image.0);
imgshow::imgshow(&out_imgage);
```

### Result
If all went well you should have a window opened containing this pictures

![Akaze result](https://rust-cv.github.io/res/tutorial-images/akaze-result.png)

### Code

You can also find the whole project [here](https://github.com/rust-cv/cv/tree/master/tutorial-code/chapter3-akaze-feature-extraction). The dependencies of the code are sightly different in the repository, as it utilizes the dependencies within the mono-repo to ensure compatibility with master.


