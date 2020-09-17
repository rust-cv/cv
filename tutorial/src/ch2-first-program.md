## Your first Rust-CV program

In this section, we will be doing our first program. It will be very basic and not even really related to computer vision. We will just be loading an image, drawing random points on it and displaying it. This will be very basic but at least you can make sure you have everything set up correctly before attacking more ambitious problems.

### Creating the project

In order to start, we will be creating a new project using `cargo`. Let's run ``cargo new --bin chapter2-first-program`` to get a simple binary project.

Then we will add the `image` crate to load an image. We will also add the `imgshow` to display images as well as the `imageproc` crates to modify images. Finally, we will add the ``rand`` crate to generate random pixel coordinates.

So in the `cargo.toml` file we add the following to the `[dependencies]` section:
```toml
[dependencies]
imgshow = { git = "https://github.com/rust-cv/cv.git" }
image = "0.23.7"
imageproc = "0.21.0"
rand = "0.7.3"
```
### Adding an image

To get something to display and to work on, we will add an image to our project. Create a `res` directory in the parent directory of the project. In this repository put a png named `0000000000.png`. You can use the same as the tutorial is using by downloading the same [image](https://raw.githubusercontent.com/rust-cv/cv/c7540dccf45af310c7f7dfa12ac31a2b04b26224/akaze/res/0000000000.png).

### Adding the use statements

As you can expect, we to bring into views the crates we have declared as dependencies. So we start our file with a vew use statements:
```rs
use image::{DynamicImage, Rgba, GenericImageView};
use imageproc::drawing;
use rand::Rng;
```

### Loading the image

Let's start by loading the image: 
```rs
let src_image = 
    image::open("../res/0000000000.png").expect("failed to open image file");
```

Nothing fancy here. we just call `image::open` with a relative path to get an image. As we don't want to handle errors, we use the `expect` function to panic on failure while displaying a nice error message.

### Drawing random points on our image

To draw random points on our image, we start by initializing our random generator:

```rs
let mut rng = rand::thread_rng();
```

Then, we create a canvas with is initialized with our existing image:
```rs
let mut canvas = drawing::Blend(src_image.to_rgba());
```
And here we can draw 50 random points :
```rs
for _ in 0..50 {
    let x : i32 = rng.gen_range(0, src_image.width() - 1) as i32;
    let y : i32 = rng.gen_range(0, src_image.height() - 1) as i32;
    drawing::draw_cross_mut(&mut canvas, Rgba([0, 255, 255, 128]), x as i32, y as i32);
}
```
For each iteration of the loop, we generate a random value for x and y which represent our point location. Then we paint a new point in our canvas.

### Displaying the image

At this point we just need to convert back our canvas to an image and the we can display it:

```rs
let out_img = DynamicImage::ImageRgba8(canvas.0);
imgshow::imgshow(&out_img);
```

### Result

If the compile you should obtain something similar as the image below:

![Random points](https://rust-cv.github.io/res/tutorial-images/random-points.png)

### Code

You can also find the whole project [here](https://github.com/rust-cv/cv/tree/master/tutorial-code/chapter2-first-program). The dependencies of the code are sightly different in the repository, as it utilizes the dependencies within the mono-repo to ensure compatibility with master.
