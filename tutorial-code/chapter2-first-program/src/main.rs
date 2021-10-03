use cv::{
    image::{
        image::{self, DynamicImage, GenericImageView, Rgba},
        imageproc::drawing,
    },
    vis::show_image::{self, event},
};
use rand::Rng;

fn main() {
    // This must happen at the beginning of main to ensure that the image is shown in the main thread.
    // This must be done to support MacOS, which only allows displaying images on the main thread.
    show_image::run_context(|| -> Result<(), Box<dyn std::error::Error>> {
        // Load the image.
        let src_image = image::open("res/0000000000.png").expect("failed to open image file");

        // Create an RNG so we can draw random points on the image.
        let mut rng = rand::thread_rng();

        // Make a canvas with the `imageproc::drawing` module.
        // We use the blend mode so that we can draw with translucency on the image.
        // We convert the image to rgba8 during this process.
        let mut image_canvas = drawing::Blend(src_image.to_rgba8());

        // Loop 50 times.
        for _ in 0..50 {
            // Generate a random pixel coordinate on the image.
            let x = rng.gen_range(0..src_image.width()) as i32;
            let y = rng.gen_range(0..src_image.height()) as i32;
            // Use the `imageproc::drawing` module to render a cross on the image.
            drawing::draw_cross_mut(&mut image_canvas, Rgba([0, 255, 255, 128]), x, y);
        }

        // Get the resulting image.
        let out_image = DynamicImage::ImageRgba8(image_canvas.0);

        // Make a window using `show_image`.
        let window = show_image::create_window("chapter2", Default::default())?;

        // This sets the window to show the image we just created.
        window.set_image("", out_image)?;

        // This event loop continues until the window is requested to be closed or
        // if the escape key is pressed.
        for event in window.event_channel()? {
            match event {
                event::WindowEvent::CloseRequested(_) => break,
                event::WindowEvent::KeyboardInput(event::WindowKeyboardInputEvent {
                    input:
                        event::KeyboardInput {
                            key_code: Some(event::VirtualKeyCode::Escape),
                            ..
                        },
                    ..
                }) => break,
                _ => {}
            }
        }

        Ok(())
    })
}
