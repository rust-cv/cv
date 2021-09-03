use eframe::{
    egui::{self, CtxRef, TextureId, Vec2},
    epi,
    epi::{Frame, Storage},
};
use image::{DynamicImage, GenericImageView};

pub struct DisplayImage {
    texture_id: TextureId,
    size: Vec2,
}
impl DisplayImage {
    pub fn new(texture_id: TextureId, size: Vec2) -> DisplayImage {
        DisplayImage { texture_id, size }
    }
}
pub struct App {
    name: String,
    image: DynamicImage,
    display_image: DisplayImage,
}
impl App {
    pub fn new(image: DynamicImage) -> Self {
        App {
            name: String::from("Rust CV"),
            image,
            display_image: DisplayImage::new(egui::TextureId::Egui, Vec2::default()),
        }
    }
}
impl epi::App for App {
    fn name(&self) -> &str {
        &self.name
    }
    fn setup(&mut self, _ctx: &CtxRef, frame: &mut Frame<'_>, _storage: Option<&dyn Storage>) {
        let Self { image, .. } = self;
        let size = Vec2::from([image.width() as f32, image.height() as f32]);
        let pixels = {
            let image_bytes = image.as_bytes();
            let mut pixels = Vec::with_capacity(image_bytes.len() / 4);
            for rgba in image_bytes.chunks_exact(4) {
                if let [r, g, b, a] = *rgba {
                    pixels.push(egui::Color32::from_rgba_unmultiplied(r, g, b, a));
                }
            }
            pixels
        };
        let image_texture_id = frame
            .tex_allocator()
            .alloc_srgba_premultiplied((image.width() as usize, image.height() as usize), &*pixels);
        self.display_image = DisplayImage::new(image_texture_id, size);
    }
    fn update(&mut self, ctx: &CtxRef, _frame: &mut Frame<'_>) {
        let Self { display_image, .. } = self;
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.image(display_image.texture_id, display_image.size);
        });
    }
}
pub fn imgshow(image: &DynamicImage) -> Result<(), std::io::Error> {
    let app = App::new(image.clone());
    let native_options = eframe::NativeOptions {
        initial_window_size: Option::from(Vec2::from(&[
            (image.width() + 15) as f32,
            (image.height() + 15) as f32,
        ])),
        ..Default::default()
    };
    eframe::run_native(Box::new(app), native_options);
    Ok(())
}
