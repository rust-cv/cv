use iced::{Application, Command, Element, Settings, Subscription};
use iced_native::input::{
    keyboard::{Event, KeyCode},
    ButtonState,
};
use image::{DynamicImage, GenericImageView};

/// This function shows an image and will not return until the user presses space or closes the window.
pub fn imgshow(image: &DynamicImage) {
    let mut settings = Settings::with_flags(image.clone());
    settings.window.size = (image.width(), image.height());
    Imgshow::run(settings)
}

struct Imgshow {
    image: iced::image::Handle,
}

#[derive(Debug, Clone, Copy)]
enum Message {
    Close,
    Nothing,
}

impl Application for Imgshow {
    type Executor = iced::executor::Default;
    type Message = Message;
    type Flags = DynamicImage;

    fn new(image: DynamicImage) -> (Self, Command<Message>) {
        let bgra_data = image.to_bgra8().into_raw();
        (
            Self {
                image: iced::image::Handle::from_pixels(image.width(), image.height(), bgra_data),
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("imgshow")
    }

    fn subscription(&self) -> Subscription<Message> {
        iced_native::subscription::events().map(|event| {
            if let iced_native::Event::Keyboard(Event::Input {
                state: ButtonState::Pressed,
                key_code: KeyCode::Space,
                modifiers: _,
            }) = event
            {
                Message::Close
            } else {
                Message::Nothing
            }
        })
    }

    fn update(&mut self, message: Message) -> Command<Message> {
        match message {
            Message::Close => std::process::exit(0),
            Message::Nothing => Command::none(),
        }
    }

    fn view(&mut self) -> Element<Message> {
        iced::Image::new(self.image.clone()).into()
    }
}
