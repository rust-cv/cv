use iced::{
    keyboard::{Event, KeyCode},
    Application, Clipboard, Command, Element, Error, Settings, Subscription,
};
use image::{DynamicImage, GenericImageView};

/// This function shows an image and will not return until the user presses space or closes the window.
pub fn imgshow(image: &DynamicImage) -> Result<(), Error> {
    let mut settings = Settings::with_flags(image.clone());
    settings.window.size = (image.width(), image.height());
    Imgshow::run(settings)
}

struct Imgshow {
    image: iced::image::Handle,
    should_exit: bool,
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
                should_exit: false,
            },
            Command::none(),
        )
    }

    fn title(&self) -> String {
        String::from("imgshow")
    }

    fn subscription(&self) -> Subscription<Message> {
        iced_native::subscription::events().map(|event| {
            if let iced_native::Event::Keyboard(Event::KeyPressed {
                key_code: KeyCode::Space | KeyCode::Escape,
                modifiers: _,
            }) = event
            {
                Message::Close
            } else {
                Message::Nothing
            }
        })
    }

    fn update(&mut self, message: Message, _: &mut Clipboard) -> Command<Message> {
        if let Message::Close = message {
            self.should_exit = true;
        }
        Command::none()
    }

    fn should_exit(&self) -> bool {
        self.should_exit
    }

    fn view(&mut self) -> Element<Message> {
        iced::Image::new(self.image.clone()).into()
    }
}
