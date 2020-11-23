mod window_content;
mod view3d_panel;
mod options_panel;
mod files_panel;
mod wgpu_rendering;
mod edit_box_guards;
mod run_parameters;
mod vslam_algo;

use std::path::PathBuf;

use kas::event::{VoidResponse};
use kas::prelude::*;
use kas::widget::*;
use kas::Right;
use kas_wgpu::Options;
use vslam_algo::{VslamInputParameters, run_vslam_algo};
use wgpu_rendering::PipeBuilder;
use window_content::*;

#[derive(Clone, Debug, VoidMsg)]
pub enum TopLevelMessage {
    Todo,
    VoidMsg,
    Run
}

#[derive(Clone, Debug, VoidMsg)]
enum MenuMsg {
    Quit,
}

#[layout(column)]
#[derive(Debug, Widget)]
#[handler(msg = VoidMsg)]
struct MainWindowWidget
{
    #[widget_core] 
    core: CoreData,
    #[layout_data] 
    layout_data: <Self as LayoutData>::Data,

    #[widget(handler = menu)] 
    menubar: MenuBar<Right, Box<dyn Menu<Msg=MenuMsg>>>,

    #[widget(handler = window_content_handler)]
    window_content: Frame<WindowContent>,
}

impl MainWindowWidget {

    fn new() -> MainWindowWidget {
        let window_content_widget = WindowContent::new();

        let menubar: MenuBar<Right, Box<dyn Menu<Msg=MenuMsg>>> = 
            MenuBar::<Right, _>::new(
                vec![
                    SubMenu::new("&App", 
                        vec![MenuEntry::new("&Quit", MenuMsg::Quit).boxed()]
                    )
                ]
            );

        MainWindowWidget {
            core: CoreData::default(),
            layout_data: <Self as LayoutData>::Data::default(),
            menubar: menubar,
            window_content: Frame::new(window_content_widget),
        }
    }

    fn menu(&mut self, mgr: &mut Manager, msg: MenuMsg) -> VoidResponse {
        match msg {
            MenuMsg::Quit => {
                *mgr += TkAction::CloseAll;
            }
        }
        Response::None
    }

    fn window_content_handler(&mut self, _mgr: &mut Manager, msg: TopLevelMessage)
        -> VoidResponse
    {
        let answer = match msg {
            TopLevelMessage::Run => {
                self.run_vslam();
                Response::None
            }
            _ => {
                todo!("Other top level messages")
            }
        };
        answer
    }

    fn run_vslam(&mut self) {
        let first_child_of_window_frame: &(dyn kas::WidgetConfig + 'static) = 
                    self.window_content.get(0usize).unwrap();

        let child_down_casted : Option<&WindowContent> = 
            first_child_of_window_frame.as_any().downcast_ref::<WindowContent>();

        let window_content = child_down_casted
            .expect("MainWindow frame should downcast to window content");

        let run_parameters = window_content.get_run_parameters();

        match run_parameters {
            Ok(parameters) => {

                let images: Vec<PathBuf> = 
                    parameters.files_path
                    .iter()
                    .map(|a| { 
                        a.into() 
                    })
                    .collect();

                let vslam_parameters: VslamInputParameters =
                    VslamInputParameters::new(
                        parameters.option_parameters.input_focal_x, 
                        parameters.option_parameters.input_focal_y, 
                        parameters.option_parameters.input_focal_x_center, 
                        parameters.option_parameters.input_focal_y_center, 
                        parameters.option_parameters.input_focal_radial_distortion, 
                        images, 
                        parameters.option_parameters.input_settings_file.into(), 
                    );

                let _vslam_result = run_vslam_algo(vslam_parameters);
                println!("done running vslam algo");
            }
            Err(_) => {
                todo!("handle parse error");
            }
        }
    }
}

fn main() -> Result<(), kas_wgpu::Error> {
    pretty_env_logger::init_timed();

    let window = Window::new(
        "VSLAM",
        MainWindowWidget::new(),
    );

    let theme = kas_theme::MultiTheme::builder()
        .add("flat", kas_theme::FlatTheme::new())
        .build();

    let mut toolkit = 
        kas_wgpu::Toolkit::new_custom(PipeBuilder, theme, Options::from_env())?;

    toolkit.add(window)?;
    toolkit.run()
}