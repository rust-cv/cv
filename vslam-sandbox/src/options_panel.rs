use kas::class::HasString;
use kas::widget::{Label, StrLabel};
use kas::{prelude::*, widget::EditBox};

use crate::{
    edit_box_guards::NumericGuard, edit_box_guards::PathGuard, run_parameters::OptionParameters,
    TopLevelMessage,
};

const RESERVE_STR: &str = "xxxxxxxxxxxxxx";

#[layout(grid)]
#[handler(msg=TopLevelMessage)]
#[derive(Debug, Widget)]
pub struct OptionsPanel {
    #[widget_core]
    core: CoreData,
    #[layout_data]
    layout_data: <Self as LayoutData>::Data,

    #[widget(row=0, col=0, handler = handler)]
    input_settings_file_label: StrLabel,
    #[widget(row=0, col=1, handler = handler)]
    input_settings_file_value: EditBox<PathGuard>,

    #[widget(row=1, col=0, handler = handler)]
    input_focal_x_label: StrLabel,
    #[widget(row=1, col=1, handler = handler)]
    input_focal_x_value: EditBox<NumericGuard>,

    #[widget(row=2, col=0, handler = handler)]
    input_focal_y_label: StrLabel,
    #[widget(row=2, col=1, handler = handler)]
    input_focal_y_value: EditBox<NumericGuard>,

    #[widget(row=3, col=0, handler = handler)]
    input_focal_x_center_label: StrLabel,
    #[widget(row=3, col=1, handler = handler)]
    input_focal_x_center_value: EditBox<NumericGuard>,

    #[widget(row=4, col=0, handler = handler)]
    input_focal_y_center_label: StrLabel,
    #[widget(row=4, col=1, handler = handler)]
    input_focal_y_center_value: EditBox<NumericGuard>,

    #[widget(row=5, col=0, handler = handler)]
    input_focal_radial_distortion_label: StrLabel,
    #[widget(row=5, col=1, handler = handler)]
    input_focal_radial_distortion_value: EditBox<NumericGuard>,
}

impl OptionsPanel {
    pub fn new() -> OptionsPanel {
        let widget = OptionsPanel {
            core: CoreData::default(),
            layout_data: <Self as LayoutData>::Data::default(),

            input_settings_file_label: StrLabel::new("Settings File:").with_reserve(RESERVE_STR),
            input_settings_file_value: EditBox::new("vslam-settings.json")
                .with_guard(PathGuard {})
                .multi_line(false),

            input_focal_x_label: StrLabel::new("Focal X:").with_reserve(RESERVE_STR),
            input_focal_x_value: EditBox::new("722.8618")
                .with_guard(NumericGuard {})
                .multi_line(false),

            input_focal_y_label: StrLabel::new("Focal Y:").with_reserve(RESERVE_STR),
            input_focal_y_value: EditBox::new("722.8618")
                .with_guard(NumericGuard {})
                .multi_line(false),

            input_focal_x_center_label: StrLabel::new("Optical Center X:")
                .with_reserve(RESERVE_STR),
            input_focal_x_center_value: EditBox::new("462.67601")
                .with_guard(NumericGuard {})
                .multi_line(false),

            input_focal_y_center_label: StrLabel::new("Optical Center Y:")
                .with_reserve(RESERVE_STR),
            input_focal_y_center_value: EditBox::new("266.67308")
                .with_guard(NumericGuard {})
                .multi_line(false),

            input_focal_radial_distortion_label: StrLabel::new("Radial Distortion:")
                .with_reserve(RESERVE_STR),
            input_focal_radial_distortion_value: EditBox::new("0.053694")
                .with_guard(NumericGuard {})
                .multi_line(false),
        };
        widget
    }

    fn get_input_settings_file_value(&self) -> String {
        self.input_settings_file_value.get_str().into()
    }

    fn get_input_focal_x_value(&self) -> String {
        self.input_focal_x_value.get_str().into()
    }

    fn get_input_focal_y_value(&self) -> String {
        self.input_focal_y_value.get_str().into()
    }

    fn get_input_focal_x_center_value(&self) -> String {
        self.input_focal_x_center_value.get_str().into()
    }

    fn get_input_focal_y_center_value(&self) -> String {
        self.input_focal_y_center_value.get_str().into()
    }

    fn get_input_focal_radial_distortion_value(&self) -> String {
        self.input_focal_radial_distortion_value.get_str().into()
    }

    pub fn get_option_parameters(&self) -> OptionParameters {
        OptionParameters {
            input_settings_file: self.get_input_settings_file_value(),
            input_focal_x: self.get_input_focal_x_value(),
            input_focal_y: self.get_input_focal_y_value(),
            input_focal_x_center: self.get_input_focal_x_center_value(),
            input_focal_y_center: self.get_input_focal_y_center_value(),
            input_focal_radial_distortion: self.get_input_focal_radial_distortion_value(),
        }
    }

    fn handler(&mut self, _mgr: &mut Manager, _msg: VoidMsg) -> Response<TopLevelMessage> {
        Response::None
    }
}
