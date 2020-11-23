use std::num::ParseFloatError; // path::Path

use kas::class::{HasStr, HasString};
use kas::event::VoidMsg;

#[derive(Clone)]
pub struct NumericGuard {}

impl kas::widget::EditGuard for NumericGuard {
    type Msg = VoidMsg;

    fn activate(_: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        None
    }

    fn focus_lost(_: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        None
    }

    fn edit(edit_box: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        let content = edit_box.get_str();
        let parsed_result: Result<f64, ParseFloatError> = content.parse();
        edit_box.set_error_state(parsed_result.is_err());
        None
    }
}

#[derive(Clone)]
pub struct PathGuard {}

impl kas::widget::EditGuard for PathGuard {
    type Msg = VoidMsg;

    fn activate(_: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        None
    }

    fn focus_lost(_: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        None
    }

    fn edit(_edit_box: &mut kas::widget::EditBox<Self>) -> Option<Self::Msg> {
        // let content =  edit_box.get_str();
        // let path = Path::new(content);
        // let is_error: bool = !path.exists();
        // edit_box.set_error_state(is_error);
        None
    }
}
