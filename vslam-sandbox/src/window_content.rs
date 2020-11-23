use std::str::FromStr;

use kas::prelude::*;
use kas::widget::{RowSplitter};

use crate::{TopLevelMessage, files_panel::FilesPanel, options_panel::OptionsPanel, run_parameters::RunParameters, view3d_panel::View3DPanel};

type WidgetType = Box<dyn Widget<Msg=TopLevelMessage>>;

#[layout(row)]
#[handler(msg=TopLevelMessage)]
#[derive(Debug, Widget)]
pub struct WindowContent {
    #[widget_core] 
    core: CoreData,
    #[layout_data] 
    layout_data: <Self as LayoutData>::Data,
    #[widget(handler = handler)]
    panes : RowSplitter<WidgetType>,
}

impl WindowContent {
    pub fn new() -> WindowContent {
        let options_panel = Box::new(OptionsPanel::new());
        let view_3d_panel = Box::new(View3DPanel::new());
        let files_panel = Box::new(FilesPanel::new());

        let child_panels: Vec<WidgetType> = 
            vec!(options_panel, view_3d_panel, files_panel);

        let widget = WindowContent {
            core: CoreData::default(),
            layout_data: <Self as LayoutData>::Data::default(),
            panes: RowSplitter::<WidgetType>::new(child_panels)
        };

        widget
    }

    fn handler(&mut self, _mgr: &mut Manager, msg: TopLevelMessage) -> Response<TopLevelMessage> {
        Response::Msg(msg)
    }

    fn get_option_panel(&self) -> &OptionsPanel {
        let child: &(dyn kas::WidgetConfig + 'static) = 
            self.panes.get(0usize).unwrap();

        let child_down_casted : Option<&OptionsPanel> = 
            child.as_any().downcast_ref::<OptionsPanel>();

        child_down_casted.unwrap()
    }

    fn get_files_panel(&self) -> &FilesPanel {
        let child: &(dyn kas::WidgetConfig + 'static) = 
            self.panes.get(4usize).unwrap();

        let child_down_casted : Option<&FilesPanel> = 
            child.as_any().downcast_ref::<FilesPanel>();

        child_down_casted.unwrap()
    }

    pub fn get_run_parameters(&self) -> Result<RunParameters, <f64 as FromStr>::Err> {

        let option_panel = self.get_option_panel();
        let files_panel = self.get_files_panel();

        let option_parameters = 
            option_panel.get_option_parameters().try_parse()?;

        let files_paths = files_panel.get_paths();

        let run_parameters = RunParameters {
            option_parameters: option_parameters,
            files_path: files_paths,
        };

        Ok(run_parameters)
    }
}