use crate::{edit_box_guards::PathGuard, TopLevelMessage};
use kas::class::HasString;
use kas::widget::{Label, StrLabel};
use kas::{
    event::Response, prelude::*, widget::Column, widget::EditBox, widget::Filler, widget::Frame,
    widget::ScrollRegion, widget::TextButton,
};

#[derive(Clone, Debug, VoidMsg)]
pub enum FilesPanelMsg {
    AddFile,
    Run,
}

#[derive(Clone, Debug, VoidMsg)]
pub enum FileEntryMsg {
    DeleteEntry(usize),
}

#[layout(row)]
#[handler(msg=FileEntryMsg)]
#[derive(Clone, Debug, Widget)]
struct ListEntry {
    #[widget_core]
    core: CoreData,
    #[layout_data]
    layout_data: <Self as kas::LayoutData>::Data,

    #[widget]
    path_label: StrLabel,

    #[widget]
    path_edit: EditBox<PathGuard>,
    path: String,

    #[widget]
    delete_button: TextButton<FileEntryMsg>,

    #[widget]
    margin_label: StrLabel,

    index: usize,
}

impl ListEntry {
    fn new(index: usize) -> ListEntry {
        let path = format!("image-{}.png", index);

        let entry = ListEntry {
            core: CoreData::default(),
            layout_data: <Self as LayoutData>::Data::default(),
            path: path.clone(),
            path_label: StrLabel::new("Path: ").with_reserve("xxxxxx"),
            path_edit: EditBox::new(path).with_guard(PathGuard {}),
            delete_button: TextButton::new("X", FileEntryMsg::DeleteEntry(index)),
            margin_label: StrLabel::new(" "),
            index,
        };

        entry
    }
}

#[layout(grid)]
#[derive(Debug, Widget)]
#[handler(msg=TopLevelMessage)]
pub struct FilesPanel {
    #[widget_core]
    core: CoreData,
    #[layout_data]
    layout_data: <Self as LayoutData>::Data,

    #[widget(row = 0, col = 1, cspan = 1, rspan = 1, valign = top, halign = centre, handler = buttons_handler)]
    add_button: TextButton<FilesPanelMsg>,

    #[widget(row = 0, col = 0, cspan = 1, rspan = 1, valign = top, halign = centre, handler = buttons_handler)]
    run_button: TextButton<FilesPanelMsg>,

    #[widget(row = 1, col = 0, cspan = 2, rspan = 1, valign = top, halign = centre)]
    frame: Frame<StrLabel>,

    #[widget(row = 2, col = 0, cspan = 2, rspan = 2, valign = stretch, halign = stretch,  handler = list_handler)]
    scroll_list: ScrollRegion<Column<ListEntry>>,
    child_contents: Vec<ListEntry>,
    list_last_index: usize,

    #[widget(row = 3, col = 0, cspan = 2, rspan = 1, valign = stretch)]
    filler: Filler,
}

impl FilesPanel {
    pub fn new() -> FilesPanel {
        let files_entries: Vec<ListEntry> = vec![];

        let widget = FilesPanel {
            core: CoreData::default(),
            layout_data: <Self as LayoutData>::Data::default(),
            frame: Frame::new(StrLabel::new("Input Images")),
            add_button: TextButton::new("Add Image", FilesPanelMsg::AddFile),
            run_button: TextButton::new("Run", FilesPanelMsg::Run),
            scroll_list: ScrollRegion::new(Column::new(files_entries.clone())).with_auto_bars(true),
            filler: Filler::new(),
            child_contents: files_entries,
            list_last_index: 0usize,
        };
        widget
    }

    fn buttons_handler(
        &mut self,
        manager: &mut Manager,
        msg: FilesPanelMsg,
    ) -> Response<TopLevelMessage> {
        let response = match msg {
            FilesPanelMsg::AddFile => {
                self.list_last_index += 1;
                let new_child = ListEntry::new(self.list_last_index);
                let action: kas::TkAction = self.scroll_list.inner_mut().push(new_child.clone());
                self.child_contents.push(new_child);
                *manager += action;
                Response::None
            }
            FilesPanelMsg::Run => Response::Msg(TopLevelMessage::Run),
        };
        response
    }

    fn list_handler(
        &mut self,
        manager: &mut Manager,
        msg: FileEntryMsg,
    ) -> Response<TopLevelMessage> {
        match msg {
            FileEntryMsg::DeleteEntry(removal_index) => {
                let mut widget_index_to_remove: isize = -1;
                let mut vec_index_to_remove: isize = -1;

                for (widget_index, widget) in self.scroll_list.inner().iter().enumerate() {
                    if widget.index == removal_index {
                        widget_index_to_remove = widget_index as isize;
                    }
                }

                for (widget_vec_index, widget) in self.child_contents.iter().enumerate() {
                    if widget.index == removal_index {
                        vec_index_to_remove = widget_vec_index as isize;
                    }
                }

                let (_widget, action) = self
                    .scroll_list
                    .inner_mut()
                    .remove(widget_index_to_remove as usize);

                self.child_contents.remove(vec_index_to_remove as usize);

                *manager += action;
            }
        }
        Response::None
    }

    fn get_scroll_list_length(&self) -> usize {
        let column: &Column<ListEntry> = self.scroll_list.inner();
        column.len()
    }

    fn get_scroll_list_index(&self, index: usize) -> &ListEntry {
        let column: &Column<ListEntry> = self.scroll_list.inner();

        let child: &(dyn kas::WidgetConfig + 'static) = column.get(index).unwrap();

        let child_down_casted: Option<&ListEntry> = child.as_any().downcast_ref::<ListEntry>();

        child_down_casted.unwrap()
    }

    pub fn get_paths(&self) -> Vec<String> {
        let nb_scroll_list_items = self.get_scroll_list_length();

        let mut paths = vec![];

        for index in 0..nb_scroll_list_items {
            let list_entry = self.get_scroll_list_index(index);
            let path = list_entry.path_edit.get_str();
            paths.push(path.into());
        }

        paths
    }
}
