use kas::{event, event::Response, prelude::*};
use kas_wgpu::draw::DrawCustom;
use kas_wgpu::draw::DrawWindow;

use crate::{wgpu_rendering::PipeWindow, TopLevelMessage};

#[widget(config = noauto)]
#[handler(handle = noauto)]
#[derive(Clone, Debug, Widget)]
pub struct View3DPanel {
    #[widget_core]
    core: CoreData,
}

impl View3DPanel {
    pub fn new() -> View3DPanel {
        let widget = View3DPanel {
            core: CoreData::default(),
        };
        widget
    }
}

impl Layout for View3DPanel {
    fn size_rules(&mut self, size_handle: &mut dyn SizeHandle, a: AxisInfo) -> SizeRules {
        let size = (match a.is_horizontal() {
            true => 30.0,
            false => 20.0,
        } * size_handle.scale_factor())
        .round() as u32;
        SizeRules::new(size, size * 3, (0, 0), StretchPolicy::Maximise)
    }

    #[inline]
    fn set_rect(&mut self, rect: Rect, _align: AlignHints) {
        self.core.rect = rect;
    }

    fn draw(&self, draw_handle: &mut dyn DrawHandle, _: &event::ManagerState, _: bool) {
        let (pass, offset, draw) = draw_handle.draw_device();

        // TODO: our view transform assumes that offset = 0.
        // Here it is but in general we should be able to handle an offset here!
        assert_eq!(offset, Coord::ZERO, "view transform assumption violated");

        let draw: &mut DrawWindow<PipeWindow> = draw
            .as_any_mut()
            .downcast_mut::<DrawWindow<PipeWindow>>()
            .unwrap();

        let param = ();
        draw.custom(pass, self.core.rect + offset, param);
    }
}

impl event::Handler for View3DPanel {
    type Msg = TopLevelMessage;

    fn handle(&mut self, mgr: &mut Manager, event: Event) -> Response<TopLevelMessage> {
        match event {
            Event::Control(_key) => {
                mgr.redraw(self.id());
                Response::None
            }
            Event::Scroll(_delta) => {
                mgr.redraw(self.id());
                Response::None
            }
            Event::Pan { alpha: _, delta: _ } => {
                mgr.redraw(self.id());
                Response::None
            }
            Event::PressStart { source, coord, .. } => {
                mgr.request_grab(
                    self.id(),
                    source,
                    coord,
                    event::GrabMode::PanFull,
                    Some(event::CursorIcon::Grabbing),
                );
                Response::None
            }
            _ => Response::None,
        }
    }
}

impl WidgetConfig for View3DPanel {
    fn configure(&mut self, mgr: &mut Manager) {
        mgr.register_nav_fallback(self.id());
    }
}
