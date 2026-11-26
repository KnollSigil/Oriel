//! A [`gdk::Paintable`] that draws a symbolic (monochrome) icon in the widget's foreground
//! colour, the way GTK recolours symbolic icons from its own icon theme.
//!
//! The icon's alpha channel is used as a mask over a fill of the colour GTK passes to
//! [`gtk::SymbolicPaintable::snapshot_symbolic`] (the CSS `color` of the `GtkImage`). The
//! original fill colour, whether the stock `#bebebe` grey, breeze's `ColorScheme-Text`
//! style or `currentColor`, is ignored, and the icon follows light/dark switches
//! without being decoded again.

use gtk::prelude::*;
use gtk::subclass::prelude::*;
use gtk::{gdk, glib, graphene, gsk};
use gtk4 as gtk;

mod imp {
    use std::cell::OnceCell;

    use super::*;

    #[derive(Default)]
    pub struct SymbolicIcon {
        pub mask: OnceCell<gdk::Texture>,
    }

    #[glib::object_subclass]
    impl ObjectSubclass for SymbolicIcon {
        const NAME: &'static str = "OrielSymbolicIcon";
        type Type = super::SymbolicIcon;
        type Interfaces = (gdk::Paintable, gtk::SymbolicPaintable);
    }

    impl ObjectImpl for SymbolicIcon {}

    impl PaintableImpl for SymbolicIcon {
        fn intrinsic_width(&self) -> i32 {
            self.mask.get().map_or(0, |t| t.width())
        }

        fn intrinsic_height(&self) -> i32 {
            self.mask.get().map_or(0, |t| t.height())
        }

        fn flags(&self) -> gdk::PaintableFlags {
            gdk::PaintableFlags::STATIC_CONTENTS | gdk::PaintableFlags::STATIC_SIZE
        }

        fn snapshot(&self, snapshot: &gdk::Snapshot, width: f64, height: f64) {
            self.snapshot_symbolic(snapshot, width, height, &[]);
        }
    }

    impl SymbolicPaintableImpl for SymbolicIcon {
        fn snapshot_symbolic(
            &self,
            snapshot: &gdk::Snapshot,
            width: f64,
            height: f64,
            colors: &[gdk::RGBA],
        ) {
            let (Some(mask), Some(snapshot)) =
                (self.mask.get(), snapshot.downcast_ref::<gtk::Snapshot>())
            else {
                return;
            };
            // `colors[0]` is the foreground; outside a styled widget, fall back to mid grey.
            let fg = colors
                .first()
                .copied()
                .unwrap_or_else(|| gdk::RGBA::new(0.5, 0.5, 0.5, 1.0));
            let bounds = graphene::Rect::new(0.0, 0.0, width as f32, height as f32);
            snapshot.push_mask(gsk::MaskMode::Alpha);
            snapshot.append_texture(mask, &bounds);
            snapshot.pop();
            snapshot.append_color(&fg, &bounds);
            snapshot.pop();
        }
    }
}

glib::wrapper! {
    pub struct SymbolicIcon(ObjectSubclass<imp::SymbolicIcon>)
        @implements gdk::Paintable, gtk::SymbolicPaintable;
}

impl SymbolicIcon {
    /// Wrap `mask`, whose alpha channel is the icon shape, to draw in the foreground colour.
    pub fn new(mask: gdk::Texture) -> Self {
        let icon: Self = glib::Object::new();
        let _ = icon.imp().mask.set(mask);
        icon
    }
}
