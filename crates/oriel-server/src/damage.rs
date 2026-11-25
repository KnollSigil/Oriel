//! Frame damage: which pixels of a frame differ from the frame before it.
//!
//! Clients report damage on `wl_surface.damage` / `wl_surface.damage_buffer`. It is converted to
//! pixels of the frame stored for the surface, mapped into the window composite, carried with the
//! frame to its encoder, and unioned across frames the encoder skips.

use smithay::utils::{Buffer as BufferCoord, Physical, Rectangle, Transform};
use smithay::wayland::compositor::Damage;

/// A damaged rectangle in frame pixels.
pub type DamageRect = Rectangle<i32, Physical>;

/// Past this many rectangles, damage collapses to their bounding box so it stays small however
/// many commits it accumulates.
const MAX_RECTS: usize = 32;

/// The pixels of a frame that may differ from the previous frame.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub enum FrameDamage {
    /// Anything may have changed.
    #[default]
    Full,
    /// Only these rectangles changed; none means the frame is identical.
    Rects(Vec<DamageRect>),
}

impl FrameDamage {
    /// No damage: the frame is identical to the previous one.
    pub fn empty() -> Self {
        Self::Rects(Vec::new())
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, Self::Rects(rects) if rects.is_empty())
    }

    pub fn add(&mut self, rect: DamageRect) {
        let Self::Rects(rects) = self else {
            return;
        };
        if rect.size.w <= 0 || rect.size.h <= 0 {
            return;
        }
        if rects.iter().any(|r| r.contains_rect(rect)) {
            return;
        }
        rects.retain(|r| !rect.contains_rect(*r));
        rects.push(rect);
        if rects.len() > MAX_RECTS {
            let bbox = rects.iter().copied().reduce(|a, b| a.merge(b));
            rects.clear();
            rects.extend(bbox);
        }
    }

    /// Union `other` into this damage.
    pub fn merge(&mut self, other: FrameDamage) {
        match other {
            Self::Full => *self = Self::Full,
            Self::Rects(rects) => {
                for rect in rects {
                    self.add(rect);
                }
            }
        }
    }

    /// The damaged rectangles clipped to a `width`x`height` frame as `(x, y, w, h)`, or `None`
    /// when the whole frame is damaged.
    pub fn clipped_rects(&self, width: u32, height: u32) -> Option<Vec<(u32, u32, u32, u32)>> {
        let Self::Rects(rects) = self else {
            return None;
        };
        let frame = DamageRect::from_size((width as i32, height as i32).into());
        Some(
            rects
                .iter()
                .filter_map(|r| r.intersection(frame))
                .map(|r| {
                    (
                        r.loc.x as u32,
                        r.loc.y as u32,
                        r.size.w as u32,
                        r.size.h as u32,
                    )
                })
                .collect(),
        )
    }

    /// This damage, of a `src_w`x`src_h` frame, where that frame is drawn at `dst` (scaled to
    /// its size) in another frame.
    pub fn placed(&self, src_w: u32, src_h: u32, dst: DamageRect) -> FrameDamage {
        let Self::Rects(rects) = self else {
            let mut out = Self::empty();
            out.add(dst);
            return out;
        };
        let mut out = Self::empty();
        if src_w == 0 || src_h == 0 {
            return out;
        }
        let unscaled = dst.size.w as u32 == src_w && dst.size.h as u32 == src_h;
        let fx = f64::from(dst.size.w) / f64::from(src_w);
        let fy = f64::from(dst.size.h) / f64::from(src_h);
        for r in rects {
            let Some(r) =
                r.intersection(DamageRect::from_size((src_w as i32, src_h as i32).into()))
            else {
                continue;
            };
            let placed = if unscaled {
                DamageRect::new(dst.loc + r.loc, r.size)
            } else {
                // Filtered resampling reads a pixel beyond each edge.
                let x0 = (f64::from(r.loc.x) * fx).floor() as i32 - 1;
                let y0 = (f64::from(r.loc.y) * fy).floor() as i32 - 1;
                let x1 = (f64::from(r.loc.x + r.size.w) * fx).ceil() as i32 + 1;
                let y1 = (f64::from(r.loc.y + r.size.h) * fy).ceil() as i32 + 1;
                DamageRect::new(
                    (dst.loc.x + x0, dst.loc.y + y0).into(),
                    (x1 - x0, y1 - y0).into(),
                )
            };
            if let Some(clipped) = placed.intersection(dst) {
                out.add(clipped);
            }
        }
        out
    }
}

/// How a surface's buffer becomes the frame stored for it. Damage only carries over between two
/// commits with the same view; otherwise the whole surface counts as damaged.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceView {
    /// Stored frame size in pixels.
    pub frame_size: (u32, u32),
    /// Surface size in logical pixels.
    pub logical_size: (u32, u32),
    pub buffer_scale: i32,
    pub transform: Transform,
    /// The `wp_viewport` source crop applied to the buffer, if any.
    pub crop: Option<Rectangle<f64, BufferCoord>>,
}

/// Convert one commit's `wl_surface` damage to pixels of the frame stored for the surface.
pub fn surface_damage_to_frame(damage: &[Damage], view: &SurfaceView) -> FrameDamage {
    let (frame_w, frame_h) = view.frame_size;
    let (log_w, log_h) = view.logical_size;
    if view.transform != Transform::Normal || frame_w == 0 || frame_h == 0 {
        return FrameDamage::Full;
    }
    let frame = DamageRect::from_size((frame_w as i32, frame_h as i32).into());
    let (crop_x, crop_y) = view
        .crop
        .map_or((0.0, 0.0), |c| (c.loc.x.round(), c.loc.y.round()));
    // A crop off the pixel grid or scaled by the readback blends neighbouring pixels.
    let crop_resamples = view.crop.is_some_and(|c| {
        c.loc.x.fract() != 0.0
            || c.loc.y.fract() != 0.0
            || c.size.w.round() as u32 != frame_w
            || c.size.h.round() as u32 != frame_h
    });

    let mut out = FrameDamage::empty();
    for d in damage {
        let (x0, y0, x1, y1) = match d {
            Damage::Buffer(r) => (
                f64::from(r.loc.x) - crop_x,
                f64::from(r.loc.y) - crop_y,
                f64::from(r.loc.x) + f64::from(r.size.w) - crop_x,
                f64::from(r.loc.y) + f64::from(r.size.h) - crop_y,
            ),
            Damage::Surface(r) => {
                if log_w == 0 || log_h == 0 {
                    return FrameDamage::Full;
                }
                let fx = f64::from(frame_w) / f64::from(log_w);
                let fy = f64::from(frame_h) / f64::from(log_h);
                (
                    f64::from(r.loc.x) * fx,
                    f64::from(r.loc.y) * fy,
                    (f64::from(r.loc.x) + f64::from(r.size.w)) * fx,
                    (f64::from(r.loc.y) + f64::from(r.size.h)) * fy,
                )
            }
        };
        let pad = if crop_resamples { 1.0 } else { 0.0 };
        let clamp = |v: f64| v.clamp(-1.0, f64::from(i32::MAX / 2)) as i32;
        let (x0, y0) = (clamp((x0 - pad).floor()), clamp((y0 - pad).floor()));
        let (x1, y1) = (clamp((x1 + pad).ceil()), clamp((y1 + pad).ceil()));
        if x1 <= x0 || y1 <= y0 {
            continue;
        }
        let rect = DamageRect::new((x0, y0).into(), (x1 - x0, y1 - y0).into());
        if let Some(clipped) = rect.intersection(frame) {
            out.add(clipped);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use smithay::utils::Logical;

    fn rect(x: i32, y: i32, w: i32, h: i32) -> DamageRect {
        DamageRect::new((x, y).into(), (w, h).into())
    }

    fn view(frame: (u32, u32), logical: (u32, u32)) -> SurfaceView {
        SurfaceView {
            frame_size: frame,
            logical_size: logical,
            buffer_scale: 1,
            transform: Transform::Normal,
            crop: None,
        }
    }

    #[test]
    fn merge_unions_and_full_absorbs() {
        let mut d = FrameDamage::empty();
        assert!(d.is_empty());
        d.merge(FrameDamage::Rects(vec![rect(0, 0, 4, 4)]));
        d.merge(FrameDamage::Rects(vec![
            rect(1, 1, 2, 2),
            rect(10, 10, 2, 2),
        ]));
        assert_eq!(
            d,
            FrameDamage::Rects(vec![rect(0, 0, 4, 4), rect(10, 10, 2, 2)])
        );
        d.merge(FrameDamage::Full);
        assert_eq!(d, FrameDamage::Full);
        d.merge(FrameDamage::Rects(vec![rect(0, 0, 1, 1)]));
        assert_eq!(d, FrameDamage::Full);
    }

    #[test]
    fn many_rects_collapse_to_bounding_box() {
        let mut d = FrameDamage::empty();
        for i in 0..=MAX_RECTS as i32 {
            d.add(rect(i * 10, i * 5, 2, 2));
        }
        let n = MAX_RECTS as i32;
        assert_eq!(
            d,
            FrameDamage::Rects(vec![rect(0, 0, n * 10 + 2, n * 5 + 2)])
        );
    }

    #[test]
    fn clipped_rects_drop_offscreen_damage() {
        let d = FrameDamage::Rects(vec![rect(-5, -5, 10, 10), rect(100, 100, 5, 5)]);
        assert_eq!(d.clipped_rects(50, 50), Some(vec![(0, 0, 5, 5)]));
        assert_eq!(FrameDamage::Full.clipped_rects(50, 50), None);
    }

    #[test]
    fn placed_offsets_and_scales() {
        let d = FrameDamage::Rects(vec![rect(2, 3, 4, 5)]);
        assert_eq!(
            d.placed(10, 10, rect(100, 200, 10, 10)),
            FrameDamage::Rects(vec![rect(102, 203, 4, 5)])
        );
        // Doubled, padded by a pixel and clipped to the destination.
        assert_eq!(
            d.placed(10, 10, rect(0, 0, 20, 20)),
            FrameDamage::Rects(vec![rect(3, 5, 10, 12)])
        );
        assert_eq!(
            FrameDamage::Full.placed(10, 10, rect(5, 5, 20, 20)),
            FrameDamage::Rects(vec![rect(5, 5, 20, 20)])
        );
    }

    #[test]
    fn surface_damage_scales_to_buffer_pixels() {
        let damage = [Damage::Surface(Rectangle::<i32, Logical>::new(
            (10, 20).into(),
            (30, 40).into(),
        ))];
        let d = surface_damage_to_frame(&damage, &view((200, 200), (100, 100)));
        assert_eq!(d, FrameDamage::Rects(vec![rect(20, 40, 60, 80)]));
    }

    #[test]
    fn whole_surface_damage_clips_to_frame() {
        // Toolkits damage (0, 0, INT32_MAX, INT32_MAX) to repaint everything.
        let damage = [Damage::Surface(Rectangle::<i32, Logical>::new(
            (0, 0).into(),
            (i32::MAX, i32::MAX).into(),
        ))];
        let d = surface_damage_to_frame(&damage, &view((300, 200), (150, 100)));
        assert_eq!(d, FrameDamage::Rects(vec![rect(0, 0, 300, 200)]));
    }

    #[test]
    fn buffer_damage_shifts_by_crop() {
        let damage = [Damage::Buffer(Rectangle::<i32, BufferCoord>::new(
            (110, 50).into(),
            (20, 10).into(),
        ))];
        let mut v = view((80, 60), (80, 60));
        v.crop = Some(Rectangle::new((100.0, 40.0).into(), (80.0, 60.0).into()));
        let d = surface_damage_to_frame(&damage, &v);
        assert_eq!(d, FrameDamage::Rects(vec![rect(10, 10, 20, 10)]));
        // Damage outside the crop is dropped.
        let outside = [Damage::Buffer(Rectangle::<i32, BufferCoord>::new(
            (0, 0).into(),
            (50, 30).into(),
        ))];
        assert!(surface_damage_to_frame(&outside, &v).is_empty());
    }

    #[test]
    fn no_damage_is_empty_and_transforms_are_full() {
        assert!(surface_damage_to_frame(&[], &view((10, 10), (10, 10))).is_empty());
        let mut v = view((10, 10), (10, 10));
        v.transform = Transform::_90;
        assert_eq!(surface_damage_to_frame(&[], &v), FrameDamage::Full);
    }
}
