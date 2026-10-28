//! Rendered frames -> a dataset pool, in this repo's format. No Bevy here, so any renderer
//! can use it (and it can move into a CortenForge crate later).
//!
//! A pool is `images/`, `labels/` (YOLO: `class cx cy w h`, normalised; an empty file means
//! "nothing here", a verified negative) and `metadata.csv` with the columns of
//! `aerial_animal_rescue.metadata.ImageRecord`. `split_dataset.py` reads it like any pool;
//! sim images are `synthetic=True`, so they only ever go to train.

use std::fmt::Write as _;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

/// Must match `aerial_animal_rescue.metadata.COLUMNS`, in order.
pub const COLUMNS: [&str; 13] = [
    "image_id",
    "file_path",
    "source",
    "source_license",
    "modality",
    "synthetic",
    "altitude_m",
    "weather",
    "scene_type",
    "sequence_id",
    "capture_date",
    "split",
    "privacy_reviewed",
];

/// A box in pixels (x right, y down), with our class id (`configs/classes.yaml`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PixelBox {
    pub class_id: u32,
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
}

impl PixelBox {
    /// Bounding box of projected points, clipped to the image. `None` if nothing of it is
    /// inside or it is under `min_side` pixels wide or high after clipping.
    pub fn from_points(
        class_id: u32,
        points: impl IntoIterator<Item = (f32, f32)>,
        width: u32,
        height: u32,
        min_side: f32,
    ) -> Option<Self> {
        let (mut x1, mut y1, mut x2, mut y2) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
        for (x, y) in points {
            x1 = x1.min(x);
            y1 = y1.min(y);
            x2 = x2.max(x);
            y2 = y2.max(y);
        }
        let b = Self {
            class_id,
            x1: x1.max(0.0),
            y1: y1.max(0.0),
            x2: x2.min(width as f32),
            y2: y2.min(height as f32),
        };
        (b.x2 - b.x1 >= min_side && b.y2 - b.y1 >= min_side).then_some(b)
    }

    /// One YOLO label line: `class cx cy w h`, normalised to the image size.
    pub fn yolo_line(&self, width: u32, height: u32) -> String {
        let (w, h) = (width as f32, height as f32);
        format!(
            "{} {:.6} {:.6} {:.6} {:.6}",
            self.class_id,
            (self.x1 + self.x2) / 2.0 / w,
            (self.y1 + self.y2) / 2.0 / h,
            (self.x2 - self.x1) / w,
            (self.y2 - self.y1) / h,
        )
    }
}

/// Tight boxes from an object-id mask (row-major, `0` = background): one per id with at
/// least `min_pixels` visible pixels, as (id, x1, y1, x2, y2) with exclusive x2/y2.
/// Ids are the renderer's object ids; the caller maps them to classes.
pub fn boxes_from_id_mask(
    mask: &[u32],
    width: usize,
    min_pixels: usize,
) -> Vec<(u32, [usize; 4], usize)> {
    use std::collections::BTreeMap;
    let mut acc: BTreeMap<u32, ([usize; 4], usize)> = BTreeMap::new();
    for (k, &id) in mask.iter().enumerate() {
        if id == 0 {
            continue;
        }
        let (x, y) = (k % width, k / width);
        let e = acc.entry(id).or_insert(([x, y, x + 1, y + 1], 0));
        e.0 = [
            e.0[0].min(x),
            e.0[1].min(y),
            e.0[2].max(x + 1),
            e.0[3].max(y + 1),
        ];
        e.1 += 1;
    }
    acc.into_iter()
        .filter(|(_, (_, n))| *n >= min_pixels)
        .map(|(id, (b, n))| (id, b, n))
        .collect()
}

/// Object-id mask from an id render (RGB bytes, row-major): object `k` is drawn with red
/// `k * step` and everything else with red 0, so `step` leaves room for small colour
/// errors in the render. Returns ids, `0` for background.
pub fn id_mask_from_rgb(rgb: &[u8], step: u8) -> Vec<u32> {
    rgb.chunks_exact(3)
        .map(|px| ((px[0] as f32) / step as f32).round() as u32)
        .collect()
}

/// Metadata for one image; the rest of the row is filled by `PoolWriter`.
#[derive(Clone, Debug, Default)]
pub struct Record {
    /// Licenses of everything in the image (our work plus any assets), e.g. "CC-BY-4.0".
    pub source_license: String,
    /// "thermal" or "rgb".
    pub modality: String,
    pub altitude_m: Option<f64>,
    /// clear | rain | fog | unknown
    pub weather: String,
    /// flood | debris_flow | pasture | urban | other
    pub scene_type: String,
    /// Scenes that share a world (terrain seed) share a sequence, so the split keeps them
    /// together.
    pub sequence_id: String,
}

pub struct PoolWriter {
    root: PathBuf,
    source: String,
    rows: Vec<String>,
}

impl PoolWriter {
    /// Start a pool at `root` (creates `images/` and `labels/`). Refuses a folder that
    /// already holds a `metadata.csv`, so a run never mixes with an older one.
    pub fn create(root: impl AsRef<Path>, source: &str) -> io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        if root.join("metadata.csv").exists() {
            return Err(io::Error::new(
                io::ErrorKind::AlreadyExists,
                format!(
                    "{} already holds a pool; pick another --out",
                    root.display()
                ),
            ));
        }
        fs::create_dir_all(root.join("images"))?;
        fs::create_dir_all(root.join("labels"))?;
        Ok(Self {
            root,
            source: source.to_string(),
            rows: Vec::new(),
        })
    }

    pub fn image_path(&self, image_id: &str, ext: &str) -> PathBuf {
        self.root.join("images").join(format!("{image_id}.{ext}"))
    }

    /// Record an image already written to `image_path(image_id, ext)`, with its boxes.
    pub fn add(
        &mut self,
        image_id: &str,
        ext: &str,
        width: u32,
        height: u32,
        boxes: &[PixelBox],
        rec: &Record,
    ) -> io::Result<()> {
        for field in [image_id, &rec.source_license, &rec.sequence_id] {
            if field.contains([',', '"', '\n']) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("bad CSV field {field:?}"),
                ));
            }
        }
        let mut label = String::new();
        for b in boxes {
            writeln!(label, "{}", b.yolo_line(width, height)).unwrap();
        }
        fs::write(
            self.root.join("labels").join(format!("{image_id}.txt")),
            label,
        )?;
        let altitude = rec
            .altitude_m
            .map(|a| format!("{a:.1}"))
            .unwrap_or_default();
        self.rows.push(format!(
            "{image_id},images/{image_id}.{ext},{},{},{},True,{altitude},{},{},{},,,False",
            self.source,
            rec.source_license,
            rec.modality,
            rec.weather,
            rec.scene_type,
            rec.sequence_id,
        ));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.rows.len()
    }

    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Write `metadata.csv`; the pool is complete after this.
    pub fn finish(self) -> io::Result<PathBuf> {
        let mut csv = COLUMNS.join(",");
        csv.push('\n');
        for r in &self.rows {
            csv.push_str(r);
            csv.push('\n');
        }
        let path = self.root.join("metadata.csv");
        fs::write(&path, csv)?;
        Ok(path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn box_from_points_clips_and_drops_tiny() {
        let b = PixelBox::from_points(0, [(-5.0, 10.0), (20.0, 30.0)], 100, 100, 1.0).unwrap();
        assert_eq!((b.x1, b.y1, b.x2, b.y2), (0.0, 10.0, 20.0, 30.0));
        assert!(PixelBox::from_points(0, [(150.0, 10.0), (160.0, 20.0)], 100, 100, 1.0).is_none());
        assert!(PixelBox::from_points(0, [(10.0, 10.0), (10.5, 20.0)], 100, 100, 1.0).is_none());
    }

    #[test]
    fn yolo_line_is_normalised_centre_and_size() {
        let b = PixelBox {
            class_id: 4,
            x1: 10.0,
            y1: 20.0,
            x2: 30.0,
            y2: 60.0,
        };
        assert_eq!(
            b.yolo_line(100, 200),
            "4 0.200000 0.200000 0.200000 0.200000"
        );
    }

    #[test]
    fn id_mask_gives_tight_boxes_and_drops_small_objects() {
        #[rustfmt::skip]
        let mask = [
            0, 1, 1, 0,
            0, 1, 0, 2,
            0, 0, 0, 0,
        ];
        let boxes = boxes_from_id_mask(&mask, 4, 2);
        assert_eq!(boxes, vec![(1, [1, 0, 3, 2], 3)]); // id 2 has 1 pixel: dropped
    }

    #[test]
    fn id_mask_decodes_red_steps_and_ignores_background() {
        let rgb = [0, 200, 0, 10, 0, 0, 31, 0, 0, 4, 90, 0];
        assert_eq!(id_mask_from_rgb(&rgb, 10), vec![0, 1, 3, 0]);
    }

    #[test]
    fn pool_writer_writes_labels_and_metadata() {
        let dir = std::env::temp_dir().join(format!("storm_sim_pool_{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let mut w = PoolWriter::create(&dir, "sim").unwrap();
        fs::write(w.image_path("sim_a", "png"), b"x").unwrap();
        let rec = Record {
            source_license: "CC-BY-4.0".into(),
            modality: "thermal".into(),
            altitude_m: Some(80.0),
            weather: "unknown".into(),
            scene_type: "other".into(),
            sequence_id: "world-1".into(),
        };
        let b = PixelBox {
            class_id: 0,
            x1: 0.0,
            y1: 0.0,
            x2: 10.0,
            y2: 10.0,
        };
        w.add("sim_a", "png", 100, 100, &[b], &rec).unwrap();
        w.add("sim_b", "png", 100, 100, &[], &rec).unwrap();
        let meta = fs::read_to_string(w.finish().unwrap()).unwrap();
        let lines: Vec<_> = meta.lines().collect();
        assert_eq!(lines[0], COLUMNS.join(","));
        assert_eq!(
            lines[1],
            "sim_a,images/sim_a.png,sim,CC-BY-4.0,thermal,True,80.0,unknown,other,world-1,,,False"
        );
        assert_eq!(
            fs::read_to_string(dir.join("labels/sim_b.txt")).unwrap(),
            ""
        );
        assert!(PoolWriter::create(&dir, "sim").is_err()); // never mixes with an older run
        fs::remove_dir_all(&dir).unwrap();
    }
}
