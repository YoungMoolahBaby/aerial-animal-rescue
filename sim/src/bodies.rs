//! Baked body models (`scripts/bake_bodies.py`): CC0 animals posed at moments of their
//! walk, graze and gallop clips, split into the torso, limb and head parts that get their
//! own temperatures. Bevy-free; `scene.rs` turns them into meshes.
//!
//! Layout: `<dir>/<kind>/<name>.body`, kind as in `Kind::dir_name` (cattle, horse, ...).

use std::io;
use std::path::Path;

const MAGIC: &[u8; 4] = b"AARB";
const VERSION: u32 = 1;

/// Triangles of one part, metres, facing +z with the feet at y = 0.
#[derive(Clone, Debug, Default)]
pub struct Part {
    pub positions: Vec<[f32; 3]>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug)]
pub struct Pose {
    pub name: String,
    /// Torso, limbs, head.
    pub parts: [Part; 3],
}

#[derive(Clone, Debug)]
pub struct Model {
    pub name: String,
    pub poses: Vec<Pose>,
}

struct Reader<'a> {
    data: &'a [u8],
    at: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> io::Result<&[u8]> {
        let end = self.at + n;
        let s = self
            .data
            .get(self.at..end)
            .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "body file too short"))?;
        self.at = end;
        Ok(s)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    fn f32(&mut self) -> io::Result<f32> {
        Ok(f32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
}

/// Parse one `.body` file.
pub fn parse(name: &str, data: &[u8]) -> io::Result<Model> {
    let bad = |msg: &str| io::Error::new(io::ErrorKind::InvalidData, format!("{name}: {msg}"));
    let mut r = Reader { data, at: 0 };
    if r.take(4)? != MAGIC {
        return Err(bad("not a body file"));
    }
    if r.u32()? != VERSION {
        return Err(bad("unknown version"));
    }
    let n = r.u32()? as usize;
    let mut poses = Vec::with_capacity(n);
    for _ in 0..n {
        let len = r.take(1)?[0] as usize;
        let pose_name = String::from_utf8(r.take(len)?.to_vec()).map_err(|_| bad("pose name"))?;
        let mut parts: [Part; 3] = Default::default();
        for part in &mut parts {
            let nv = r.u32()? as usize;
            let ni = r.u32()? as usize;
            part.positions = (0..nv)
                .map(|_| Ok([r.f32()?, r.f32()?, r.f32()?]))
                .collect::<io::Result<_>>()?;
            part.indices = (0..ni).map(|_| r.u32()).collect::<io::Result<_>>()?;
            if part.indices.iter().any(|&i| i as usize >= nv) {
                return Err(bad("index out of range"));
            }
        }
        poses.push(Pose {
            name: pose_name,
            parts,
        });
    }
    if poses.is_empty() {
        return Err(bad("no poses"));
    }
    Ok(Model {
        name: name.to_string(),
        poses,
    })
}

/// All models of one kind (`<dir>/<kind>/*.body`), sorted by name; none if the folder
/// doesn't exist.
pub fn load_kind(dir: &Path, kind: &str) -> io::Result<Vec<Model>> {
    let folder = dir.join(kind);
    if !folder.is_dir() {
        return Ok(Vec::new());
    }
    let mut paths: Vec<_> = std::fs::read_dir(&folder)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|e| e == "body"))
        .collect();
    paths.sort();
    paths
        .iter()
        .map(|p| {
            let name = p.file_stem().unwrap().to_string_lossy();
            parse(&name, &std::fs::read(p)?)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    type RawPart = (Vec<[f32; 3]>, Vec<u32>);

    fn file(poses: &[(&str, [RawPart; 3])]) -> Vec<u8> {
        let mut b = MAGIC.to_vec();
        b.extend(VERSION.to_le_bytes());
        b.extend((poses.len() as u32).to_le_bytes());
        for (name, parts) in poses {
            b.push(name.len() as u8);
            b.extend(name.as_bytes());
            for (pos, idx) in parts {
                b.extend((pos.len() as u32).to_le_bytes());
                b.extend((idx.len() as u32).to_le_bytes());
                b.extend(pos.iter().flatten().flat_map(|v| v.to_le_bytes()));
                b.extend(idx.iter().flat_map(|v| v.to_le_bytes()));
            }
        }
        b
    }

    fn tri() -> RawPart {
        (
            vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            vec![0, 1, 2],
        )
    }

    #[test]
    fn reads_what_the_baker_writes() {
        let data = file(&[("Walk@0.25", [tri(), (vec![], vec![]), tri()])]);
        let m = parse("Cow", &data).unwrap();
        assert_eq!(m.poses.len(), 1);
        assert_eq!(m.poses[0].name, "Walk@0.25");
        assert_eq!(m.poses[0].parts[0].indices, vec![0, 1, 2]);
        assert!(m.poses[0].parts[1].positions.is_empty());
        assert_eq!(m.poses[0].parts[2].positions[2], [0.0, 1.0, 0.0]);
    }

    #[test]
    fn rejects_broken_files() {
        let mut data = file(&[("Idle", [tri(), tri(), tri()])]);
        assert!(parse("x", &data[..data.len() - 2]).is_err());
        let n = data.len();
        data[n - 4] = 9; // an index past the vertices
        assert!(parse("x", &data).is_err());
        assert!(parse("x", b"nope").is_err());
    }
}
