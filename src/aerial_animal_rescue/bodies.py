"""Bake animated glTF animals into static body meshes for the storm sim.

Each model is posed at chosen moments of its animation clips (skinning on the CPU), split
into the sim's three temperature parts by each vertex's strongest bone (torso, limbs,
head), turned to face +z with the feet at y = 0, scaled to a real nose-to-rump length,
and written as a small binary file that `sim/src/bodies.rs` reads:

    b"AARB", u32 version, u32 poses, then per pose: u8 name length, name (utf-8), and per
    part (torso, limb, head): u32 vertices, u32 indices, f32 xyz * vertices, u32 indices.

All little-endian. Bones are named like Quaternius' animal rig (Body, Torso*, Neck*,
Head, Ear*, *Leg*, Tail*); see `part_of`.
"""

import json
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

TORSO, LIMB, HEAD = 0, 1, 2
MAGIC = b"AARB"
VERSION = 1

COMPONENT = {
    5120: np.int8,
    5121: np.uint8,
    5122: np.int16,
    5123: np.uint16,
    5125: np.uint32,
    5126: np.float32,
}
WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

# (clip, times as fractions of its length): standing still, grazing, walking, running.
POSES = [
    ("Idle", [0.0, 0.5]),
    ("Idle_2", [0.3]),
    ("Idle_Headlow", [0.3, 0.7]),
    ("Idle_2_HeadLow", [0.3, 0.7]),
    ("Eating", [0.2, 0.5, 0.8]),
    ("Walk", [0.0, 0.25, 0.5, 0.75]),
    ("Gallop", [0.0, 0.33, 0.66]),
]


def part_of(bone: str) -> int:
    """Temperature part of a bone: head and ears; the thin legs below the body (forearm
    and gaskin down) and the tail; the rest is torso, hips and thighs included."""
    if bone.startswith(("Head", "Ear")):
        return HEAD
    if bone.startswith(("FrontUpperLeg", "BackUpperLeg", "Tail", "FF", "IK", "Pole")):
        return LIMB
    if "LowerLeg" in bone:
        return LIMB
    return TORSO


def part_of_material(name: str, cold: tuple[str, ...] = ()) -> int | None:
    """Horns, hooves and antlers are cold whatever bone moves them. `cold` names a
    model's other cold materials (Stag's antlers are a plain "Material.001")."""
    lower = name.lower()
    if name in cold or "horn" in lower or "hoove" in lower or "hoof" in lower:
        return LIMB
    return None


@dataclass
class Gltf:
    doc: dict
    bin: bytes

    @classmethod
    def read(cls, path: Path) -> "Gltf":
        data = path.read_bytes()
        magic, _, _ = struct.unpack_from("<4sII", data, 0)
        if magic != b"glTF":
            raise ValueError(f"{path} is not a GLB file")
        json_len, kind = struct.unpack_from("<I4s", data, 12)
        assert kind == b"JSON"
        doc = json.loads(data[20 : 20 + json_len])
        rest = 20 + json_len
        bin_len, kind = struct.unpack_from("<I4s", data, rest)
        assert kind == b"BIN\x00"
        return cls(doc, data[rest + 8 : rest + 8 + bin_len])

    def accessor(self, i: int) -> np.ndarray:
        a = self.doc["accessors"][i]
        view = self.doc["bufferViews"][a["bufferView"]]
        dtype = np.dtype(COMPONENT[a["componentType"]])
        width = WIDTH[a["type"]]
        offset = view.get("byteOffset", 0) + a.get("byteOffset", 0)
        stride = view.get("byteStride") or dtype.itemsize * width
        arr = np.ndarray(
            (a["count"], width),
            dtype=dtype,
            buffer=self.bin,
            offset=offset,
            strides=(stride, dtype.itemsize),
        ).copy()
        if a.get("normalized"):
            arr = arr.astype(np.float64) / np.iinfo(dtype).max
        return arr


def quat_matrix(q) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def trs(t, r, s) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = quat_matrix(np.asarray(r, float)) * np.asarray(s, float)
    m[:3, 3] = t
    return m


def slerp(a, b, u: float) -> np.ndarray:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if np.dot(a, b) < 0:
        b = -b
    d = np.clip(np.dot(a, b), -1.0, 1.0)
    if d > 0.9995:
        q = a + u * (b - a)
        return q / np.linalg.norm(q)
    th = np.arccos(d)
    return (np.sin((1 - u) * th) * a + np.sin(u * th) * b) / np.sin(th)


def sample(times: np.ndarray, values: np.ndarray, interp: str, t: float, rotation: bool):
    if interp == "CUBICSPLINE":
        values = values.reshape(len(times), 3, -1)[:, 1]  # keep the values, drop tangents
        interp = "LINEAR"
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    k = int(np.searchsorted(times, t)) - 1
    if interp == "STEP":
        return values[k]
    u = (t - times[k]) / (times[k + 1] - times[k])
    if rotation:
        return slerp(values[k], values[k + 1], u)
    return values[k] + u * (values[k + 1] - values[k])


class Model:
    def __init__(self, gltf: Gltf, cold: tuple[str, ...] = ()):
        self.g = gltf
        self.cold = cold
        nodes = gltf.doc["nodes"]
        self.parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}
        self.rest = [
            (
                np.asarray(n.get("translation", [0, 0, 0]), float),
                np.asarray(n.get("rotation", [0, 0, 0, 1]), float),
                np.asarray(n.get("scale", [1, 1, 1]), float),
            )
            for n in nodes
        ]
        self.clips = {
            a.get("name", f"clip{i}"): a for i, a in enumerate(gltf.doc.get("animations", []))
        }

    def clip_length(self, name: str) -> float:
        a = self.clips[name]
        return max(float(self.g.accessor(s["input"]).max()) for s in a["samplers"])

    def globals(self, clip: str | None, t: float) -> list[np.ndarray]:
        local = [list(x) for x in self.rest]
        if clip is not None:
            a = self.clips[clip]
            for ch in a["channels"]:
                path = ch["target"]["path"]
                if path not in ("translation", "rotation", "scale"):
                    continue
                s = a["samplers"][ch["sampler"]]
                v = sample(
                    self.g.accessor(s["input"])[:, 0],
                    self.g.accessor(s["output"]).astype(float),
                    s.get("interpolation", "LINEAR"),
                    t,
                    path == "rotation",
                )
                local[ch["target"]["node"]][("translation", "rotation", "scale").index(path)] = v
        nodes = self.g.doc["nodes"]
        out: list[np.ndarray | None] = [None] * len(nodes)

        def world(i: int) -> np.ndarray:
            if out[i] is None:
                n = nodes[i]
                m = (
                    np.array(n["matrix"], float).reshape(4, 4).T
                    if "matrix" in n
                    else trs(*local[i])
                )
                out[i] = world(self.parent[i]) @ m if i in self.parent else m
            return out[i]

        return [world(i) for i in range(len(nodes))]

    def bone_part(self, node: int) -> int:
        """Part of a node, from its own name or its nearest named-bone ancestor."""
        nodes = self.g.doc["nodes"]
        while True:
            name = nodes[node].get("name", "")
            p = part_of(name)
            if p != TORSO or node not in self.parent:
                return p
            node = self.parent[node]

    def posed(self, clip: str | None, t: float):
        """Vertices (n, 3), per-vertex parts and triangles (m, 3) of the posed model."""
        doc = self.g.doc
        mats = self.globals(clip, t)
        verts, parts, tris = [], [], []
        for ni, node in enumerate(doc["nodes"]):
            if "mesh" not in node:
                continue
            skin = doc["skins"][node["skin"]] if "skin" in node else None
            if skin is not None:
                ibm = self.g.accessor(skin["inverseBindMatrices"]).reshape(-1, 4, 4)
                ibm = ibm.transpose(0, 2, 1)
                joint_mats = np.stack(
                    [mats[j] @ b for j, b in zip(skin["joints"], ibm, strict=True)]
                )
                joint_part = np.array([self.bone_part(j) for j in skin["joints"]])
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                pos = self.g.accessor(prim["attributes"]["POSITION"]).astype(float)
                homo = np.hstack([pos, np.ones((len(pos), 1))])
                if skin is not None:
                    joints = self.g.accessor(prim["attributes"]["JOINTS_0"]).astype(int)
                    weights = self.g.accessor(prim["attributes"]["WEIGHTS_0"]).astype(float)
                    weights /= weights.sum(1, keepdims=True)
                    m = np.einsum("nk,nkij->nij", weights, joint_mats[joints])
                    world = np.einsum("nij,nj->ni", m, homo)[:, :3]
                    part = joint_part[joints[np.arange(len(pos)), weights.argmax(1)]]
                else:
                    world = (homo @ mats[ni].T)[:, :3]
                    part = np.full(len(pos), self.bone_part(ni))
                mat_name = ""
                if "material" in prim:
                    mat_name = doc["materials"][prim["material"]].get("name", "")
                forced = part_of_material(mat_name, self.cold)
                if forced is not None:
                    part = np.full(len(pos), forced)
                idx = (
                    self.g.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
                    if "indices" in prim
                    else np.arange(len(pos)).reshape(-1, 3)
                )
                tris.append(idx + sum(len(v) for v in verts))
                verts.append(world)
                parts.append(part)
        return np.vstack(verts), np.concatenate(parts), np.vstack(tris)


def triangle_parts(parts: np.ndarray, tris: np.ndarray) -> np.ndarray:
    """Majority part of each triangle's corners (torso when all three differ)."""
    p = parts[tris]
    out = np.full(len(tris), TORSO)
    for k in (LIMB, HEAD, TORSO):
        out[(p == k).sum(1) >= 2] = k
    return out


@dataclass
class Frame:
    """Turns raw model space into ours: face +z, feet at 0, nose-to-rump `length_m`."""

    rotation: np.ndarray
    centre: np.ndarray
    scale: float

    @classmethod
    def from_rest(cls, model: Model, length_m: float) -> "Frame":
        verts, parts, _ = model.posed(None, 0.0)
        body = verts[parts != LIMB]  # no tail or legs in the length
        head = verts[parts == HEAD].mean(0)
        middle = body.mean(0)
        forward = head - middle
        forward[1] = 0.0
        forward /= np.linalg.norm(forward)
        # rotate about y so `forward` becomes +z
        yaw = np.arctan2(forward[0], forward[2])
        c, s = np.cos(-yaw), np.sin(-yaw)
        rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        turned = body @ rot.T
        length = turned[:, 2].max() - turned[:, 2].min()
        allv = verts @ rot.T
        centre = np.array(
            [
                (allv[:, 0].max() + allv[:, 0].min()) / 2,
                0.0,
                (turned[:, 2].max() + turned[:, 2].min()) / 2,
            ]
        )
        return cls(rot, centre, length_m / length)

    def apply(self, verts: np.ndarray) -> np.ndarray:
        v = (verts @ self.rotation.T - self.centre) * self.scale
        v[:, 1] -= v[:, 1].min()
        return v


def bake(
    path: Path, length_m: float, cold: tuple[str, ...] = ()
) -> list[tuple[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Posed parts of one model: [(pose name, [(vertices, triangles) per part])]."""
    model = Model(Gltf.read(path), cold)
    frame = Frame.from_rest(model, length_m)
    out = []
    for clip, fractions in POSES:
        if clip not in model.clips:
            continue
        length = model.clip_length(clip)
        for f in fractions:
            verts, parts, tris = model.posed(clip, f * length)
            verts = frame.apply(verts)
            tparts = triangle_parts(parts, tris)
            split = []
            for k in (TORSO, LIMB, HEAD):
                sel = tris[tparts == k]
                used, inverse = np.unique(sel, return_inverse=True)
                split.append(
                    (verts[used].astype(np.float32), inverse.reshape(-1, 3).astype(np.uint32))
                )
            out.append((f"{clip}@{f:.2f}", split))
    return out


def write_body(path: Path, poses) -> None:
    with path.open("wb") as f:
        f.write(MAGIC + struct.pack("<II", VERSION, len(poses)))
        for name, parts in poses:
            raw = name.encode()
            f.write(struct.pack("<B", len(raw)) + raw)
            for verts, tris in parts:
                f.write(struct.pack("<II", len(verts), tris.size))
                f.write(np.ascontiguousarray(verts, "<f4").tobytes())
                f.write(np.ascontiguousarray(tris, "<u4").tobytes())


def read_body(path: Path):
    """Inverse of `write_body` (for tests and checks)."""
    data = path.read_bytes()
    assert data[:4] == MAGIC
    version, n = struct.unpack_from("<II", data, 4)
    assert version == VERSION
    at, poses = 12, []
    for _ in range(n):
        ln = data[at]
        name = data[at + 1 : at + 1 + ln].decode()
        at += 1 + ln
        parts = []
        for _ in range(3):
            nv, ni = struct.unpack_from("<II", data, at)
            at += 8
            verts = np.frombuffer(data, "<f4", nv * 3, at).reshape(-1, 3)
            at += nv * 12
            tris = np.frombuffer(data, "<u4", ni, at).reshape(-1, 3)
            at += ni * 4
            parts.append((verts, tris))
        poses.append((name, parts))
    return poses
