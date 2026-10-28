import numpy as np

from aerial_animal_rescue.bodies import (
    HEAD,
    LIMB,
    TORSO,
    part_of,
    part_of_material,
    quat_matrix,
    read_body,
    sample,
    triangle_parts,
    write_body,
)


def test_bones_map_to_temperature_parts():
    assert part_of("Head") == HEAD
    assert part_of("Ear3.L") == HEAD
    assert part_of("FrontLowerLeg.R") == LIMB
    assert part_of("BackUpperLeg.L") == LIMB
    assert part_of("Tail4") == LIMB
    # hips, thighs and shoulders are body mass, as warm as the torso
    assert part_of("BackLeg.L") == TORSO
    assert part_of("FrontShoulder.R") == TORSO
    assert part_of("Neck2") == TORSO


def test_horns_hooves_and_named_materials_are_cold():
    assert part_of_material("Horns") == LIMB
    assert part_of_material("Hooves") == LIMB
    assert part_of_material("Material.001", ("Material.001",)) == LIMB
    assert part_of_material("Main") is None


def test_triangles_take_the_majority_part():
    parts = np.array([TORSO, LIMB, LIMB, HEAD])
    tris = np.array([[0, 1, 2], [0, 1, 3], [3, 3, 0]])
    assert triangle_parts(parts, tris).tolist() == [LIMB, TORSO, HEAD]


def test_quaternion_turns_a_quarter_about_y():
    s = np.sqrt(0.5)
    m = quat_matrix(np.array([0.0, s, 0.0, s]))
    assert np.allclose(m @ [0, 0, 1], [1, 0, 0])
    assert np.allclose(m @ m.T, np.eye(3))


def test_keyframes_interpolate_and_clamp():
    times = np.array([0.0, 1.0])
    values = np.array([[0.0, 0.0, 0.0], [2.0, 4.0, 6.0]])
    assert np.allclose(sample(times, values, "LINEAR", 0.5, False), [1, 2, 3])
    assert np.allclose(sample(times, values, "STEP", 0.5, False), [0, 0, 0])
    assert np.allclose(sample(times, values, "LINEAR", 3.0, False), [2, 4, 6])


def test_body_files_round_trip(tmp_path):
    tri = (
        np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32),
        np.array([[0, 1, 2]], np.uint32),
    )
    empty = (np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint32))
    path = tmp_path / "x.body"
    write_body(path, [("Walk@0.25", [tri, empty, tri])])
    [(name, parts)] = read_body(path)
    assert name == "Walk@0.25"
    assert np.array_equal(parts[0][0], tri[0])
    assert np.array_equal(parts[2][1], tri[1])
    assert len(parts[1][0]) == 0
