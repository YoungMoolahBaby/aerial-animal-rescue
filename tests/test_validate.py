import shutil

from aerial_animal_rescue.split import build_processed
from aerial_animal_rescue.validate import summary_tables, validate_dataset
from tests.conftest import make_pool


def processed(tmp_path, registry):
    rows = [
        {"image_id": f"own_{f}_{k}", "sequence_id": f"fl{f}"} for f in range(6) for k in range(2)
    ]
    pool = make_pool(tmp_path / "pool", rows)
    out = tmp_path / "processed"
    build_processed([pool], out, list(registry.names), seed=0)
    return out


def first(out, sub="labels"):
    return sorted((out / sub).rglob("*.*"))[0]


def test_clean_dataset_passes(tmp_path, registry):
    rep = validate_dataset(processed(tmp_path, registry), list(registry.names))
    assert rep.ok, rep.errors
    assert "dog" in summary_tables(rep)


def test_out_of_bounds_and_bad_class(tmp_path, registry):
    out = processed(tmp_path, registry)
    first(out).write_text("9 0.5 0.5 0.1 0.1\n0 0.95 0.5 0.2 0.2\n")
    errors = "\n".join(validate_dataset(out, list(registry.names)).errors)
    assert "class id 9" in errors
    assert "out of image bounds" in errors


def test_missing_label(tmp_path, registry):
    out = processed(tmp_path, registry)
    first(out).unlink()
    assert any("missing label" in e for e in validate_dataset(out, list(registry.names)).errors)


def test_duplicate_image(tmp_path, registry):
    out = processed(tmp_path, registry)
    a, b = sorted((out / "images").rglob("*.png"))[:2]
    shutil.copyfile(a, b)
    assert any("duplicate of" in e for e in validate_dataset(out, list(registry.names)).errors)


def test_duplicate_box(tmp_path, registry):
    out = processed(tmp_path, registry)
    first(out).write_text("1 0.5 0.5 0.2 0.2\n1 0.5 0.5 0.2 0.2\n")
    assert any("duplicate box" in e for e in validate_dataset(out, list(registry.names)).errors)
