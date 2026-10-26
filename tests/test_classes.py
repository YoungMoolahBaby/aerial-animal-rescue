import pytest

from aerial_animal_rescue.classes import ClassMapper, ClassRegistry
from tests.conftest import REPO


def test_classes_yaml(registry):
    assert registry.names == (
        "person",
        "dog",
        "cat",
        "horse",
        "cattle",
        "small_livestock",
        "other_animal",
    )
    assert registry.id_of("cattle") == 4


def test_mapping_and_drop_logging(registry):
    mapper = ClassMapper(registry, {"Cow": "cattle", "Person": "person", "Car": None})
    assert mapper.map("cow") == 4  # case-insensitive
    assert mapper.map("Person") == 0
    assert mapper.map("Car") is None  # dropped on purpose
    assert mapper.map("Bicycle") is None  # not listed -> dropped
    assert mapper.dropped == {"Car": 1, "Bicycle": 1}
    assert mapper.kept == {"cattle": 1, "person": 1}
    assert "Bicycle: 1" in mapper.report()


def test_unknown_target_class_is_an_error(registry):
    with pytest.raises(KeyError):
        ClassMapper(registry, {"cow": "cows"})


def test_non_contiguous_ids_rejected(tmp_path):
    path = tmp_path / "classes.yaml"
    path.write_text("classes:\n  - {id: 0, name: a}\n  - {id: 2, name: b}\n")
    with pytest.raises(ValueError):
        ClassRegistry.from_yaml(path)


def test_sources_yaml_class_maps_are_valid(registry):
    from aerial_animal_rescue.sources import load_sources

    for src in load_sources(REPO / "configs" / "sources.yaml").values():
        ClassMapper(registry, src.class_map)  # raises on a typo in a target name
