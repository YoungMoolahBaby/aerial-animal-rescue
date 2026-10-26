"""Target classes and mapping of source-dataset labels onto them."""

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from aerial_animal_rescue.config import CLASSES_YAML, load_yaml


@dataclass(frozen=True)
class ClassRegistry:
    """The ordered list of our target classes; list index == YOLO class id."""

    names: tuple[str, ...]

    @classmethod
    def from_yaml(cls, path: str | Path = CLASSES_YAML) -> "ClassRegistry":
        entries = sorted(load_yaml(path)["classes"], key=lambda c: c["id"])
        ids = [c["id"] for c in entries]
        if ids != list(range(len(ids))):
            raise ValueError(f"class ids must be contiguous from 0, got {ids}")
        names = tuple(c["name"] for c in entries)
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate class names in {path}")
        return cls(names)

    def id_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError:
            raise KeyError(f"unknown target class {name!r}; known: {self.names}") from None

    def __len__(self) -> int:
        return len(self.names)


@dataclass
class ClassMapper:
    """Maps a source dataset's class names onto our class ids.

    `class_map` comes from configs/sources.yaml: source name -> our class name,
    or null to drop it on purpose. Names not listed are dropped too. Every drop
    is counted so the conversion log shows exactly what was discarded.
    Matching is case-insensitive.
    """

    registry: ClassRegistry
    class_map: dict[str, str | None]
    dropped: Counter = field(default_factory=Counter)
    kept: Counter = field(default_factory=Counter)

    def __post_init__(self) -> None:
        self._map = {}
        for src, dst in self.class_map.items():
            self._map[str(src).lower()] = None if dst is None else self.registry.id_of(dst)

    def map(self, source_class: str | int) -> int | None:
        key = str(source_class).lower()
        target = self._map.get(key)
        if target is None:
            self.dropped[str(source_class)] += 1
        else:
            self.kept[self.registry.names[target]] += 1
        return target

    def report(self) -> str:
        lines = ["kept (our class: count):"]
        lines += [f"  {k}: {v}" for k, v in sorted(self.kept.items())] or ["  (none)"]
        lines.append("dropped (source class: count):")
        lines += [f"  {k}: {v}" for k, v in sorted(self.dropped.items())] or ["  (none)"]
        return "\n".join(lines)
