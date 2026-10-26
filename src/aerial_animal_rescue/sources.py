"""External dataset registry (configs/sources.yaml) and downloading.

Rules:
  - Sources marked `download: manual` (or with no file URLs) only print instructions.
  - Anything over 1 GB, or of unknown size, asks for confirmation first.
  - Failures are reported, never fatal for the other sources.
"""

import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tqdm import tqdm

from aerial_animal_rescue.config import EXTERNAL_DIR, SOURCES_YAML, load_yaml

MAX_UNASKED_BYTES = 1 * 1024**3  # 1 GB


@dataclass
class Source:
    name: str
    title: str
    url: str
    license: str
    redistribution: str
    download: str
    role: str
    modality: str
    files: list[str] = field(default_factory=list)
    size_gb: float | None = None
    manual_instructions: str = ""
    converter: dict[str, Any] = field(default_factory=dict)
    class_map: dict[str, str | None] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def is_automatic(self) -> bool:
        return self.download == "automatic" and bool(self.files)


def load_sources(path: str | Path = SOURCES_YAML) -> dict[str, Source]:
    sources = {}
    for name, d in (load_yaml(path)["sources"] or {}).items():
        sources[name] = Source(
            name=name,
            title=d.get("title", name),
            url=d.get("url", "TODO"),
            license=str(d.get("license", "TODO")),
            redistribution=str(d.get("redistribution", "TODO")),
            download=d.get("download", "manual"),
            role=d.get("role", "annotations"),
            modality=d.get("modality", "rgb"),
            files=list(d.get("files") or []),
            size_gb=d.get("size_gb"),
            manual_instructions=d.get("manual_instructions", ""),
            converter=dict(d.get("converter") or {}),
            class_map=dict(d.get("class_map") or {}),
            raw=d,
        )
    return sources


def manual_instructions(src: Source, dest: Path) -> str:
    text = src.manual_instructions.strip() or f"Download it from {src.url}"
    return (
        f"[{src.name}] manual download needed - {src.title}\n"
        f"  landing page: {src.url}\n  license: {src.license}\n"
        f"  put the files in: {dest}\n" + "\n".join("  " + ln for ln in text.splitlines())
    )


def remote_size(url: str) -> int | None:
    """Content-Length via a HEAD request (follows redirects); None if unknown."""
    req = urllib.request.Request(url, method="HEAD")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            length = resp.headers.get("Content-Length")
            return int(length) if length else None
    except Exception:
        return None


def confirm(prompt: str) -> bool:
    try:
        return input(f"{prompt} [y/N] ").strip().lower() in {"y", "yes"}
    except EOFError:  # non-interactive: never assume yes
        return False


def download_file(url: str, dest: Path, size: int | None) -> None:
    """Stream `url` to `dest` with a progress bar (via a .part file)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    with (
        urllib.request.urlopen(url, timeout=60) as resp,
        open(part, "wb") as out,
        tqdm(total=size, unit="B", unit_scale=True, desc=dest.name) as bar,
    ):
        while chunk := resp.read(1 << 20):
            out.write(chunk)
            bar.update(len(chunk))
    part.replace(dest)


def filename_for(url: str) -> str:
    """Pick a file name from a URL (handles Zenodo's `.../files/<name>/content`)."""
    parts = [p for p in url.split("?")[0].split("/") if p]
    if parts and parts[-1] == "content" and len(parts) > 1:
        return parts[-2]
    return parts[-1]


def fetch_source(src: Source, root: Path = EXTERNAL_DIR, assume_yes: bool = False) -> bool:
    """Download one source into `root/<name>/`. Returns True if everything is present."""
    dest_dir = root / src.name
    if not src.is_automatic:
        print(manual_instructions(src, dest_dir))
        return False
    ok = True
    for url in src.files:
        dest = dest_dir / filename_for(url)
        if dest.exists():
            print(f"[{src.name}] already have {dest}")
            continue
        size = remote_size(url)
        if size is None or size > MAX_UNASKED_BYTES:
            human = "unknown size" if size is None else f"{size / 1024**3:.1f} GB"
            question = f"[{src.name}] {dest.name} is {human} (license {src.license}). Download?"
            if not (assume_yes or confirm(question)):
                print(f"[{src.name}] skipped {url}")
                ok = False
                continue
        try:
            download_file(url, dest, size)
        except Exception as e:  # report and continue with the next file
            print(f"[{src.name}] FAILED {url}: {e}\n{manual_instructions(src, dest_dir)}")
            ok = False
    return ok
