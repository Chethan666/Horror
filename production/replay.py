"""Reruns that repeat themselves: every model answer a video depends on, kept on disk.

A video is made from many decisions: the storyboard, the designs, the sound picks, the
performance tags (the planner model) and the checker's verdicts (the vision model). Pictures,
voices and sounds are already cached by their exact request (production/paid.py), but the
requests themselves come from those decisions, and a model asked twice decides differently.
So a rerun of a failed video used to buy every picture again.

Here every planner answer and every text answer from a paid model (the checker, the judge,
the screening) is saved under its request. A rerun with the same cache folder gets the same
answers, so it asks for the same pictures and voices, and the studio's cache hands them back:
nothing is paid twice. The folder is `MEDIA_CACHE_DIR` (the workflow keeps it as an artifact
for a few days, so `resume=<run number>` picks a failed run up where it stopped, and
`redo=<beats>` repaints only the pictures a person asked for).

`pack()` copies a cache for keeping: pictures become high-quality JPEGs under the same names
(a tenth of the size); everything else is copied as it is.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from .paid import digest

log = logging.getLogger(__name__)


def _write(path: Path, data: Any) -> None:
    """Written whole or not at all: parallel workers and a killed run never leave half a file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def _read(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def _is_text(result: Any) -> bool:
    """A text answer (a verdict, a judgement): pictures, sounds and clips are cached as files."""
    return (isinstance(result, dict) and isinstance(result.get("output") or result.get("text"), str)
            and not any(k in result for k in ("images", "image", "audio", "video")))


class Replay:
    """The media client with its text answers kept: the same request gets the same answer."""

    def __init__(self, inner: Any, folder: Path):
        self.inner, self.folder = inner, folder
        self.replayed = 0

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        path = self.folder / f"{digest(endpoint, args)}.json"
        kept = _read(path) if path.exists() else None
        if isinstance(kept, dict):
            self.replayed += 1
            return kept
        result = self.inner.run(endpoint, args)
        if _is_text(result):
            _write(path, result)
        return result

    def __getattr__(self, name: str) -> Any:  # fell_back, schema(), price(): the client's own
        return getattr(self.inner, name)


class _Planner:
    def __init__(self, client: Any, folder: Path, role: str, owner: "ReplayRouter"):
        self.client, self.folder, self.role, self.owner = client, folder, role, owner

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        path = self.folder / f"{task}-{digest(self.role, task, system, prompt, schema)}.json"
        kept = _read(path) if path.exists() else None
        if isinstance(kept, (dict, list)):
            self.owner.replayed += 1
            return kept
        result = self.client.generate_json(task, system, prompt, schema)
        if isinstance(result, (dict, list)):
            _write(path, result)
        return result

    def __getattr__(self, name: str) -> Any:
        return getattr(self.client, name)


class ReplayRouter:
    """The story factory's router with the planner's answers kept, request by request."""

    def __init__(self, router: Any, folder: Path):
        self.router, self.folder = router, folder
        self.replayed = 0

    def one(self, role: str) -> _Planner:
        return _Planner(self.router.one(role), self.folder, role, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.router, name)


def cache_root(cfg: dict[str, Any], work: Path) -> Path:
    """Where paid results and model answers are kept: MEDIA_CACHE_DIR (kept between runs by the
    workflow), else `stills.cache_dir`, else the work folder (deleted with it)."""
    chosen = os.environ.get("MEDIA_CACHE_DIR") or str(cfg.get("cache_dir") or "")
    return Path(chosen).resolve() if chosen else work / "cache"


def pack(src: Path, dst: Path, quality: int = 95, min_bytes: int = 1_000_000) -> dict[str, Any]:
    """A copy of the cache to keep: large PNG pictures as JPEGs under the same names (Pillow and
    FFmpeg read a picture by its content, not its name), the rest unchanged."""
    from PIL import Image

    files = shrunk = 0
    before = after = 0
    if not src.is_dir():
        return {"files": 0, "mb": 0.0}
    for path in sorted(p for p in src.rglob("*") if p.is_file() and not p.name.endswith(".tmp")):
        out = dst / path.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        size = path.stat().st_size
        before += size
        files += 1
        if path.suffix.lower() == ".png" and size >= min_bytes:
            try:
                with Image.open(path) as im:
                    im.convert("RGB").save(out, "JPEG", quality=quality)
                shrunk += 1
                after += out.stat().st_size
                continue
            except OSError:
                pass  # not a picture after all: kept as it is
        shutil.copy2(path, out)
        after += size
    return {"files": files, "pictures_shrunk": shrunk, "mb_before": round(before / 1e6, 1),
            "mb": round(after / 1e6, 1)}


def parse_redo(text: str) -> dict[int, str]:
    """Pictures a person asked to have repainted, from the workflow's overrides:
    "12,31:no_text_on_the_wall" -> {12: "", 31: "no text on the wall"} (underscores are spaces,
    since overrides are split at spaces). A beat with no note is simply painted again."""
    out: dict[int, str] = {}
    for item in (text or "").replace(";", ",").split(","):
        item = item.strip()
        if not item:
            continue
        number, _, note = item.partition(":")
        try:
            out[int(number)] = note.replace("_", " ").strip()
        except ValueError:
            log.warning("redo: %r is not a beat number; ignored", item)
    return out
