"""The edit's running order: a cold open, a title card, the story cut shot by shot, an end screen.

Shots cut hard on the narration's own sentence breaks; the only fades are into the story
after the title and out of it before the end screen. Segment lengths are counted in whole
frames from the programme's running time, so picture and sound never drift apart. The
stills edit (production/stills/post.py) renders each segment.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter

from . import draw
from .look import W, H
from .shots import Shot

log = logging.getLogger(__name__)

FPS = 24


def _ff(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-800:]}")


def probe(path: Path) -> float:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    try:
        return float(result.stdout.strip())
    except ValueError:
        return 0.0


@dataclass
class Segment:
    name: str  # output file inside the work folder
    start: float  # programme time
    end: float
    picture: Path | None = None
    camera: str = "none"  # the camera move over it (production/stills/camera.py)
    clip: Path | None = None  # a moving clip instead of the still (production/stills/motion.py)
    atmosphere: tuple[str, ...] = ()  # rain, fog, flicker
    fade_in: float = 0.0
    fade_out: float = 0.0
    tension: int = 2  # the beat's tension: how fast the camera moves
    dim: float = 1.0  # below 1: the picture darkened (a scare's build-up, before the reveal)
    loop: bool = False  # the clip plays round and round at its own speed (the host's loop), never slowed or frozen
    layers: tuple[Path, Path] | None = None  # (background plate, foreground RGBA): a depth parallax shot

    def frames(self) -> int:
        return max(round(self.end * FPS) - round(self.start * FPS), 1)


def card(background: Path | None, lines: list[tuple[str, int, tuple[int, int, int]]], out: Path,
         left: bool = False, dim: float = 0.35, size: tuple[int, int] = (W, H), anchor: str = "center") -> Path:
    """A title or end card: the style frame dimmed and blurred, with bone-white text. `anchor`:
    where the words sit ("center", "lower" or "upper"), so every episode's card is not the same."""
    w, h = size
    if background and background.exists():
        with Image.open(background) as im:
            base = im.convert("RGB").resize((w, h), Image.LANCZOS)
        base = ImageEnhance.Brightness(base.filter(ImageFilter.GaussianBlur(3 * h / 1080))).enhance(dim)
    else:
        base = Image.new("RGB", (w, h), (8, 8, 8))
    d = ImageDraw.Draw(base)
    total = sum(int(points * 1.25) for _, points, _ in lines)
    y = {"lower": int(h * 0.70) - total // 2, "upper": int(h * 0.16)}.get(anchor, (h - total) // 2)
    y = min(max(y, int(h * 0.06)), h - total - int(h * 0.06))
    for text, points, colour in lines:
        f = draw.font(points)
        width = d.textlength(text, font=f)
        x = int(w * 0.08) if left else int((w - width) / 2)
        d.text((x, y), text, font=f, fill=colour, stroke_width=max(points // 18, 2), stroke_fill=(0, 0, 0))
        y += int(points * 1.25)
    base.save(out)
    return out


def wrap(text: str, width: int = 26) -> list[str]:
    words, lines, line = text.split(), [], ""
    for w in words:
        if len(line) + len(w) + 1 > width and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    if line:
        lines.append(line)
    return lines[:3]


def programme(shots: list[Shot], pictures: dict[int, Path], cold: list[int],
              cold_seconds: float, title_seconds: float, end_seconds: float, story_seconds: float,
              title_card: Path, end_card: Path) -> list[Segment]:
    """The running order. Programme time = cold open, then title, then the story, then the end screen."""
    segs: list[Segment] = []
    t = 0.0
    if cold:
        first = shots[cold[0]].t0
        for k, i in enumerate(cold):
            s = shots[i]
            a, b = s.t0 - first, (shots[cold[k + 1]].t0 - first) if k + 1 < len(cold) else cold_seconds
            segs.append(Segment(f"seg_cold_{k}.mp4", a, b, pictures.get(i),
                                fade_in=0.2 if k == 0 else 0.0))
        t = cold_seconds
    if title_seconds:
        segs.append(Segment("seg_title.mp4", t, t + title_seconds, title_card))
        t += title_seconds
    offset = t
    for k, s in enumerate(shots):
        segs.append(Segment(f"seg_{s.index:03d}.mp4", s.t0 + offset, s.t1 + offset, pictures.get(s.index), fade_in=0.4 if k == 0 else 0.0,
                            fade_out=1.0 if k == len(shots) - 1 else 0.0))
    t = offset + story_seconds
    if end_seconds:
        segs.append(Segment("seg_end.mp4", t, t + end_seconds, end_card, fade_in=0.5,
                            fade_out=1.5))
    return segs


def pick_cold_open(shots: list[Shot], scenes: list[dict[str, Any]], min_s: float = 6.0,
                   max_s: float = 14.0, host: set[int] | frozenset[int] = frozenset(),
                   before: float = 0.65) -> list[int]:
    """The scariest moment, before the ending: a scare (or the most important shot) in a key scene,
    with the shots after it until 6-14 s are covered. Never a spoiler: never the host's frames, never
    the story's final scene, and only from the first `before` of the story's running time, since a
    twist or reveal sits in the last third (run 113 opened on the guest book signed with the
    teller's own name, the story's last line, so the ending was given away in the first second)."""
    if not shots:
        return []
    story = [s for s in shots if s.index not in host] or shots
    last_scene = story[-1].scene
    start_t, end_t = story[0].t0, story[-1].t1
    cutoff = start_t + before * max(end_t - start_t, 0.0)
    pool = ([s for s in story if s.scene != last_scene and s.t0 > 20 and s.t0 < cutoff]
            or [s for s in story if s.scene != last_scene and s.t0 < cutoff]
            or [s for s in story if s.scene != last_scene] or story)
    def rank(s: Shot) -> tuple:
        # What the viewer sees of the threat counts most: run 115 opened on a boarded door the storyboard
        # had called its one scare, over the figure leaning over her shoulder (threat 3) a minute later.
        key = bool(scenes[s.scene].get("key_moment")) if s.scene < len(scenes) else False
        weight = 3 * s.threat + s.tension + s.importance + 2 * s.scare + key
        return (-weight, s.index)
    start = sorted(pool, key=rank)[0]
    chosen = [start.index]
    covered = start.seconds
    i = start.index + 1
    while covered < min_s and i < len(shots) and shots[i].scene == start.scene:
        if covered + shots[i].seconds > max_s:
            break
        chosen.append(i)
        covered += shots[i].seconds
        i += 1
    return chosen
