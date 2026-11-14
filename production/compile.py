"""Compilations: finished episodes joined into one long video.

Long "scary stories to fall asleep to" videos are one of horror YouTube's largest niches, and the
channel already owns everything they need. Each episode is cut where its end screen begins (a
segment boundary in our own encode, so the picture is copied, not re-encoded), a short card
naming the next story sits between them, and the sound is re-encoded once for the whole film.
The description lists every story with its time stamp (YouTube makes them chapters) and keeps
every music credit. The episodes come from their production runs' artifacts (kept 14 days), or
any folders holding video.mp4 and metadata.json.
"""

from __future__ import annotations

import json
import logging
import subprocess
from pathlib import Path
from typing import Any

from . import draw, edit
from .shots import stamp

log = logging.getLogger(__name__)

CARD_SECONDS = 4.0


def _ff(args: list[str], cwd: Path) -> None:
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", *args], cwd=cwd,
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed: {result.stderr.strip()[-600:]}")


def _size(video: Path) -> tuple[int, int]:
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height",
                            "-of", "csv=p=0", str(video)], capture_output=True, text=True).stdout.strip()
    w, h = (int(x) for x in probe.split(",")[:2])
    return w, h


def card_clip(text: str, background: Path | None, out: Path, size: tuple[int, int], work: Path) -> Path:
    """A few seconds between two stories: the next one's name over its own picture, dimmed, silent."""
    from .stills import camera
    from .stills.post import render

    k = size[1] / 1080
    picture = edit.card(background, [("NEXT STORY", round(40 * k), draw.RED), ("", round(20 * k), draw.CHALK)] +
                        [(line, round(84 * k), draw.CHALK) for line in edit.wrap(text, 26)],
                        work / f"{out.stem}.png", dim=0.22, size=size)
    seg = edit.Segment(str(work / f"{out.stem}_v.mp4"), 0.0, CARD_SECONDS, picture, camera="push_in",
                       fade_in=0.4, fade_out=0.4)
    render(seg, work, {"grain": 0.018, "vignette": 0.22}, 0, size)
    _ff(["-i", seg.name, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=48000", "-c:v", "copy",
         "-c:a", "aac", "-b:a", "192k", "-shortest", *camera.COLOUR_TAGS, str(out)], work)
    return out


def compile_episodes(folders: list[Path], out: Path, title: str = "") -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    parts: list[Path] = []
    chapters: list[tuple[float, str]] = []
    credits: list[str] = []
    t = 0.0
    size: tuple[int, int] | None = None
    for k, folder in enumerate(folders):
        meta = json.loads((folder / "metadata.json").read_text())
        video = folder / "video.mp4"
        size = size or _size(video)
        if k:
            thumb = next((p for p in (folder / "thumbnail_1.jpg",) if p.exists()), None)
            card = card_clip(str(meta.get("title", "")), thumb, work / f"card_{k:02d}.mp4", size, work)
            parts.append(card)
            t += CARD_SECONDS
        end = float(meta.get("end_screen_starts") or meta.get("duration_seconds") or 0.0)
        cut = work / f"part_{k:02d}.mp4"
        # the end screen starts on a segment of its own, so this cut falls on a key frame: the picture is copied
        _ff(["-i", str(video), "-t", f"{end:.3f}", "-c", "copy", str(cut)], work)
        chapters.append((t, str(meta.get("title", f"Story {k + 1}"))))
        parts.append(cut)
        t += end
        credits += [c for c in meta.get("sound_credits") or [] if c not in credits]
    (work / "parts.txt").write_text("".join(f"file '{p.name}'\n" for p in parts))
    film = out / "compilation.mp4"
    _ff(["-f", "concat", "-safe", "0", "-i", "parts.txt", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
         "-ar", "48000", "-ac", "2", "-movflags", "+faststart", str(film.resolve())], work)
    name = title or f"{len(folders)} Horror Stories to Listen to in the Dark"
    description = ("Every story here is fiction, told by The Storyteller.\n\n"
                   + "\n".join(f"{stamp(at)} {story}" for at, story in chapters)
                   + ("\n\nMusic and sound:\n" + "\n".join(credits) if credits else ""))
    meta = {"title": name, "description": description, "chapters": [[round(a, 1), s] for a, s in chapters],
            "duration_seconds": round(t, 1), "made_for_kids": False, "synthetic_media": True,
            "sound_credits": credits, "episodes": [str(f) for f in folders], "tags": ["horror", "scary stories"]}
    (out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    return {"video": str(film), "seconds": round(t, 1), "stories": len(folders)}
