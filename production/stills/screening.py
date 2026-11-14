"""The screening: the finished cut watched, with its sound, before a person reviews it.

Every other check looks at one piece: a picture, a take, a sheet, the loudness. Nobody watched
the film. Here a model that watches video (Gemini on Vertex AI) sees a small copy of the finished
episode, one frame a second with its sound, and reports like a test audience and an editor: where
a viewer would get bored or confused, where a picture fights its line, whether each scare lands,
and where a typical viewer would leave. Each note is pinned to the beat on screen at that moment
(beat_times), so a person can have exactly those pictures redone (the workflow's redo=...).

It never changes the video: it reports, and QA lists the serious notes. About $0.40 for a
10-minute episode on the stronger checker; nothing without a Google key (fal's checker cannot
watch video).
"""

from __future__ import annotations

import base64
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

from production.media import MediaError

from ..paid import OverBudget
from .board import Board

log = logging.getLogger(__name__)

SYSTEM = ("You watch finished episodes of Hollow Lines, a horror storytelling channel on YouTube, twice over: as "
          "its target viewer and as a horror film editor. You are honest and specific. Answer with JSON only.")


def proxy(video: Path, out: Path, height: int = 360, crf: int = 32) -> Path:
    """A small copy to watch: one frame a second (the model samples about that often anyway), quiet mono sound."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", str(video), "-vf",
                             f"fps=1,scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf),
                             "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "32k", "-ac", "1", str(out)],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise MediaError(f"screening copy failed: {result.stderr[-300:]}")
    return out


def seconds_of(stamp: Any) -> float | None:
    """'3:05', '1:02:03' or 185 -> seconds."""
    if isinstance(stamp, (int, float)):
        return float(stamp)
    parts = re.findall(r"\d+(?:\.\d+)?", str(stamp or ""))
    if not parts:
        return None
    total = 0.0
    for p in parts[-3:]:
        total = total * 60 + float(p)
    return total


def beat_at(t: float | None, times: list[dict[str, Any]]) -> int | None:
    if t is None:
        return None
    for row in times:
        if row["start"] - 0.25 <= t < row["end"] + 0.25:
            return int(row["beat"])
    return None


def prompt(title: str, seconds: float, opening: float) -> str:
    return (f"This is the finished episode \"{title}\" ({seconds / 60:.1f} minutes): a horror story told in still "
            "pictures under slow camera moves, with a narrator, music and sound effects. It opens on a flash-"
            f"forward of the scariest moment, then the title, then the host (a man with a lantern); the story itself "
            f"starts at about {opening:.0f} s. Watch it all, with the sound.\n\n"
            "Report what would make a viewer leave or stop believing it: a stretch that drags, a picture that "
            "does not show what is being said, a face that does not show the fear in the words, a person or place "
            "that changes look, a scare that does not land (and why), sound that fights the voice, an ending that "
            "falls flat. Only real problems, worst first, at most 12. Times as m:ss from the start of the video.\n"
            'Reply exactly as JSON: {"score": 1-10, "hook": "does the first 30 seconds hold you, and why", '
            '"notes": [{"at": "m:ss", "kind": "drags|confusing|picture|face|continuity|scare|sound|ending", '
            '"severity": 1-3, "note": "what is wrong and what would fix it"}], '
            '"scares": [{"at": "m:ss", "landed": true or false, "why": "short"}], '
            '"leave_at": "m:ss where a typical viewer would leave, or empty"}')


class ScreeningAgent:
    name = "screening"
    critical = False  # a report for the reviewer; the video stands without it

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        settings = cfg.get("screening") or {}
        studio = board.studio
        video: Path | None = d.get("video")
        if not settings.get("enabled", True) or studio is None or video is None or not Path(video).exists():
            return {"watched": False}
        role = "vision_pro" if str(studio.models.get("vision_pro", "")).startswith("vertex:") else "vision"
        model = str(studio.models.get(role, ""))
        if not model.startswith("vertex:"):
            return {"watched": False, "why": "only Gemini on Vertex AI can watch the video"}
        small = board.work / "screening.mp4"
        try:
            proxy(Path(video), small)
            if small.stat().st_size > float(settings.get("max_mb", 18)) * 1e6:
                proxy(Path(video), small, height=240, crf=38)
        except (MediaError, OSError) as exc:
            return {"watched": False, "why": str(exc)[:200]}
        seconds = float(d.get("total") or 0.0)
        units = max(1.0, seconds / 60.0 * 4)  # priced like four looks a minute (video and its sound)
        try:
            usd = studio.budget.reserve(model, units)
        except OverBudget:
            return {"watched": False, "why": "no budget left"}
        uri = "data:video/mp4;base64," + base64.b64encode(small.read_bytes()).decode()
        try:
            result = studio.fal.run(model, dict(studio.args.get(role) or {}, system_prompt=SYSTEM, video_url=uri,
                                                prompt=prompt(str((d.get("titles") or [board.story.get("title")])[0]),
                                                              seconds, float(d.get("offset") or 0.0)),
                                                max_tokens=4096))
        except MediaError as exc:
            studio.budget.settle(usd, "screening", model, ok=False)
            return {"watched": False, "why": str(exc)[:200]}
        studio.budget.settle(usd, "screening", model, ok=True, units=units)
        match = re.search(r"\{.*\}", str(result.get("output") or ""), re.S)
        try:
            verdict = json.loads(match.group(0)) if match else {}
        except ValueError:
            verdict = {}
        if not isinstance(verdict, dict) or not verdict:
            return {"watched": True, "readable": False}
        times = d.get("beat_times") or []
        notes = []
        for n in (verdict.get("notes") or [])[:12]:
            if not isinstance(n, dict) or not str(n.get("note", "")).strip():
                continue
            at = seconds_of(n.get("at"))
            try:
                severity = min(max(int(n.get("severity", 2)), 1), 3)
            except (TypeError, ValueError):
                severity = 2
            notes.append({"at": n.get("at"), "beat": beat_at(at, times), "kind": str(n.get("kind", ""))[:20],
                          "severity": severity, "note": str(n["note"])[:240]})
        scares = [{"at": s.get("at"), "beat": beat_at(seconds_of(s.get("at")), times), "landed": bool(s.get("landed")),
                   "why": str(s.get("why", ""))[:160]} for s in (verdict.get("scares") or []) if isinstance(s, dict)]
        try:
            score = max(1, min(10, int(verdict.get("score"))))
        except (TypeError, ValueError):
            score = None
        leave = verdict.get("leave_at") or ""
        return {"watched": True, "score": score, "hook": str(verdict.get("hook", ""))[:300], "notes": notes,
                "scares": scares, "leave_at": leave, "leave_beat": beat_at(seconds_of(leave), times) if leave else None,
                "redo_hint": ",".join(str(n["beat"]) for n in notes if n["severity"] == 3 and n["beat"] is not None
                                      and n["kind"] in ("picture", "face", "continuity"))}
