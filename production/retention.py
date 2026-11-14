"""Where viewers left: YouTube's retention curve read against the episode's own beats.

The average percentage viewed says how much of a video people watched, not where they gave up.
YouTube Analytics reports the share of viewers still watching at each hundredth of a video (the
audience retention report). Laid over the beat times kept at production (`beat_times`: when each
picture was on screen and what was said over it), the steepest drops point at the moments that
lost people: a slow opening, a beat that dragged, a reveal that came too late. They become
written lessons the story factory's writers see with every new pitch (the learnings table), next
to the lessons its judges write from the best and worst performers.
"""

from __future__ import annotations

from typing import Any

PREFIX = "Retention:"


def drops(curve: list[tuple[float, float]], seconds: float, beat_times: list[dict[str, Any]],
          top: int = 3, skip: float = 0.03) -> list[dict[str, Any]]:
    """The steepest falls in the curve [(share of the video, share of viewers still watching)], each
    with the time and the beat on screen; the first `skip` of the video (everyone's early leavers)
    is left out. Largest first."""
    points = sorted((float(r), float(w)) for r, w in curve if r is not None and w is not None)
    found = []
    for (r0, w0), (r1, w1) in zip(points, points[1:]):
        if r0 < skip or r1 <= r0:
            continue
        fall = w0 - w1
        if fall <= 0:
            continue
        at = r0 * seconds
        beat = next((b for b in beat_times if b["start"] <= at < b["end"]), None)
        found.append({"at": round(at, 1), "fall": round(fall, 4), "beat": beat["beat"] if beat else None,
                      "line": beat.get("line", "") if beat else "", "tension": beat.get("tension") if beat else None,
                      "scare": bool(beat.get("scare")) if beat else False})
    found.sort(key=lambda x: -x["fall"])
    kept: list[dict[str, Any]] = []
    for d in found:  # one drop per stretch: neighbours of a kept drop are the same moment
        if all(abs(d["at"] - k["at"]) > 0.03 * seconds for k in kept):
            kept.append(d)
        if len(kept) >= top:
            break
    return kept


def still_watching(curve: list[tuple[float, float]], at_ratio: float) -> float | None:
    points = sorted(curve)
    for r, w in points:
        if r >= at_ratio:
            return float(w)
    return float(points[-1][1]) if points else None


def stamp(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 60}:{seconds % 60:02d}"


def lessons(title: str, curve: list[tuple[float, float]], seconds: float,
            beat_times: list[dict[str, Any]]) -> tuple[list[str], dict[str, Any]]:
    """Written lessons from one episode's curve, and what was measured."""
    if not curve or seconds <= 0:
        return [], {}
    hook = still_watching(curve, min(30.0 / seconds, 1.0))
    worst = drops(curve, seconds, beat_times)
    out = []
    if hook is not None:
        out.append(f"{PREFIX} \"{title}\": {hook:.0%} of viewers were still watching at 0:30 "
                   f"({'a strong' if hook >= 0.7 else 'a weak'} opening).")
    for d in worst:
        what = f"beat {d['beat']} (\"{d['line'][:70]}\"" + (", a scare" if d["scare"] else
                                                             f", tension {d['tension']}" if d["tension"] else "") + ")"
        out.append(f"{PREFIX} \"{title}\": the steepest drop was at {stamp(d['at'])}, during {what}: "
                   f"{d['fall']:.0%} of viewers left there.")
    return out, {"still_watching_at_30s": hook, "drops": worst}
