"""The storyboard: the narration timeline cut into beats, each staged as one still picture.

A beat covers one or more spoken chunks of one scene and lasts 5-11 s, so a picture holds
long enough to be looked at but changes before it goes stale. An open model (the story
factory's planner role) stages each beat like a storyboard artist: the framing, exactly
what the picture shows, who is in it, one sound, whether it is a scare and how much it
matters, and whether it is seen through a security camera, a phone or an old photo when
the story itself looks through one. The host's intro and outro show The Storyteller. If the model fails, the scene's
own plan fills the beat in.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from story_factory.llm import LLMError

from .. import shots as shotlist
from ..shots import FRAMINGS, Shot
from .board import Board

log = logging.getLogger(__name__)

HOST = "The Storyteller"
ATMOSPHERES = ("rain", "fog", "flicker")
VIEWS = ("film", "cctv", "phone", "polaroid")
SYSTEM = ("You are the storyboard artist of Hollow Lines, a horror storytelling channel told in still "
          "pictures, like frames of a dark horror film, over a narrator's voice. You stage every beat for dread and clarity: one "
          "idea per picture, a strong composition, faces and body language that read at phone size, darkness "
          "used as space, the threat mostly unseen. Answer with JSON only.")
SCHEMA = {
    "type": "object", "required": ["beats"],
    "properties": {"beats": {"type": "array", "items": {
        "type": "object", "required": ["index", "line", "framing", "picture", "characters", "sound", "scare",
                                       "importance"],
        "properties": {
            "index": {"type": "integer"},
            "line": {"type": "string"},
            "framing": {"type": "string", "enum": list(FRAMINGS)},
            "picture": {"type": "string"},
            "characters": {"type": "array", "items": {"type": "string"}},
            "sound": {"type": "string"},
            "scare": {"type": "boolean"},
            "importance": {"type": "integer"},
            "view": {"type": "string", "enum": list(VIEWS)},
            "atmosphere": {"type": "array", "items": {"type": "string", "enum": list(ATMOSPHERES)}},
            "motion": {"type": "string"},
            "tension": {"type": "integer"},
            "threat": {"type": "integer"},
        }}}},
}


def prompt(beats: list[Shot], scenes: list[dict[str, Any]], cast: list[dict[str, Any]], red: str,
           host_beats: set[int]) -> str:
    cast_text = "; ".join(f"{c.get('name')}: {c.get('design') or c.get('look', '')}" for c in cast) or "none"
    by_scene: dict[int, list[Shot]] = {}
    for b in beats:
        by_scene.setdefault(b.scene, []).append(b)
    blocks = []
    for k, group in by_scene.items():
        sc = scenes[k] if k < len(scenes) else {}
        lines = "\n".join(f"  beat {b.index} ({b.seconds:.1f} s){' [HOST]' if b.index in host_beats else ''}: "
                          f"\"{b.text}\"" for b in group)
        blocks.append(f"SCENE {k} - location: {sc.get('location', '')}; characters: "
                      f"{', '.join(map(str, sc.get('characters') or []))}; mood: {sc.get('mood', '')}; "
                      f"key moment: {bool(sc.get('key_moment'))}\n  plan: {sc.get('visual', '')}\n{lines}")
    return (f"Cast (fixed designs): {cast_text}\nThe story's red accent: {red or 'none'}\n\n" + "\n\n".join(blocks) +
            "\n\nStage every beat listed above as ONE still picture, one entry per beat, in order:\n"
            "- index: the beat's number; line: the beat's narrated words, copied exactly (this is how your "
            "picture is matched to its words: a picture belongs to its own line, never the next one)\n"
            "- framing: vary it within a scene (establish wide, then medium, close, insert); never the same "
            "framing three times in a row\n"
            "- picture: ONE clear moment a painter gets right first time, in at most 40 words: who is where, "
            "doing what, with what feeling on their face, the one or two telling objects, the light source, and "
            "the red accent if it is in frame. Show exactly what THIS beat's words describe (the same action, "
            "pose and place: 'standing behind' is standing, not leaning), nothing from a later line. Only what "
            "the line needs: no exact positions of hands or props, no small costume details (the cast's sheets "
            "fix those), no states a camera cannot show at once (a screen both dark and glowing), nobody the "
            "line does not involve. Keep the threat half-seen: a silhouette, a shape in a doorway, eyes in the "
            "dark, placed off-centre. "
            f"Beats marked [HOST] show {HOST}, the channel's host, in his place between the worlds, speaking "
            "to us; describe him with the moment's mood\n"
            "- characters: cast names in the picture (exact spelling), [] if none\n"
            "- motion: if this picture were filmed, the one small movement in it, in at most 12 words "
            "(\"her fingers tighten on the receiver\", \"the hooded shape leans closer\"); the camera holds still\n"
            "- sound: one short sound effect for this beat (\"a door creaks open slowly\"), or \"\"\n"
            "- scare: true only for the two or three beats of the whole story where something frightening is "
            "revealed (a scare that comes often stops scaring)\n"
            "- importance: 1 (connective) to 5 (the moment the story turns on)\n"
            "- tension: 1 (calm, ordinary life) to 5 (terror): how tense this moment feels. It sets how long the "
            "picture holds, how the camera moves, how loud the music is and how the voice reads. Most beats are "
            "2-3, the build 3-4, only the peaks 5; let it breathe (a calm beat after a peak)\n"
            "- threat: how much of the threat THIS picture shows: 0 not in it, 1 only heard or implied (a sound's "
            "source, a door ajar), 2 a shape in the dark, 3 partly seen (a hand, an outline, eyes at the edge of "
            "the light), 4 fully seen. It rises across the story: 0-2 in the first half, fully seen (4) at most "
            "once, at the climax, or never. What the viewer imagines is scarier than what a picture shows\n"
            "- view: \"film\" for almost every beat; \"cctv\", \"phone\" or \"polaroid\" only when the words "
            "say we are looking at a security-camera feed, a phone photo or video, or an old photograph, and "
            "then the picture shows what that camera sees. Never for [HOST] beats\n"
            "- atmosphere: what visibly moves in the air of this picture, for the edit to animate: \"rain\" only "
            "when rain falls between the camera and the scene (outdoors, never over an interior or behind "
            "glass), \"fog\" for mist or fog in the air around the camera (outdoors, not fog seen through a window), \"flicker\" when the picture's light is a failing lamp, "
            "a guttering candle or a stuttering screen; [] for most beats. Keep it consistent within a scene\n"
            "No text, signs or captions to read in any picture.")


class StoryboardAgent:
    name = "storyboard"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        timeline, spans, scenes = d["timeline"], d["spans"], d["scenes"]
        low, high = float(cfg.get("beat_min_seconds", 5.0)), float(cfg.get("beat_max_seconds", 11.0))
        # The rhythm follows the tension: calm scenes hold their pictures longer, terrifying ones cut faster.
        limits = shotlist.scene_limits(scenes, low, high) if cfg.get("rhythm", True) else {}
        beats = shotlist.cut(timeline.chunks, shotlist.scene_index(timeline.chunks, spans), timeline.duration,
                             low, high, limits)
        host_beats = self._host_beats(beats, timeline, d)
        for b in beats:
            shotlist.fallback(b, scenes[b.scene] if b.scene < len(scenes) else {}, beats)
            if b.index in host_beats:
                b.characters = [HOST]
                b.tension, b.threat, b.scare = 2, 0, False
                b.picture = (f"{HOST} sits by his lantern's red flame in the dark, looking at us as he speaks: "
                             f"\"{b.text[:160]}\"")
        names = [str(c.get("name")) for c in d["cast"] if c.get("name")] + [HOST]
        directed = self._direct(board, beats, scenes, names, host_beats)
        left = [b for b in beats if not b.directed]
        if left and directed:  # the model skipped some: ask again for just those
            directed += self._direct(board, left, scenes, names, host_beats)
        checked = check(board, beats, scenes, host_beats)
        allowed = limit_scares(beats, timeline.duration / 60.0, float(cfg.get("scares_per_minute", 0.35)),
                               int(cfg.get("max_scares", 3)))
        climb = ladder(beats, host_beats, int(cfg.get("max_full_reveals", 1)))
        kept_safe = keep_children_safe(beats, d.get("cast") or [], host_beats)
        d.update(beats=beats, host_beats=host_beats)
        return {"beats": len(beats), "directed_by_model": directed, "host_beats": len(host_beats),
                "children_kept_safe": kept_safe,
                "scares": sum(1 for b in beats if b.scare), "scares_allowed": allowed, "check": checked,
                "reveal_ladder": climb, "tension": [b.tension for b in beats],
                "average_seconds": round(timeline.duration / max(len(beats), 1), 1)}

    def _host_beats(self, beats: list[Shot], timeline: Any, d: dict[str, Any]) -> set[int]:
        frames = d.get("frames") or []  # the intro and outro, whoever tells the story
        if not frames:
            return set()
        times = [(timeline.time_at(a), timeline.time_at(b)) for a, b in frames]
        out = set()
        for b in beats:
            mid = (b.t0 + b.t1) / 2
            if any(t0 - 0.3 <= mid <= t1 + 0.3 for t0, t1 in times):
                out.add(b.index)
        return out

    def _direct(self, board: Board, beats: list[Shot], scenes: list[dict[str, Any]], names: list[str],
                host_beats: set[int], batch_scenes: int = 5) -> int:
        if board.router is None:
            return 0
        red = str((board.data.get("bible") or {}).get("red_accent", ""))
        got = 0
        scene_ids = sorted({b.scene for b in beats})
        for start in range(0, len(scene_ids), batch_scenes):
            wanted = set(scene_ids[start:start + batch_scenes])
            group = [b for b in beats if b.scene in wanted]
            try:
                result = board.router.one("planner").generate_json(
                    "storyboard", SYSTEM, prompt(group, scenes, board.data["cast"], red, host_beats), SCHEMA)
            except (LLMError, ValueError, TypeError) as exc:
                log.warning("Storyboard for scenes %s from the scene plan: %s", sorted(wanted), exc)
                continue
            for item, beat in match(result.get("beats", []) if isinstance(result, dict) else [], group):
                if beat.directed or not str(item.get("picture", "")).strip():
                    continue
                beat.framing = item.get("framing") if item.get("framing") in FRAMINGS else beat.framing
                beat.picture = str(item["picture"]).strip()[:700]
                chars = [shotlist._canonical(str(n), names) for n in item.get("characters") or [] if str(n).strip()]
                beat.characters = ([HOST] if beat.index in host_beats else chars)[:3]
                beat.sound = str(item.get("sound", "")).strip()[:120]
                beat.motion = str(item.get("motion", "")).strip()[:160]
                beat.scare = bool(item.get("scare")) and beat.index not in host_beats
                try:
                    beat.importance = min(max(int(item.get("importance", 2)), 1), 5)
                except (TypeError, ValueError):
                    pass
                beat.tension = _level(item.get("tension"), 1, 5, beat.tension)
                beat.threat = _level(item.get("threat"), 0, 4, beat.threat) if beat.index not in host_beats else 0
                view = str(item.get("view") or "film")
                beat.view = view if view in VIEWS and beat.index not in host_beats else "film"
                air = item.get("atmosphere")
                beat.atmosphere = ([str(a) for a in air if str(a) in ATMOSPHERES][:2] if isinstance(air, list)
                                   else shotlist.atmosphere_in(beat.picture))
                place = str(scenes[beat.scene].get("location", "")) if beat.scene < len(scenes) else ""
                if beat.view != "film" or not shotlist.outside(beat.picture):
                    beat.atmosphere = shotlist.outdoor_air(beat.atmosphere, beat.picture, place,
                                                           "" if beat.view == "film" else "screen")
                beat.directed = True
                got += 1
        return got


def _level(value: Any, low: int, high: int, default: int) -> int:
    try:
        return min(max(int(value), low), high)
    except (TypeError, ValueError):
        return default


def limit_scares(beats: list[Shot], minutes: float, per_minute: float = 0.35, most: int = 3) -> int:
    """At most `per_minute` scares a minute (never fewer than one, never more than `most`): the most
    important keep their stinger and the strongest painter; the rest stay tense beats."""
    allowed = max(1, min(most, round(minutes * per_minute)))
    scares = sorted((b for b in beats if b.scare), key=lambda b: (-b.importance, -b.tension, b.index))
    for b in scares[allowed:]:
        b.scare = False
    return allowed


def keep_children_safe(beats: list[Shot], cast: list[dict[str, Any]], host_beats: set[int]) -> list[int]:
    """A child is never in a frightening frame: in a picture with a child, the threat leaves the frame
    (threat 0, no scare) and the prompt says the child is safe. The fear stays in the words, the sound
    and the pictures of the adults. Photorealistic children in peril are where Google's rules are
    strictest; returns the beats changed."""
    from .art import child_age
    from .wording import CHILD_SAFE, mentions_child

    children = {str(c.get("name")) for c in cast
                if isinstance(c, dict) and c.get("name") and child_age(str(c.get("look") or c.get("design") or ""))}
    changed = []
    for b in beats:
        if b.index in host_beats:
            continue
        has_child = any(n in children for n in b.characters) or mentions_child(b.picture)
        frightening = b.threat > 0 or b.scare or b.tension >= 4
        if has_child and frightening:
            b.threat, b.scare = 0, False
            if CHILD_SAFE.strip() not in b.picture:
                b.picture = b.picture.rstrip() + CHILD_SAFE
            changed.append(b.index)
    return changed


def ladder(beats: list[Shot], host_beats: set[int], max_full: int = 1) -> dict[str, Any]:
    """The reveal ladder: how much of the threat may be shown rises toward the climax (the most
    important scare): a shape in the dark at most before halfway there, partly seen until the climax,
    fully seen at most `max_full` times from the climax on. Image models show a monster clearly
    unless told otherwise, and a threat shown early stops frightening."""
    story = [b for b in beats if b.index not in host_beats]
    if not story:
        return {}
    scares = [b for b in story if b.scare]
    seen = [b for b in story if b.threat >= 3]  # pictures staged to show the threat itself
    # The climax is where the threat is shown, not only where the scare flag sits: run 118 flagged "I looked
    # at the camera" and capped the real reveal a few seconds later ("leaning over my shoulder to write")
    # at "mostly in darkness", so the checker refused every picture of the figure and kept a hand.
    climax = max(seen + scares or story[len(story) // 2:],
                 key=lambda b: (b.threat >= 3, b.scare, b.importance, b.tension, b.index))
    at = next(k for k, b in enumerate(story) if b.index == climax.index)
    if climax.threat == 3:
        climax.threat = 4  # the reveal: seen, in low light and partly in shadow, never refused as "too clear"
    lowered = full = 0
    for k, b in enumerate(story):
        cap = 2 if k < at / 2 else 3 if k < at else 4
        if b.threat == 4 and cap == 4:
            full += 1
            if full > max_full:
                cap = 3
        if b.threat > cap:
            b.threat = cap
            lowered += 1
    return {"climax": climax.index, "lowered": lowered}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def match(items: list[Any], group: list[Shot]) -> list[tuple[dict[str, Any], Shot]]:
    """Each staged picture with the beat whose words it quotes. Run 94: the model numbered its
    pictures one beat late, so matching by number put every picture under the line before it."""
    out = []
    taken: set[int] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        quoted = _words(str(item.get("line", "")))
        best, score = None, 0.0
        for b in group:
            if b.index in taken:
                continue
            words = _words(b.text)
            if not words or not quoted:
                continue
            same = sum(1 for w in quoted if w in words) / max(len(quoted), len(words))
            if same > score:
                best, score = b, same
        if best is None or score < 0.6:  # no clear quote: fall back to the number
            best = next((b for b in group if b.index == item.get("index") and b.index not in taken), None)
            if best is None:
                continue
        taken.add(best.index)
        out.append((item, best))
    return out


CHECK_SYSTEM = ("You are the script supervisor of Hollow Lines, a horror channel told in still pictures over a "
                "narrator. You make sure every picture shows exactly what is being said while it is on screen. "
                "Answer with JSON only.")
CHECK_SCHEMA = {"type": "object", "required": ["fix"], "properties": {"fix": {"type": "array", "items": {
    "type": "object", "required": ["index", "problem", "picture"],
    "properties": {"index": {"type": "integer"}, "problem": {"type": "string"}, "picture": {"type": "string"},
                   "characters": {"type": "array", "items": {"type": "string"}}}}}}}


def check(board: Board, beats: list[Shot], scenes: list[dict[str, Any]], host_beats: set[int]) -> dict[str, Any]:
    """The storyboard gate, before any picture is paid for: every picture description is held
    against its narrated line, the lines before it and the cast's fixed looks; the ones that
    don't match are rewritten. A description costs nothing to fix; a picture does."""
    if board.router is None or not beats:
        return {"checked": False}
    cast = "; ".join(f"{c.get('name')}: {c.get('look', '')}" for c in board.data.get("cast", []) if c.get("name"))
    rows = "\n".join(f"{b.index}. LINE: \"{b.text}\"\n   PICTURE: {b.picture}\n   PEOPLE: "
                     f"{', '.join(b.characters) or 'none'}" for b in beats if b.index not in host_beats)
    text = (f"Cast (fixed looks): {cast or 'none'}\n\n{rows}\n\nEach PICTURE is on screen while its LINE is "
            "narrated. Find every picture that does not show what its own line says: a different action or pose "
            "(the line says standing, the picture says leaning or only a hand), a different place, something from "
            "an earlier or later line instead of this one, a person the line does not involve, or a person's fixed "
            "look broken (a face the cast notes hide must stay hidden). For each, give the problem and a new "
            "PICTURE in the same style (one clear moment in at most 40 words: who is where doing what, the "
            "feeling, the light, the threat half-seen; no exact positions of hands or props) that shows exactly "
            "this line, and its PEOPLE. Leave pictures that match out. "
            '{"fix": [{"index": N, "problem": "...", "picture": "...", "characters": [...]}]}')
    try:
        result = board.router.one("planner").generate_json("storyboard_check", CHECK_SYSTEM, text, CHECK_SCHEMA)
    except (LLMError, ValueError, TypeError) as exc:
        log.warning("Storyboard check unavailable: %s", exc)
        return {"checked": False}
    names = [str(c.get("name")) for c in board.data.get("cast", []) if c.get("name")]
    by_index = {b.index: b for b in beats if b.index not in host_beats}
    fixed = []
    for f in result.get("fix", []) if isinstance(result, dict) else []:
        beat = by_index.get(f.get("index")) if isinstance(f, dict) else None
        picture = str((f or {}).get("picture", "")).strip() if isinstance(f, dict) else ""
        if beat is None or len(picture) < 20:
            continue
        beat.picture = picture[:700]
        if not shotlist.outside(picture):
            beat.atmosphere = shotlist.outdoor_air(beat.atmosphere, picture)  # the new picture may be indoors
        if isinstance(f.get("characters"), list):
            beat.characters = [shotlist._canonical(str(n), names + [HOST]) for n in f["characters"] if str(n).strip()][:3]
        fixed.append({"beat": beat.index, "problem": str(f.get("problem", ""))[:200]})
    return {"checked": True, "fixed": fixed}
