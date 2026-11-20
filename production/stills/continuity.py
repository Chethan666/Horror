"""The continuity agent: what the story has changed, held in every picture after it.

Each picture is painted from its own description, so a picture does not know that, three beats
earlier, the hallway light went out, the door was left open or her hand was cut. Run 104 showed
a night-light glowing in the very pictures whose narration said it was off; a word match
(art.lights_off) now catches lights, and this agent catches the rest: the planner reads the
whole storyboard once and lists every visible change, from the beat it happens to the last beat
it holds. The illustrator puts the facts that hold for a beat into its prompt, and the critic
fails a picture that contradicts them. One planner call per video (free on the planner's
allowance); nothing is painted here.
"""

from __future__ import annotations

import logging
from typing import Any

from story_factory.llm import LLMError

from .board import Board

log = logging.getLogger(__name__)

SYSTEM = ("You are the script supervisor of Hollow Lines, a horror channel told in still pictures. You keep "
          "continuity: what the story changes stays changed in every later picture. Answer with JSON only.")
SCHEMA = {"type": "object", "required": ["facts"], "properties": {"facts": {"type": "array", "items": {
    "type": "object", "required": ["fact", "from", "until"],
    "properties": {"fact": {"type": "string"}, "from": {"type": "integer"}, "until": {"type": "integer"}}}}}}


def prompt(rows: str, cast: str) -> str:
    return (f"Cast (their sheets fix their looks): {cast or 'none'}\n\nThe pictures in order, each with the words "
            f"narrated over it:\n{rows}\n\n"
            "List what the STORY changes that later pictures must keep showing: a light switched off or on, a door "
            "or window left open or shut, an injury or blood, wet, torn or changed clothes, something a person "
            "picks up and carries, something broken or moved, the weather starting or stopping. For each: the fact "
            "as a picture must show it (\"Mara's left hand is wrapped in a bloody cloth\"), the picture where it "
            "starts (from) and the last picture where it still holds (until). Only facts a later picture could "
            "get wrong; nothing the cast sheets already fix; at most 20. Reply exactly as JSON: "
            '{"facts": [{"fact": "...", "from": N, "until": M}]}')


class ContinuityAgent:
    name = "continuity"
    critical = False  # without it, each picture follows its own description as before

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        beats = d.get("beats") or []
        host = set(d.get("host_beats") or ())
        story = [b for b in beats if b.index not in host]
        if board.router is None or not cfg.get("continuity", True) or len(story) < 3:
            d["continuity"] = {}
            return {"facts": 0}
        rows = "\n".join(f"{b.index}. LINE: \"{str(b.text)[:180]}\"  PICTURE: {str(b.picture)[:160]}" for b in story)
        cast = "; ".join(f"{c.get('name')}: {str(c.get('look', ''))[:80]}" for c in d.get("cast", [])
                         if isinstance(c, dict) and c.get("name"))
        try:
            result = board.router.one("planner").generate_json("continuity", SYSTEM, prompt(rows, cast), SCHEMA)
        except (LLMError, ValueError, TypeError) as exc:
            log.warning("No continuity sheet: %s", exc)
            d["continuity"] = {}
            return {"facts": 0, "error": str(exc)[:200]}
        known = {b.index for b in story}
        last = max(known)
        scenes = d.get("scenes") or []
        place = {b.index: " ".join(str((scenes[b.scene] if b.scene < len(scenes) else {}).get("location", "")).lower().split())
                 for b in story}
        names = [str(c.get("name", "")).lower() for c in d.get("cast", []) if isinstance(c, dict) and c.get("name")]
        skipped = 0
        per_beat: dict[int, list[str]] = {}
        kept = []
        for item in (result.get("facts") or [])[:20] if isinstance(result, dict) else []:
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact", "")).strip().rstrip(".")[:160]
            try:
                start, end = int(item.get("from")), int(item.get("until", last))
            except (TypeError, ValueError):
                continue
            if len(fact) < 8 or start not in known:
                continue
            end = min(max(end, start), last)
            kept.append({"fact": fact, "from": start, "until": end})
            # A fact about a place stays in that place; one about a person travels with them. Run 118:
            # "Room 6's door is nailed shut with planks" reached every office picture after it, and the
            # office was painted with a boarded door in it.
            personal = any(n and n in fact.lower() for n in names)
            for i in range(start, end + 1):
                if i not in known:
                    continue
                if not personal and place.get(i) and place.get(start) and place[i] != place[start]:
                    skipped += 1
                    continue
                if len(per_beat.setdefault(i, [])) < 4 and fact not in per_beat[i]:
                    per_beat[i].append(fact)
        per_beat = {i: facts for i, facts in per_beat.items() if facts}
        d["continuity"] = per_beat
        return {"facts": len(kept), "beats_with_facts": len(per_beat), "kept_in_their_place": skipped,
                "sheet": kept[:20]}
