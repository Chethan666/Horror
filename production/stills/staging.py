"""The cinematographer: every picture staged in the set as it was actually painted.

The storyboard is written before any set exists, from words, by a model that cannot see. Then
the art director paints each place's location sheet, and the illustrator paints every picture
from that sheet. When a description asks for something the painted set does not have (a window
on the wrong wall, a seat that is not there, an angle the room cannot give), the painter must
choose between the words and the sheet, and either way the picture is wrong.

So after the sheets are made, a vision model looks at each location sheet with the pictures
set there and rewrites only the descriptions that do not fit the set: the same moment, the
same people and action and framing, placed where the set allows. One look per place (the
stronger checker when one is configured, else the everyday one): a few cents a video, before
any picture is paid for.
"""

from __future__ import annotations

import logging
from typing import Any

from .board import Board
from .checks import ask_vision
from .storyboard import HOST

log = logging.getLogger(__name__)

SYSTEM = ("You are the cinematographer of Hollow Lines, a horror channel told in still pictures. You place every "
          "shot inside the set as it was built. Answer with JSON only.")


class StagingAgent:
    name = "staging"
    critical = False  # without it, every picture keeps its storyboard description

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        if not cfg.get("staging", True) or board.studio is None:
            return {"staged": False}
        host = set(d.get("host_beats") or ())
        rewritten: list[dict[str, Any]] = []
        looked = 0
        for place in d.get("sets") or []:
            sheet = place.get("sheet")
            group = [b for b in d.get("beats", []) if b.scene in place.get("scenes", []) and b.index not in host
                     and HOST not in b.characters]
            if sheet is None or not group:
                continue
            rows = "\n".join(f"{b.index}. ({b.framing}) LINE: \"{str(b.text)[:160]}\"\n   PICTURE: {b.picture}"
                             for b in group)
            prompt = (f"The image is the location sheet of {place['name']}, the set where these pictures are shot "
                      f"(it may show two views side by side): {str(place.get('design', ''))[:500]}\n\n{rows}\n\n"
                      "For each PICTURE whose description does not fit this set as painted (it asks for a window, "
                      "door, seat, light or object that is somewhere else or missing, or an angle this set cannot "
                      "give), rewrite it so it fits: the same moment, the same people and action, the same framing, "
                      "at most 40 words, saying where in the set the camera looks. Keep the threat as hidden as "
                      "the original says. Leave every picture that already fits out. Reply exactly as JSON: "
                      '{"pictures": [{"index": N, "picture": "...", "why": "short reason"}]}')
            verdict = ask_vision(board.studio, prompt, sheet, "staging", SYSTEM, width=1536, role="vision_pro")
            looked += 1
            by_index = {b.index: b for b in group}
            for item in (verdict or {}).get("pictures") or []:
                if not isinstance(item, dict):
                    continue
                try:
                    beat = by_index.get(int(item.get("index")))
                except (TypeError, ValueError):
                    continue
                text = str(item.get("picture", "")).strip()
                if beat is None or len(text) < 20:
                    continue
                beat.picture = text[:700]
                rewritten.append({"beat": beat.index, "place": place["name"], "why": str(item.get("why", ""))[:160]})
        return {"staged": True, "places": looked, "rewritten": rewritten}
