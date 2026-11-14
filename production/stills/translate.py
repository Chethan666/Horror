"""Other languages: the title, the description and the captions, translated for YouTube.

The channel is told in English for a global audience. YouTube shows a viewer the title and
description in their own language when the video carries a translation of them (localizations),
and offers captions in each language uploaded. The checker model on Vertex translates them (a
fraction of a cent per language); the captions keep their own timing, only the words change. The
publish step uploads them with the video. Without Google's model nothing is translated.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from production.media import MediaError

from .. import timing
from ..paid import OverBudget

log = logging.getLogger(__name__)

NAMES = {"es": "Spanish", "pt": "Portuguese (Brazil)", "hi": "Hindi", "id": "Indonesian", "fr": "French",
         "de": "German", "it": "Italian", "ja": "Japanese", "ko": "Korean", "ar": "Arabic", "tr": "Turkish",
         "ru": "Russian", "pl": "Polish", "nl": "Dutch", "vi": "Vietnamese", "th": "Thai", "tl": "Filipino"}
SYSTEM = ("You translate a horror channel's YouTube texts. Keep the mood, the suspense and every name; translate "
          "naturally, the way a native horror narrator would say it, never word for word. Answer with JSON only.")


def _ask(studio: Any, prompt: str, units: float = 1.0) -> dict[str, Any] | None:
    model = str(studio.models.get("vision", "")) if studio is not None else ""
    if not model.startswith("vertex:"):
        return None
    try:
        usd = studio.budget.reserve(model, units)
    except OverBudget:
        return None
    try:
        result = studio.fal.run(model, {"prompt": prompt, "system_prompt": SYSTEM, "max_tokens": 8192})
    except MediaError as exc:
        studio.budget.settle(usd, "translate", model, ok=False)
        log.warning("Translation failed: %s", str(exc)[:200])
        return None
    studio.budget.settle(usd, "translate", model, ok=True, units=units)
    match = re.search(r"\{.*\}", str(result.get("output") or ""), re.S)
    try:
        got = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    return got if isinstance(got, dict) else None


def localize(studio: Any, title: str, description: str, cues: list[Any], languages: list[str],
             out: Any) -> dict[str, Any]:
    """{"localizations": {lang: {"title", "description"}}, "captions": {lang: file name}}; caption files
    are written to `out` as captions_<lang>.srt."""
    localizations: dict[str, dict[str, str]] = {}
    captions: dict[str, str] = {}
    lines = [str(c.text) for c in cues]
    for lang in languages:
        name = NAMES.get(lang, lang)
        meta = _ask(studio, f"Translate into {name}. The title stays under 100 characters and keeps its hook; the "
                            "description keeps its line breaks, time stamps and links exactly.\n"
                            + json.dumps({"title": title, "description": description}, ensure_ascii=False)
                            + '\nReply exactly as JSON: {"title": "...", "description": "..."}')
        if meta and str(meta.get("title", "")).strip():
            localizations[lang] = {"title": str(meta["title"]).strip()[:100],
                                   "description": str(meta.get("description", "")).strip()[:5000]}
        if not lines:
            continue
        said = _ask(studio, f"Translate each caption line into {name}, one line for one line, in order; keep them "
                            "short enough to read at a glance.\n" + json.dumps({"lines": lines}, ensure_ascii=False)
                            + '\nReply exactly as JSON: {"lines": ["...", ...]} with exactly '
                            f"{len(lines)} lines.", units=max(1.0, len(lines) / 40))
        got = said.get("lines") if said else None
        if isinstance(got, list) and len(got) == len(lines):
            path = out / f"captions_{lang}.srt"
            path.write_text(timing.srt([timing.Cue(c.start, c.end, str(t)) for c, t in zip(cues, got)]))
            captions[lang] = path.name
    return {"localizations": localizations, "captions": captions}
