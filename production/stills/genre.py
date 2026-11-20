"""Another genre than horror for one episode (a script's `stills.genre`).

The agents' instructions were written for Hollow Lines' horror: the storyboard stages "for dread",
the sound director avoids "a cheerful or melodic track under a horror story", the checkers judge
"a horror film". An episode of another kind sets `genre`, and every instruction sent to the planner
and to the models on Google is re-toned before it leaves: the horror words are said in the genre's
words, and a note says plainly what this episode is and is not. ElevenLabs' spoken text is never
touched (only instructions and descriptions are).

    "genre": {"adjective": "romantic",
              "note": "This episode is a sensual romantic drama, not horror: ...",
              "words": {"dread": "longing"}}    # optional: more replacements
"""

from __future__ import annotations

import re
from typing import Any

# horror's words -> neutral or the genre's ({adj} is the genre's adjective). Longer phrases first.
BASE = [
    (r"\bslow-burn horror film\b", "slow-burn {adj} film"),
    (r"\bdark horror film\b", "{adj} film"),
    (r"\bhorror storytelling channel\b", "cinematic storytelling channel"),
    (r"\bhorror channel\b", "cinematic storytelling channel"),
    (r"\bhorror film score\b", "{adj} film score"),
    (r"\bhorror (film|story|stories|picture|pictures|narrator|audio stories|score|image|images|viewers?)\b", r"{adj} \1"),
    (r"\bhorror\b", "{adj}"),
    (r"\bjump scares?\b", "emotional turn"),
    (r"\bscares\b", "turning points"),
    (r"\bscare\b", "turning point"),
    (r"\bscary\b", "intense"),
    (r"\bterror\b", "intensity"),
    (r"\bdread\b", "anticipation"),
    (r"\bfrightening\b", "intense"),
]


def rules(genre: dict[str, Any]) -> list[tuple[re.Pattern[str], str]]:
    adj = str(genre.get("adjective") or "dramatic")
    pairs = [(p, r.replace("{adj}", adj)) for p, r in BASE]
    pairs = [(rf"\b{re.escape(k)}\b", v) for k, v in (genre.get("words") or {}).items()] + pairs
    return [(re.compile(p, re.I), r) for p, r in pairs]


def retone(text: str, genre: dict[str, Any], compiled: list | None = None) -> str:
    if not text:
        return text
    for pattern, replacement in compiled or rules(genre):
        text = pattern.sub(replacement, text)
    return text


def system_note(genre: dict[str, Any]) -> str:
    note = str(genre.get("note") or "").strip()
    return f"\n\nTHIS EPISODE: {note}" if note else ""


class GenreRouter:
    """The planner, with every system prompt and prompt re-toned for the episode's genre."""

    def __init__(self, inner: Any, genre: dict[str, Any]):
        self.inner, self.genre, self._rules = inner, genre, rules(genre)

    def one(self, role: str) -> Any:
        return _GenrePlanner(self.inner.one(role), self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class _GenrePlanner:
    def __init__(self, inner: Any, router: GenreRouter):
        self.inner, self.router = inner, router

    def _system(self, system: str) -> str:
        return retone(system, self.router.genre, self.router._rules) + system_note(self.router.genre)

    def generate_json(self, task: str, system: str, prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
        return self.inner.generate_json(task, self._system(system), retone(prompt, self.router.genre, self.router._rules),
                                        schema)

    def generate(self, task: str, system: str, prompt: str, *args: Any, **kwargs: Any) -> Any:
        return self.inner.generate(task, self._system(system), retone(prompt, self.router.genre, self.router._rules),
                                   *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


class GenreClient:
    """The media client, with the instructions and descriptions sent to Google re-toned (pictures,
    checks, the screening). ElevenLabs requests pass untouched: their text is the story itself."""

    def __init__(self, inner: Any, genre: dict[str, Any]):
        self.inner, self.genre, self._rules = inner, genre, rules(genre)

    def run(self, endpoint: str, args: dict[str, Any]) -> dict[str, Any]:
        if "elevenlabs" not in str(endpoint):
            args = dict(args)
            if args.get("prompt"):
                args["prompt"] = retone(str(args["prompt"]), self.genre, self._rules)
            if args.get("system_prompt"):
                args["system_prompt"] = (retone(str(args["system_prompt"]), self.genre, self._rules)
                                         + system_note(self.genre))
        return self.inner.run(endpoint, args)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)
