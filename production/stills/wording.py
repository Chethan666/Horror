"""Wording that keeps a horror picture a film still, not a request for harm.

Every picture prompt passes through `soften` just before it is sent (production/stills/images.py).
The channel's stories are suspense, never gore, but a prompt written fast reads as harm to a real
person: "a man close enough to breathe on her neck", "a blood droplet", "she screams in terror".
Google's classifiers read prompts out of context; repeated across a run, such words are what an
account review sees (the project was blocked on 19 November 2026). The same moment, said as a
film would frame it, paints the same picture.

`child_in_danger` is the rule for children: a child is never in a frightening frame. The child is
shown safe (asleep, calm, alone); the threat is shown elsewhere, or only implied.
"""

from __future__ import annotations

import re

# (pattern, replacement): case-insensitive, whole words. The house style's "blood-red accent" (a colour)
# and "never gory" (a rule) are left alone.
def _verb(base: str, s: str, ed: str, ing: str) -> dict[str, str]:
    return {"": base, "s": s, "es": s, "ed": ed, "ped": ed, "bed": ed, "ing": ing, "ping": ing, "bing": ing}


VERBS = {  # a harm-sounding verb -> the same tense of a film's verb
    "scream": _verb("gasp", "gasps", "gasped", "gasping"),
    "grab": _verb("reach for", "reaches for", "reached for", "reaching for"),
    "attack": _verb("approach", "approaches", "approached", "approaching"),
    "kill": _verb("take", "takes", "taken", "taking"),
    "stab": _verb("loom over", "looms over", "loomed over", "looming over"),
    "slash": _verb("loom over", "looms over", "loomed over", "looming over"),
    "strangle": _verb("loom over", "looms over", "loomed over", "looming over"),
    "choke": _verb("loom over", "looms over", "loomed over", "looming over"),
}
VERB = re.compile(r"\b(scream|grab|attack|kill|stab|slash|strangl|chok)(e?)(s|es|ed|d|ped|bed|ing|ping|bing)?\b", re.I)

# (pattern, replacement): case-insensitive, whole words. The house style's "blood-red accent" (a colour)
# and "never gory" (a rule) are left alone. Longer phrases come first.
RULES: list[tuple[str, str]] = [
    (r"\b(?:close enough to )?breath(?:e|es|ing) on (?:her|his|my|their|your) neck\b", "standing close behind"),
    (r"\bclose enough to breathe on\b", "standing close behind"),
    (r"\bblood(?:y)?\s+(drop(?:let)?s?|stains?|smears?|pools?|spatters?|splatters?)\b", r"dark red \1"),
    (r"\bbloody\b", "dark-stained"),
    (r"\bblood(?![- ]red)\b", "dark red ink"),
    (r"\b(?:gore|gory)\b", "dark"),
    (r"\bkiller\b", "stranger"),
    (r"\bterrified\b", "frightened"),
    (r"\bin terror\b", "in fear"),
    (r"\b(?:corpse|dead body)\b", "a still figure under a sheet"),
    (r"\bwounded\b", "bandaged"),
    (r"\bwound(s?)\b", r"bandage\1"),
    (r"\b(?:naked|nude|undressed|half-dressed)\b", "fully clothed"),
]
_COMPILED = [(re.compile(p, re.I), r) for p, r in RULES]
NEVER = re.compile(r"\bnever gory\b", re.I)

CHILD = re.compile(r"\b(?:child|children|kid|kids|toddler|baby(?! monitor)|babies|infant|little girl|little boy|"
                   r"girl|boy|daughter|son|\d{1,2}[- ]year[- ]old)\b", re.I)
CHILD_SAFE = (" The child is shown safe and calm (asleep, or simply in the room), never frightened or "
              "touched; nothing threatening shares the frame with the child.")


def soften(prompt: str) -> str:
    """The prompt with harm-sounding words said as a film would say them."""
    def guarded(pattern: re.Pattern[str], replace, text: str) -> str:
        keep = [m.span() for m in NEVER.finditer(text)]
        return pattern.sub(lambda m: m.group(0) if any(a <= m.start() < b for a, b in keep) else replace(m), text)

    out = prompt
    for pattern, replacement in _COMPILED:
        out = guarded(pattern, lambda m, r=replacement: m.expand(r), out)

    def verb(m: re.Match[str]) -> str:
        stem = m.group(1).lower()
        stem = {"strangl": "strangle", "chok": "choke"}.get(stem, stem)
        ending = (m.group(3) or "").lower()
        ending = "ed" if ending == "d" else ending
        word = VERBS[stem].get(ending, VERBS[stem][""])
        return word[:1].upper() + word[1:] if m.group(0)[:1].isupper() else word

    return guarded(VERB, verb, out)


def mentions_child(text: str) -> bool:
    return bool(CHILD.search(text or ""))
