"""The channel's theme: a few seconds of music under "The lantern is lit." in every episode.

Made once (`python -m production theme`, or the workflow's override theme=make), listened to, and
the chosen one is kept in the repository at `stills.theme_file`; from then on every episode opens
with the same sound, the way a viewer knows a show before the first word. Lyria on Vertex AI
makes the candidates (about $0.06 each, from the Google credits).
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from production.media import MediaError

from ..score import NOT_MUSIC
from ..speech import decode

log = logging.getLogger(__name__)

PROMPT = ("A short, haunting signature theme for a horror storytelling channel: three slow notes on an old music "
          "box over a low, breathing cello drone, one distant bell, a long dark reverb, very quiet and intimate, "
          "like a lantern being lit in a cold room. Instrumental, slow, minor key.")


def trim(samples: np.ndarray, rate: int, seconds: float = 7.0, fade: float = 1.5) -> np.ndarray:
    """The theme's first `seconds`, faded out at the end and in at the very start (no click)."""
    out = samples[: int(seconds * rate)].astype(np.float32).copy()
    k, start = min(int(fade * rate), len(out)), min(int(0.02 * rate), len(out))
    ramp_out, ramp_in = np.linspace(1, 0, k, dtype=np.float32), np.linspace(0, 1, start, dtype=np.float32)
    if out.ndim == 2:
        ramp_out, ramp_in = ramp_out[:, None], ramp_in[:, None]
    out[len(out) - k:] *= ramp_out
    out[:start] *= ramp_in
    peak = float(np.max(np.abs(out))) or 1.0
    return out * (0.8 / peak)


def make(stills: dict[str, Any], client: Any, out: Path, candidates: int = 3,
         model: str = "vertex:lyria-002") -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    made, failed = [], []
    for k in range(candidates):
        try:
            got = client.run(model, {"prompt": PROMPT, "negative_prompt": NOT_MUSIC, "seed": 1000 + k})
            data = base64.b64decode(str(got["audio"]["url"]).split(",", 1)[1])
        except (MediaError, KeyError, IndexError, ValueError) as exc:
            failed.append(str(exc)[:300])
            continue
        raw = out / f"raw_{k + 1}.wav"
        raw.write_bytes(data)
        theme = trim(decode(raw, 48_000, channels=2), 48_000)
        path = out / f"theme_{k + 1}.wav"
        soundfile.write(path, theme, 48_000)
        raw.unlink(missing_ok=True)
        made.append(path.name)
    return {"candidates": made, "failed": failed, "keep": f"listen, then commit the one you like as "
                                                         f"{stills.get('theme_file', 'production/assets/theme/lantern.wav')}"}
