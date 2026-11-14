"""Sound design: half of every scare.

Layers, from the bottom: a room tone for each place (looped), a music score that follows the
story's tension (three cues: unease, dread, terror, crossfaded as the story moves between them),
a sound effect for every shot that asks for one, and a stinger on every scare. Before each scare
the music and room tone drop away, so the hit lands in silence. Everything dips under the voice.
If a sound fails, the synthesized drone and cues from `sound.py` fill in.

The mix is stereo, for the headphones many horror viewers wear: the voice in the centre, music
and room tone wide, each effect placed to one side (a whisper close to one ear, footsteps that
cross from one side to the other), stingers wide.
"""

from __future__ import annotations

import logging
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from production.media import MediaError

from . import sound
from .paid import OverBudget, Studio, digest
from .shots import Shot
from .speech import decode

log = logging.getLogger(__name__)

CUES = {
    "unease": "Dark ambient horror film score, slow and sparse: a low cello drone, distant detuned piano notes, "
              "soft eerie textures, quiet and patient, instrumental, no drums, no vocals.",
    "dread": "Tense horror film score building dread: pulsing low strings, a slow heartbeat-like pulse, rising "
             "dissonant textures, creeping and suspenseful, instrumental, no vocals.",
    "terror": "Intense horror film climax score: aggressive dissonant strings, pounding low percussion, "
              "screeching high violins, frantic and terrifying, instrumental, no vocals.",
}
STING = "A sudden horror jump-scare stinger: a sharp dissonant orchestral hit with a fast reverse swell into it."
# What an original score must never be (Lyria takes a negative prompt).
NOT_MUSIC = ("vocals, singing, voice, speech, lyrics, choir words, upbeat, happy, cheerful, pop, dance beat, "
             "drum kit, electronic dance music")
CALM, UNEASY, TENSE = 0, 1, 2


def level_of(shot: Shot, mood: str = "") -> int:
    """The music's level for a beat, from its tension (1-5, set by the storyboard): terror only at
    the peaks, dread through the build, unease for the rest."""
    t = int(getattr(shot, "tension", 0) or 0)
    if not t:
        return tension(mood)
    return TENSE if t >= 5 else UNEASY if t >= 3 else CALM


def tension(mood: str) -> int:
    mood = mood.lower()
    if any(w in mood for w in ("terror", "panic", "horror", "shock", "frantic", "violent")):
        return TENSE
    if any(w in mood for w in ("tense", "dread", "fear", "menac", "urgent", "eerie", "ominous", "uneas")):
        return UNEASY
    return CALM


class Score:
    def __init__(self, studio: Studio | None, work: Path, rate: int, parallel: int = 4):
        self.studio, self.work, self.rate, self.parallel = studio, work, rate, parallel
        self.cue_names = list(CUES)  # which score cues to generate (fewer is cheaper)
        self.min_music = 60.0  # shortest cue generated, in seconds
        # Sounds picked from the library by the sound director, by name (music_dread, room_<place>,
        # fx_007, sting): used instead of generating. `generate_missing` False never pays for sound.
        self.chosen: dict[str, Path] = {}
        self.generate_missing = True
        self.make_kinds: set[str] | None = None  # None: any kind may be made; else only these
        # The sound director's own descriptions for sounds it chose to have made (by name), and
        # the effects it chose to leave out.
        self.prompts: dict[str, str] = {}
        self.skip: set[str] = set()
        self.generated: list[dict[str, Any]] = []  # what was paid for, to be saved into the library
        # Sounds made first even when the library has one (the original score: "music_dread", ...); the
        # library's pick is then the fallback if making fails.
        self.prefer_made: set[str] = set()
        self.motif = ""  # the story's recurring sound, woven into an original score
        self.seed = 0  # the score's seed (from the story), so a rerun asks for the same music
        self.music_clips = 2  # Lyria makes about 30 s a piece: a cue is this many pieces, crossfaded
        # The channel's theme (a few seconds, made once and kept in the repository) and where it plays:
        # under "The lantern is lit.", so every episode opens with the same sound.
        self.theme: np.ndarray | None = None
        self.theme_at: list[float] = []
        self.report: dict[str, Any] = {"music": [], "ambience": 0, "effects": 0, "synthesized": []}
        work.mkdir(parents=True, exist_ok=True)

    # --- generated sounds -----------------------------------------------------------
    def _library(self, name: str, channels: int = 1) -> np.ndarray | None:
        picked = self.chosen.get(name)
        if picked is None:
            return None
        try:
            return decode(picked, self.rate, channels=channels)
        except MediaError as exc:
            log.warning("Library sound for %s unreadable: %s", name, str(exc)[:120])
            return None

    def _gen(self, role: str, args: dict[str, Any], name: str, units: float = 1.0,
             channels: int = 1) -> np.ndarray | None:
        if name not in self.prefer_made:
            found = self._library(name, channels)
            if found is not None:
                return found
        made = self._make(role, args, name, units, channels)
        if made is None and name in self.prefer_made:
            return self._library(name, channels)
        return made

    def _make(self, role: str, args: dict[str, Any], name: str, units: float = 1.0,
              channels: int = 1) -> np.ndarray | None:
        if not self.generate_missing or self.studio is None or not self.studio.models.get(role):
            return None
        if self.make_kinds is not None and role not in self.make_kinds:
            return None  # e.g. music from the library only, effects made when missing
        try:
            path = self.studio.call(role, args, "audio", self.work / f"{name}.mp3", f"sound_{role}", units=units)
            self.generated.append({"name": name, "role": role, "path": path,
                                   "text": str(args.get("prompt") or args.get("text") or ""),
                                   "loop": bool(args.get("loop")), "model": self.studio.models.get(role, "")})
            return decode(path, self.rate, channels=channels)
        except (MediaError, OverBudget) as exc:
            log.warning("Sound %s not generated: %s", name, str(exc)[:200])
            return None

    def music(self, seconds: float) -> dict[str, np.ndarray]:
        length = int(min(max(seconds, self.min_music), 150))
        model = self.studio.models.get("music", "") if self.studio is not None else ""

        def one(item: tuple[str, str]) -> tuple[str, np.ndarray | None]:
            key, prompt = item
            prompt = self.prompts.get(f"music_{key}", prompt)
            if "lyria" in model:  # Lyria (Google, on Vertex AI): about 30 s a piece, priced per piece
                text = prompt + (f" A quiet recurring motif suggests {self.motif}." if self.motif else "")
                name = f"music_{key}"
                parts = []
                for k in range(max(1, int(self.music_clips))):
                    got = self._make("music", {"prompt": text, "negative_prompt": NOT_MUSIC,
                                               "seed": (self.seed + 7919 * k + hash_key(key)) % 2_147_483_647},
                                     name if k == 0 else f"{name}_{k + 1}", units=1.0, channels=2)
                    if got is None:
                        break
                    parts.append(got)
                if parts:
                    return key, _joined(parts, self.rate)
                return key, self._library(name, channels=2)
            if "ace-step" in model:  # ACE-Step (open source): billed by the second
                args = {"prompt": prompt[:500], "lyrics": "[Instrumental]", "duration": float(length)}
                units = float(length)
            else:  # ElevenLabs: billed in whole minutes
                args = {"prompt": prompt, "music_length_ms": length * 1000, "force_instrumental": True}
                units = math.ceil(length / 60) * 60
            return key, self._gen("music", args, f"music_{key}", units=units, channels=2)
        with ThreadPoolExecutor(max_workers=3) as pool:
            got = {k: v for k, v in pool.map(one, [(k, CUES[k]) for k in self.cue_names if k in CUES])
                   if v is not None and v.size}
        self.report["music"] = sorted(got)
        return got

    def effect(self, text: str, name: str, seconds: float | None = None, loop: bool = False,
               channels: int = 1) -> np.ndarray | None:
        if name in self.skip:
            return None
        args: dict[str, Any] = {"text": self.prompts.get(name, text)}
        if seconds:
            args["duration_seconds"] = round(min(max(seconds, 0.5), 22.0), 1)
        if loop:
            args["loop"] = True
        return self._gen("sfx", args, name, units=seconds or 3.0, channels=channels)

    # --- the mix ----------------------------------------------------------------------
    def mix(self, voice: np.ndarray, speech: list[tuple[float, float]], shots: list[Shot], offset: float,
            scenes: list[dict[str, Any]], stings: list[float],
            music_until: float, max_effects: int = 45) -> np.ndarray:
        """`voice` is the whole programme's speech track; shot times are on the story timeline, so
        `offset` (cold open + title) moves them onto the programme. `stings` are extra stinger times
        (the title card); every scare shot gets one as well."""
        rate, n = self.rate, voice.size
        total = n / rate

        # Speech ducking envelope.
        duck = np.ones(n, dtype=np.float32)
        for t0, t1 in speech:
            duck[int(max(t0 - 0.2, 0) * rate): int(min(t1 + 0.25, total) * rate)] = 0.0
        duck = _smooth(duck, int(0.3 * rate))  # 0 under speech, 1 between lines

        # Tension over time, per shot.
        level = np.zeros(n, dtype=np.int8)
        for s in shots:
            mood = str(scenes[s.scene].get("mood", "")) if s.scene < len(scenes) else ""
            level[int((s.t0 + offset) * rate): int((s.t1 + offset) * rate)] = level_of(s, mood)
        level[: int(offset * rate)] = TENSE if offset > 0 else 0  # the cold open is the scariest moment

        out = np.repeat(voice.astype(np.float32)[:, None], 2, axis=1)  # the voice in the centre

        # Music: each tension run plays its cue, looped, crossfaded at the joins (wide, in stereo).
        cues = {k: _stereo(v) for k, v in self.music(total).items()}
        bed = np.zeros((n, 2), dtype=np.float32)
        if cues:
            names = {CALM: "unease", UNEASY: "dread", TENSE: "terror"}
            for a, b, lv in _runs(level, rate, min_seconds=12.0):
                cue = cues[names[lv]] if names[lv] in cues else next(iter(cues.values()))
                bed[a:b] += _crossfaded(_loop(cue, b - a, start=a % max(len(cue), 1)), rate)
        else:
            self.report["synthesized"].append("music")
            bed = _wide(sound.drone(n, rate) * 0.5, rate)
        music_gain = (0.10 + 0.14 * duck)[:, None]
        end_i = int(min(music_until, total) * rate)
        bed[end_i:] = 0
        tail = int(3 * rate)
        bed[max(end_i - tail, 0):end_i] *= np.linspace(1, 0, end_i - max(end_i - tail, 0))[:, None]

        # Room tone per place (wide).
        room = np.zeros((n, 2), dtype=np.float32)
        places: dict[str, np.ndarray | None] = {}
        for s in shots:
            place = str(scenes[s.scene].get("location", "")).strip() if s.scene < len(scenes) else ""
            if place and place not in places and len(places) < 8:
                places[place] = None
        def amb(place: str) -> tuple[str, np.ndarray | None]:
            return place, self.effect(f"Loopable night ambience of {place}: quiet, eerie room tone, no music, "
                                      "no voices.", f"room_{digest(place)}", 20.0, loop=True, channels=2)
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            places.update(dict(pool.map(amb, list(places))))
        self.report["ambience"] = sum(1 for v in places.values() if v is not None)
        rng = np.random.default_rng(3)
        for s in shots:
            place = str(scenes[s.scene].get("location", "")).strip() if s.scene < len(scenes) else ""
            a, b = int((s.t0 + offset) * rate), min(int((s.t1 + offset) * rate), n)
            sample = places.get(place)
            if sample is None or not sample.size:
                sample = np.stack([sound._smooth_noise(max(b - a, 1), rate, 500, rng),
                                   sound._smooth_noise(max(b - a, 1), rate, 500, rng)], axis=1) * 0.3
            room[a:b] += _loop(_stereo(sample), b - a, start=a % max(len(sample), 1))
        room = np.stack([_smooth(room[:, 0], int(0.5 * rate)), _smooth(room[:, 1], int(0.5 * rate))], axis=1)
        room_gain = (0.10 + 0.08 * duck)[:, None]

        # Silence before each scare: music and room tone drop away for a beat (for a built scare, through
        # its whole dark build-up), and the stinger lands on the reveal.
        scare_times = [s.t0 + float(getattr(s, "reveal", 0) or 0) + offset for s in shots if s.scare] + list(stings)
        hush = np.ones(n, dtype=np.float32)
        built = {round(s.t0 + s.reveal + offset, 3): s.t0 + offset for s in shots if s.scare and getattr(s, "reveal", 0)}
        for t in scare_times:
            start = built.get(round(t, 3), t - 0.9)
            a, b = int(max(min(start, t - 0.9), 0) * rate), int(t * rate)
            hush[a:b] = 0.04
        hush = _smooth(hush, int(0.12 * rate))[:, None]
        out += bed * music_gain * hush + room * room_gain * hush

        # Stingers, wide.
        sting = self.effect(STING, "sting", 3.0, channels=2)
        if sting is None:
            self.report["synthesized"].append("stinger")
            sting = sound.cue("thunder", rate, rng)
        sting = _stereo(sting) if sting.ndim == 2 else _wide(sting, rate)
        for t in scare_times:
            _place(out, sting * 0.55, int(t * rate))

        # The channel's theme under the ritual line.
        if self.theme is not None and len(self.theme):
            for t in self.theme_at:
                _place(out, _stereo(self.theme) * 0.45, int(max(t - 0.25, 0) * rate))
            self.report["theme"] = len(self.theme_at)

        # One effect per shot that asks for one, each placed in the stereo field; past the limit, the
        # scares and the tensest beats keep theirs.
        asking = [s for s in shots if s.sound and f"fx_{s.index:03d}" not in self.skip]
        keep = {s.index for s in sorted(asking, key=lambda s: (not s.scare, -int(getattr(s, "tension", 2) or 2),
                                                                  -s.importance, s.index))[:max_effects]}
        wanted = [s for s in asking if s.index in keep]
        def fx(s: Shot) -> tuple[Shot, np.ndarray | None]:
            return s, self.effect(s.sound, f"fx_{s.index:03d}", None)
        placed: dict[str, int] = {}
        with ThreadPoolExecutor(max_workers=self.parallel) as pool:
            for s, sample in pool.map(fx, wanted):
                if sample is None:
                    names = sound.cues_for([s.sound])
                    sample = sound.cue(names[0], rate, rng) if names else None
                if sample is not None and sample.size:
                    self.report["effects"] += 1
                    a = int((s.t0 + offset + 0.15) * rate)
                    where = position(s.sound, s.index)
                    placed[where[0]] = placed.get(where[0], 0) + 1
                    _place(out, pan(_mono(sample), *where[1:]) * 0.4 * (0.55 + 0.45 * duck[a] if a < n else 1.0), a)
        self.report["stereo"] = placed

        fade = int(1.0 * rate)
        out[:fade] *= np.linspace(0.2, 1, min(fade, n))[: len(out[:fade])][:, None]
        out[-fade:] *= np.linspace(1, 0, min(fade, n))[: len(out[-fade:])][:, None]
        peak = float(np.max(np.abs(out))) or 1.0
        log.info("Sound: %s", self.report)
        return (out * min(1.0, 0.95 / peak)).astype(np.float32)


def hash_key(text: str) -> int:
    """A stable number for a word (Python's own hash changes from run to run)."""
    return int(digest(text), 16) % 1_000_003


def _joined(parts: list[np.ndarray], rate: int, seconds: float = 3.0) -> np.ndarray:
    """Pieces of music played one after another, each crossfaded into the next."""
    out = parts[0]
    for part in parts[1:]:
        k = min(int(seconds * rate), len(out) // 2, len(part) // 2)
        if k <= 0:
            out = np.concatenate([out, part])
            continue
        ramp = np.linspace(0, 1, k, dtype=np.float32)
        if out.ndim == 2:
            ramp = ramp[:, None]
        mixed = out[-k:] * (1 - ramp) + part[:k] * ramp
        out = np.concatenate([out[:-k], mixed, part[k:]])
    return out.astype(np.float32)


def _smooth(x: np.ndarray, width: int) -> np.ndarray:
    if width < 2:
        return x
    kernel = np.ones(width, dtype=np.float32) / width
    return np.convolve(x, kernel, mode="same").astype(np.float32)


def _loop(sample: np.ndarray, length: int, start: int = 0) -> np.ndarray:
    """`sample` repeated to `length` frames, from `start`: mono (n,) or stereo (n, 2)."""
    frames = len(sample)
    if frames == 0 or length <= 0:
        return np.zeros((max(length, 0),) + sample.shape[1:], dtype=np.float32)
    reps = (start + length) // frames + 1
    tiled = np.tile(sample, (reps,) + (1,) * (sample.ndim - 1))
    return tiled[start:start + length].astype(np.float32)


def _crossfaded(part: np.ndarray, rate: int, seconds: float = 2.0) -> np.ndarray:
    k = min(int(seconds * rate), len(part) // 2)
    if k > 0:
        ramp = np.linspace(0, 1, k)
        if part.ndim == 2:
            ramp = ramp[:, None]
        part[:k] *= ramp
        part[-k:] *= ramp[::-1]
    return part


def _stereo(sample: np.ndarray) -> np.ndarray:
    """(n, 2): a stereo sound as it is, a mono one in both channels."""
    return sample if sample.ndim == 2 else np.repeat(sample[:, None], 2, axis=1)


def _mono(sample: np.ndarray) -> np.ndarray:
    return sample.mean(axis=1) if sample.ndim == 2 else sample


def _wide(sample: np.ndarray, rate: int, ms: float = 11.0) -> np.ndarray:
    """A mono sound spread across the stereo field: the right channel a few milliseconds late."""
    shift = int(rate * ms / 1000)
    right = np.concatenate([np.zeros(shift, dtype=np.float32), sample[: max(len(sample) - shift, 0)]])
    return np.stack([sample, right[: len(sample)]], axis=1).astype(np.float32)


def pan(sample: np.ndarray, start: float, end: float | None = None) -> np.ndarray:
    """A mono sound placed in the stereo field (-1 left, 0 centre, 1 right), at equal loudness wherever
    it sits; with `end`, it moves from `start` to `end` over its length."""
    where = np.full(len(sample), start, dtype=np.float32) if end is None else np.linspace(start, end, len(sample))
    angle = (np.clip(where, -1, 1) + 1) * np.pi / 4
    return np.stack([sample * np.cos(angle), sample * np.sin(angle)], axis=1).astype(np.float32)


MOVING = ("footstep", "steps", "walk", "running", "runs past", "passes", "crawl", "scrap", "drag", "car passes")
CLOSE = ("whisper", "breath", "breathing")


def position(text: str, index: int) -> tuple[str, float, float | None]:
    """Where an effect sits: (kind, start, end). Whispers and breaths close to one ear, footsteps
    crossing from one side to the other, what the words place on a side there, the rest a little
    off centre, alternating, so effects never all come from the middle."""
    low = (text or "").lower()
    side = -1.0 if index % 2 else 1.0
    if " left" in f" {low}":
        return "side", -0.6, None
    if " right" in f" {low}":
        return "side", 0.6, None
    if any(w in low for w in CLOSE):
        return "close", 0.75 * side, None
    if any(w in low for w in MOVING):
        return "moving", -0.6 * side, 0.6 * side
    if any(w in low for w in ("behind", "upstairs", "outside", "distant", "far", "next room", "through the wall")):
        return "off", 0.5 * side, None
    return "near", 0.25 * side, None


def _runs(level: np.ndarray, rate: int, min_seconds: float) -> list[tuple[int, int, int]]:
    """Stretches of one tension level, short ones merged into their neighbour; overlapping by 2 s
    so the crossfades meet."""
    runs: list[list[int]] = []
    i = 0
    while i < level.size:
        j = i
        lv = int(level[i])
        step = rate // 4
        while j < level.size and int(level[j]) == lv:
            j += step
        j = min(j, level.size)
        if runs and (j - i) < min_seconds * rate:
            runs[-1][1] = j
        else:
            runs.append([i, j, lv])
        i = j
    overlap = 2 * rate
    return [(max(a - overlap, 0), min(b + overlap, level.size), lv) for a, b, lv in runs]


def _place(out: np.ndarray, sample: np.ndarray, at: int) -> None:
    if at >= len(out) or len(sample) == 0:
        return
    at = max(at, 0)
    b = min(at + len(sample), len(out))
    if out.ndim == 2 and sample.ndim == 1:
        sample = _stereo(sample)
    out[at:b] += sample[: b - at]
