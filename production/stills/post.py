"""Post-production agents: the edit plan, sound design, the render, packaging and QA."""

from __future__ import annotations

import json
import re
import logging
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import soundfile

from .. import draw, edit, package, shots as shotlist, timing
from ..edit import FPS, Segment, _ff, probe
from ..look import W, H
from ..library import Library
from ..score import Score
from ..vertex import traffic
from ..shots import Shot
from .board import Board
from .checks import thumbnail_ok
from . import camera
from .images import Painter
from . import sounds

log = logging.getLogger(__name__)

CHANNEL = "HOLLOW LINES"


class EditPlanAgent:
    """The running order: cold open, title card, the story, end screen; the speech track and captions."""

    name = "edit_plan"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        audio, rate, timeline = d["audio"], d["rate"], d["timeline"]
        story_seconds = timeline.duration
        cold = (edit.pick_cold_open(beats, d["scenes"], 6.0, 16.0, set(d.get("host_beats") or ()),
                                    float(cfg.get("cold_open_before", 0.65))) if cfg.get("cold_open", True) else [])
        cold = [i for i in cold if i not in d["host_beats"]]
        cold_a = beats[cold[0]].t0 if cold else 0.0
        cold_b = beats[cold[-1]].t1 if cold else 0.0
        cold_seconds = cold_b - cold_a
        title_seconds = float(cfg.get("title_seconds", 3.0))
        end_seconds = float(cfg.get("end_screen_seconds", 20.0))
        offset = cold_seconds + title_seconds
        total = offset + story_seconds + end_seconds
        track = np.zeros(int(round(total * rate)) + rate, dtype=np.float32)
        if cold:
            part = audio[int(cold_a * rate): int(cold_b * rate)]
            track[: part.size] = part
        track[int(offset * rate): int(offset * rate) + audio.size] = audio
        track = track[: int(round(total * rate))]
        speech = [(c.t0 + offset, c.t1 + offset) for c in timeline.chunks]
        speech += [(c.t0 - cold_a, c.t1 - cold_a) for c in timeline.chunks if cold and c.t0 >= cold_a and c.t1 <= cold_b + 0.01]
        cues = timing.caption_cues(timeline)
        prog = [timing.Cue(c.start - cold_a, c.end - cold_a, c.text) for c in cues
                if cold and c.start >= cold_a and c.end <= cold_b + 0.01]
        prog += [timing.Cue(c.start + offset, c.end + offset, c.text) for c in cues]
        (board.out / "captions.srt").write_text(timing.srt(prog))
        # A scare is built, not shown at once: the picture stays nearly dark while the line begins, then
        # cuts to the reveal (the stinger and a jolt of the camera land there, on the cut).
        built = []
        if cfg.get("scare_build", True):
            for b in beats:
                if b.scare and b.seconds >= 3.0 and b.index not in d.get("host_beats", ()):
                    b.reveal = round(min(max(0.4 * b.seconds, 1.0), 2.5), 2)
                    built.append(b.index)
        # Where each beat sits in the finished video: YouTube's retention curve is read against it later.
        d["beat_times"] = [{"beat": b.index, "start": round(b.t0 + offset, 2), "end": round(b.t1 + offset, 2),
                            "scene": b.scene, "tension": b.tension, "scare": b.scare, "line": str(b.text)[:90]}
                           for b in beats]
        d.update(cold=cold, cold_seconds=cold_seconds, title_seconds=title_seconds, end_seconds=end_seconds,
                 offset=offset, total=total, track=track, speech=speech, cues=prog, story_seconds=story_seconds)
        return {"cold_open_beats": cold, "cold_open_seconds": round(cold_seconds, 1),
                "total_seconds": round(total, 1), "scares_built": built}


class SoundAgent:
    """Score cues that follow the tension, room tone per place, effects, a stinger per scare.

    Sounds come from the channel's sound library first (picked by the planner from a short
    list per need); only what the library cannot cover is generated, and that is saved for
    the library afterwards."""

    name = "sound"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d = board.data
        score = Score(board.studio, board.work / "sound", d["rate"], int(board.cfg.get("parallel", 4)))
        score.cue_names = list(board.cfg.get("music_cues") or score.cue_names)
        score.min_music = float(board.cfg.get("music_min_seconds", 60.0))
        lib_cfg = dict(board.cfg.get("sound_library") or {})
        library = Library.load(lib_cfg, board.work / "library") if lib_cfg.get("enabled", True) else None
        if library is not None and not library.sounds:
            library = None
        # The sound director decides each sound: from the library, newly made, or left out.
        can_make = bool(lib_cfg.get("generate_missing", True)) and board.studio is not None
        # Which kinds may be made (`generate_kinds`: "music" and/or "sfx" for effects, room tone and
        # the sting), and only those whose model is configured.
        kinds = {k for k in (lib_cfg.get("generate_kinds") or ("music", "sfx"))
                 if can_make and board.studio is not None and board.studio.models.get(k)}
        # An original score: with Lyria on Vertex (paid from the Google credits, a few cents a cue) every
        # cue is made for this episode first; the library's pick plays only if making fails.
        music_model = str(board.studio.models.get("music", "")) if board.studio is not None else ""
        original = (can_make and lib_cfg.get("music_source", "original") == "original"
                    and (music_model.startswith("vertex:") or "elevenlabs/music" in music_model))
        score.max_music = float(board.cfg.get("music_max_seconds", 150))
        if original:
            kinds.add("music")
            score.prefer_made = {f"music_{c}" for c in score.cue_names}
            score.motif = str((d.get("bible") or {}).get("sound_motif", "")).strip()[:200]
            from ..paid import digest
            score.seed = int(digest(board.story.get("id"), board.story.get("title")), 16) % 1_000_000_007
        can_make = can_make and bool(kinds)
        decided = sounds.pick(board, library, score.cue_names, int(board.cfg.get("max_effects", 30)),
                              float(lib_cfg.get("min_score", 0.55)), can_make=can_make,
                              music_made="music" in kinds)
        library_report: dict[str, Any] = dict(decided.report, library=library is not None)
        score.prompts, score.skip = dict(decided.make), set(decided.skip)
        # a script's own score (e.g. an episode that is not horror): its words for each cue win
        score.prompts.update({f"music_{k}": str(v) for k, v in (board.cfg.get("music_prompts") or {}).items()})
        if library is not None:
            score.chosen = sounds.fetch_all(library, decided.chosen, int(board.cfg.get("parallel", 4)))
            used = [decided.chosen[n].id for n in score.chosen]
            d["sounds_used"] = used
            d["credits"] = library.credits(used)
            library_report.update(fetched=len(score.chosen), credits=len(d["credits"]))
        score.generate_missing = can_make
        score.make_kinds = kinds
        stings = [d["cold_seconds"]] if d["cold"] and d["title_seconds"] else []
        theme_file = Path(str(board.cfg.get("theme_file") or ""))
        if theme_file.name and not theme_file.is_absolute():
            theme_file = Path(__file__).resolve().parents[2] / theme_file
        if theme_file.name and theme_file.is_file():
            try:
                from ..speech import decode
                score.theme = decode(theme_file, d["rate"], channels=2)
                score.theme_at = [c.t0 + d["offset"] for c in d["timeline"].chunks
                                  if "lantern is lit" in str(c.text).lower()][:2]
            except Exception as exc:  # noqa: BLE001 - a missing theme never stops the sound
                log.warning("Theme not used: %s", exc)
        try:
            mixed = score.mix(d["track"], d["speech"], d["beats"], d["offset"], d["scenes"], stings=stings,
                              music_until=d["total"], max_effects=int(board.cfg.get("max_effects", 30)))
        except Exception:
            board.data["mix"] = board.work / "mix.wav"
            soundfile.write(board.data["mix"], d["track"], d["rate"])  # the voices alone, if all else fails
            raise
        mixed, balance = voice_over_bed(mixed, d["track"], d["speech"], d["rate"],
                                        float(board.cfg.get("voice_over_bed_db", 12.0)))
        board.data["mix"] = board.work / "mix.wav"
        soundfile.write(board.data["mix"], mixed, d["rate"])
        soundfile.write(board.out / "narration.wav", d["audio"], d["rate"])
        saved = sounds.save_generated(score.generated, board.out, board.story) \
            if lib_cfg.get("save_generated", True) else 0
        (board.out / "sounds_used.json").write_text(json.dumps(d.get("sounds_used", [])))
        return dict(score.report, library=library_report, generated=len(score.generated), saved_for_library=saved,
                    mix=balance, original_score=original)


def voice_over_bed(mixed: np.ndarray, voice: np.ndarray, speech: list[tuple[float, float]], rate: int,
                   want_db: float = 12.0) -> tuple[np.ndarray, dict[str, Any]]:
    """The mix check (`speech`: programme times): while someone speaks, the voice must stand `want_db` above the music, room tone
    and effects under it. If it doesn't, the bed is lowered (the voice is never touched) and the
    check is repeated. Returns the mix and what was measured."""
    n = min(len(mixed), len(voice))
    mask = np.zeros(n, dtype=bool)
    for a, b in speech:
        mask[int(max(a, 0) * rate): int(max(b, 0) * rate)] = True
    if not mask.any():
        return mixed, {"checked": False}
    stereo = mixed.ndim == 2
    centre = voice[:n, None] if stereo else voice[:n]  # the voice sits in the middle of a stereo mix
    bed = mixed[:n] - centre
    heard = bed.mean(axis=1) if stereo else bed

    def ratio(gain: float) -> float:
        v = float(np.sqrt(np.mean(voice[:n][mask] ** 2)) + 1e-9)
        x = float(np.sqrt(np.mean((heard[mask] * gain) ** 2)) + 1e-9)
        return 20 * np.log10(v / x)

    before = ratio(1.0)
    gain = 1.0
    if before < want_db:
        gain = float(10 ** ((before - want_db) / 20))
        out = mixed.copy()
        out[:n] = centre + bed * gain
        mixed = out
    return mixed, {"checked": True, "voice_over_bed_db": round(before, 1),
                   "after_db": round(ratio(gain), 1), "bed_lowered_db": round(-20 * np.log10(gain), 1) if gain < 1 else 0}


def render(seg: Segment, work: Path, finish: dict[str, float], seed: int = 0, size: tuple[int, int] = (W, H),
           textures: dict[str, Path] | None = None) -> str:
    """One still for exactly its frames: laid out for its move, grained, moved, given its
    atmosphere, vignetted, faded."""
    n = seg.frames()
    seconds = n / FPS
    move = seg.camera if seg.camera in camera.MOVES else "none"
    plate = work / f"{Path(seg.name).stem}_canvas.png"
    if seg.clip is not None:  # a moving clip: it carries its own movement, the finish is the same
        source = (["-stream_loop", "-1"] if seg.loop else []) + ["-i", str(seg.clip)]
        base = camera.clip_filter(seg.clip, seconds, float(finish.get("grain", 0)), size,
                                  max_stretch=1.0 if seg.loop else 1.25)
        move = "clip"
    elif seg.layers is not None:  # depth parallax: the near layer moves more than the far one
        back, front = camera.parallax_routes(seconds, seg.tension)
        bg, fg = seg.layers
        layer = work / f"{Path(seg.name).stem}_front.png"
        size_bg = camera.canvas(bg, plate, max(back[0], back[1]), float(finish.get("grain", 0)), seed, size,
                                float(finish.get("halation", 0)))
        size_fg = camera.canvas_layer(fg, layer, max(front[0], front[1]), float(finish.get("grain", 0)), seed, size)
        source = ["-loop", "1", "-framerate", str(FPS), "-i", plate.name,
                  "-loop", "1", "-framerate", str(FPS), "-i", layer.name]
        base = (f"{camera.move_filter(back, n, size_bg, size)}[far];[1:v]"
                f"{camera.move_filter(front, n, size_fg, size, alpha=True)}[near];[far][near]overlay=format=yuv420")
        move = "parallax"
    else:
        route = camera.path(move, seconds, seg.tension)
        canvas = camera.canvas(seg.picture, plate, max(route[0], route[1]), float(finish.get("grain", 0)), seed,
                               size, float(finish.get("halation", 0)))
        source = ["-loop", "1", "-framerate", str(FPS), "-i", plate.name]
        base = camera.move_filter(route, n, canvas, size)
    inputs, steps = camera.atmosphere(seg.atmosphere, textures or {}, 2 if seg.layers is not None and
                                      seg.clip is None else 1, size)
    tail = [camera.vignette_filter(float(finish.get("vignette", 0))),
            camera.letterbox_filter(float(finish.get("letterbox", 0)), size)]
    if seg.dim < 1.0:  # a scare's build-up: the same picture, nearly dark (in YUV, so the colour labels hold)
        k = max(min(seg.dim, 1.0), 0.0)
        tail.insert(0, f"lutyuv=y='16+(val-16)*{k:.3f}':u='128+(val-128)*{k:.3f}':v='128+(val-128)*{k:.3f}'")
    if seg.fade_in:
        tail.append(f"fade=t=in:st=0:d={seg.fade_in:.2f}")
    if seg.fade_out:
        tail.append(f"fade=t=out:st={max(seconds - seg.fade_out, 0):.2f}:d={seg.fade_out:.2f}")
    tail = [t for t in tail if t]
    graph = ";".join([f"[0:v]{base}[shot]", *steps,
                      f"[shot]{','.join(tail) or 'null'}[out]"])
    still = move == "none" and not seg.atmosphere
    tune = ["-tune", "stillimage"] if still else []
    # aq-mode 3 spends more of the bitrate on dark areas, where a horror film lives and where
    # an encoder otherwise leaves blocks and bands.
    _ff([*source, *inputs, "-filter_complex", graph, "-map", "[out]",
         "-frames:v", str(n), "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-x264-params", "aq-mode=3",
         # rain and flicker change every pixel every frame: a ceiling keeps the file sane (YouTube
         # asks for 16-24 Mbit/s at 1440p)
         "-maxrate", "24M", "-bufsize", "24M",
         *tune, "-pix_fmt", "yuv420p", *camera.COLOUR_TAGS, "-r", str(FPS), "-an", seg.name], work)
    plate.unlink(missing_ok=True)
    (work / f"{Path(seg.name).stem}_front.png").unlink(missing_ok=True)
    return seg.name


class EditAgent:
    """Renders the programme: every still for exactly its frames, dips to black between scenes."""

    name = "edit"
    critical = True

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        titles = d["packaging"].get("titles") or [board.story.get("title", "")]
        frame = d.get("style_frame")
        size = tuple(int(v) for v in cfg.get("video_size") or (W, H))
        k = size[1] / 1080  # lettering scales with the frame
        # The title card changes from episode to episode (where the words sit, what lies behind them): the
        # same card on every video is what YouTube's rules on mass-produced content look for.
        anchor, left, behind = title_layout(board.story, cfg)
        background = frame
        if behind == "key":
            key = max((b for b in beats if b.index not in d.get("host_beats", ()) and not b.scare),
                      key=lambda b: (b.importance, -b.index), default=None)  # never a scare: no spoilers
            background = d["pictures"].get(key.index) if key is not None else frame
        title_card = edit.card(background or frame, [(CHANNEL, round(40 * k), draw.RED)] +
                               [(line, round(92 * k), draw.CHALK) for line in edit.wrap(str(titles[0]), 24)],
                               board.work / "title_card.png", left=left, size=size, anchor=anchor,
                               dim=0.25 if behind == "key" else 0.35)
        end_line = str(d["packaging"].get("end_screen_line", "")) or "Another story is waiting."
        end_card = edit.card(frame, [(line, round(60 * k), draw.CHALK) for line in edit.wrap(end_line, 30)] +
                             [("", round(30 * k), draw.CHALK), ("NEXT STORY  >", round(54 * k), draw.RED)],
                             board.work / "end_card.png", left=True, dim=0.25, size=size)
        segs = edit.programme(beats, d["pictures"], d["cold"], d["cold_seconds"], d["title_seconds"],
                              d["end_seconds"], d["story_seconds"], title_card, end_card)
        first = len(d["cold"]) + (1 if d["title_seconds"] else 0)
        story = segs[first:first + len(beats)]
        motion = str(cfg.get("motion", "drift"))
        for seg in segs:  # the end card holds still (YouTube's end-screen elements sit on it)
            seg.camera = "none"
        cold = segs[:len(d["cold"])]
        for seg, beat in zip(story, beats):
            seg.tension = int(getattr(beat, "tension", 2) or 2)
        if motion == "drift":  # every story picture gets the move that suits it
            for n, (seg, beat) in enumerate(zip(story, beats)):
                seg.camera = camera.move_for(beat, n, after_scare=n > 0 and beats[n - 1].scare)
            for n, seg in enumerate(cold):
                seg.camera = "push_in_fast" if n == len(cold) - 1 else "push_in"
            if d["title_seconds"]:
                segs[len(cold)].camera = "push_in"
        clips: dict[int, Path] = d.get("clips") or {}
        for seg, beat in list(zip(story, beats)) + [(seg, beats[i]) for seg, i in zip(cold, d["cold"])]:
            seg.clip = clips.get(beat.index)  # a beat the motion stage animated plays its clip
        layered: dict[int, tuple[Path, Path]] = d.get("layers") or {}
        for seg, beat in zip(story, beats):  # depth parallax where the depth agent split the picture
            if beat.index in layered and seg.clip is None and motion == "drift" and seg.camera != "none":
                seg.layers = layered[beat.index]
        loop = host_loop(cfg)
        if loop is not None:  # the host breathes: his own looping clip, made once, under every intro and outro
            for seg, beat in zip(story, beats):
                if beat.index in d.get("host_beats", ()) and seg.clip is None:
                    seg.clip, seg.loop = loop, True
        textures: dict[str, Path] = {}
        if cfg.get("atmosphere", True):  # rain, fog or a failing light, where the storyboard saw them
            for seg, beat in list(zip(story, beats)) + [(seg, beats[i]) for seg, i in zip(cold, d["cold"])]:
                seg.atmosphere = tuple(a for a in beat.atmosphere if a in camera.ATMOSPHERES)
            needed = {a for seg in segs for a in seg.atmosphere}
            if "rain" in needed:
                textures["rain"] = camera.rain_texture(board.work / "atmosphere_rain.png", size)
            if "fog" in needed:
                textures["fog"] = camera.fog_texture(board.work / "atmosphere_fog.png", size)
        dip = float(cfg.get("scene_dip_seconds", 0.35))
        for k in range(len(story) - 1):  # a breath of black where the scene changes
            if beats[k].scene != beats[k + 1].scene and dip:
                story[k].fade_out = max(story[k].fade_out, dip)
                story[k + 1].fade_in = max(story[k + 1].fade_in, dip)
        segs = build_scares(segs, story, beats, float(cfg.get("scare_build_dim", 0.22)), motion == "drift")
        look = d.get("look") or {}
        finish = {"grain": float(look.get("grain", 0)), "vignette": float(look.get("vignette", 0)),
                  "halation": float(look.get("halation", 0)), "letterbox": float(cfg.get("letterbox", 0) or 0)}
        work = board.work
        with ThreadPoolExecutor(max_workers=int(cfg.get("render_workers", 3))) as pool:
            names = list(pool.map(lambda job: render(job[1], work, finish, job[0], size, textures), enumerate(segs)))
        (work / "segments.txt").write_text("".join(f"file '{n}'\n" for n in names))
        _ff(["-f", "concat", "-safe", "0", "-i", "segments.txt", "-c", "copy", "picture.mp4"], work)
        burn = None
        if cfg.get("burn_captions"):
            burn = work / "captions.ass"
            burn.write_text(timing.ass(d["cues"], *size))
        vf = ["-vf", f"subtitles={burn.name}"] if burn else []
        codec = ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-x264-params", "aq-mode=3", "-pix_fmt",
                 "yuv420p", *camera.COLOUR_TAGS] if vf else ["-c:v", "copy"]
        video = board.out / "video.mp4"
        _ff(["-i", "picture.mp4", "-i", str(d["mix"]), *vf, *codec,
             "-af", loudnorm(Path(d["mix"]), float(cfg.get("loudness_lufs", -14.0))), "-ar", "48000", "-ac", "2",
             "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(video)], work)
        d.update(video=video, end_line=end_line, titles=titles)
        moves: dict[str, int] = {}
        air: dict[str, int] = {}
        for seg in segs:
            kind = "clip" if seg.clip is not None else seg.camera
            moves[kind] = moves.get(kind, 0) + 1
            for a in seg.atmosphere:
                air[a] = air.get(a, 0) + 1
        return {"segments": len(segs), "size": f"{size[0]}x{size[1]}", "motion": motion, "moves": moves,
                "atmosphere": air, "seconds": round(probe(video), 1)}


def host_loop(cfg: dict[str, Any]) -> Path | None:
    """The host's looping clip (`host_loop`), once it has been made and kept in the repository."""
    name = str(cfg.get("host_loop") or "")
    if not name:
        return None
    path = Path(name) if Path(name).is_absolute() else Path(__file__).resolve().parents[2] / name
    return path if path.is_file() else None


TITLE_LAYOUTS = (("center", False, "frame"), ("lower", True, "key"), ("upper", True, "frame"),
                 ("lower", False, "key"))


def title_layout(story: dict[str, Any], cfg: dict[str, Any]) -> tuple[str, bool, str]:
    """(where the words sit, left-aligned, the background: the style frame or the key picture), chosen
    by the story so the same episode always gets the same card."""
    if not cfg.get("title_variety", True):
        return TITLE_LAYOUTS[0]
    from ..paid import digest

    return TITLE_LAYOUTS[int(digest(story.get("id"), story.get("title")), 16) % len(TITLE_LAYOUTS)]


def build_scares(segs: list[Segment], story: list[Segment], beats: list[Shot], dim: float,
                 moving: bool = True) -> list[Segment]:
    """Each built scare (a beat with a `reveal`) becomes two shots: its picture held nearly dark while
    the line begins, then a hard cut to the full picture at the reveal, with a jolt of the camera."""
    split = {id(seg): beat for seg, beat in zip(story, beats) if getattr(beat, "reveal", 0) > 0 and seg.clip is None}
    out: list[Segment] = []
    for seg in segs:
        beat = split.get(id(seg))
        if beat is not None:
            cut = seg.start + beat.reveal
            if cut - seg.start >= 0.8 and seg.end - cut >= 1.0:
                out.append(Segment(seg.name.replace(".mp4", "_build.mp4"), seg.start, cut, seg.picture,
                                   camera="push_in" if moving else "none", atmosphere=seg.atmosphere,
                                   fade_in=seg.fade_in, tension=seg.tension, dim=dim))
                seg.start, seg.fade_in = cut, 0.0
                seg.camera = "jolt" if moving else "none"
        out.append(seg)
    return out


THUMB_ANGLES = (
    "the main character in close-up, frozen in fear, eyes wide, the threat looming in the darkness just behind them",
    "the threat itself, closer than it should be: a dark shape with one terrible detail catching the light",
    "a single small figure in the story's key place at night, something wrong in the darkness around them",
)


class PackagerAgent:
    """Thumbnails to test, chapters, the upload details and the review kit."""

    name = "packager"
    critical = False

    def thumbnails(self, board: Board) -> list[Path]:
        cfg, d = board.cfg, board.data
        painter: Painter = d["painter"]
        text = str(d["packaging"].get("thumbnail_text", "")).strip()
        brief = str(d["packaging"].get("thumbnail_brief", "")).strip()
        beats: list[Shot] = d["beats"]
        key = next((b for b in beats if b.scare), None) or max(beats, key=lambda b: b.importance)
        refs = [p for p in [d.get("style_frame"), d["pictures"].get(key.index), *list(d.get("sheets", {}).values())[:2]]
                if p is not None]
        made = []
        for k, angle in enumerate(THUMB_ANGLES[: int(cfg.get("thumbnails", 3))]):
            out = board.out / f"thumbnail_{k + 1}.jpg"
            lettering = str((d.get("look") or {}).get("lettering") or "big bold hand-lettered words")
            words = (f" The words \"{text.upper()}\" in {lettering}, in the left third, the last word in blood "
                     "red, readable at phone size; nothing else written." if text else " No text.")
            prompt = (f"A YouTube thumbnail. {d['house']} Match the look of the reference images and keep each "
                      f"person's face exactly as in them.{words} Picture: "
                      f"{angle}. Brief: {brief} Extreme contrast, one clear focal point, lots of darkness, nothing "
                      "small or busy.")
            raw = board.work / f"thumb_{k + 1}.png"
            got, _ = painter.paint(prompt, refs, raw, ["thumbnail", k, prompt], cfg["chains"]["thumbnail"])
            if got is not None:  # the words must read exactly right: one repaint if they don't
                ok, why = thumbnail_ok(board.studio, got, text)
                self.checks.append({"thumbnail": k + 1, "ok": ok, "why": why})
                if not ok:
                    again = prompt + f" Correct this: {why}. Spell the words exactly, one time each."
                    raw2 = board.work / f"thumb_{k + 1}_fix.png"
                    fixed, _ = painter.paint(again, refs, raw2, ["thumbnail", k, again], cfg["chains"]["thumbnail"])
                    if fixed is not None:
                        ok2, why2 = thumbnail_ok(board.studio, fixed, text)
                        self.checks.append({"thumbnail": k + 1, "repainted": True, "ok": ok2, "why": why2})
                        if ok2:  # both failing: the first stands, QA flags it
                            got = fixed
            source = got or d["pictures"][key.index]
            from PIL import Image
            with Image.open(source) as im:
                picture = im.convert("RGB")
                if got is None:
                    picture = draw.thumbnail(picture, text).convert("RGB")
                picture.resize((1280, 720), Image.LANCZOS).save(out, quality=90)
            made.append(out)
        return made

    def run(self, board: Board) -> dict[str, Any]:
        cfg, d = board.cfg, board.data
        beats: list[Shot] = d["beats"]
        self.checks: list[dict[str, Any]] = []
        thumbs = self.thumbnails(board)
        package.contact_sheet(beats, d["pictures"], board.out / "contact_sheet.jpg", moving=set(d.get("clips") or {}))
        package.preview(d["video"], board.out / "preview_60s.mp4")
        chapters = shotlist.chapters(board.router, d["scenes"], beats, d["offset"])
        description = str(d["packaging"].get("description", ""))
        if d.get("credits"):  # CC BY music and sounds: the credit lines their licenses ask for
            description += "\n\nMusic and sound:\n" + "\n".join(d["credits"])
        if chapters:
            description += "\n\n" + "\n".join(f"{shotlist.stamp(t)} {name}" for t, name in chapters)
        story = board.story
        metadata = {
            "story_id": story.get("id"), "title": d["titles"][0], "other_titles": d["titles"][1:],
            "description": description, "tags": d["packaging"].get("tags", []),
            "pinned_comment": d["packaging"].get("pinned_comment", "") or story.get("comment_question", ""),
            "end_screen_line": d["end_line"], "credit": story.get("description_credit", ""), "made_for_kids": False,
            "duration_seconds": round(d["total"], 1), "chapters": [[round(t, 1), n] for t, n in chapters],
            "sound_credits": d.get("credits", []), "sounds_used": d.get("sounds_used", []),
            "thumbnails": [p.name for p in thumbs],
            "end_screen_starts": round(d["offset"] + d["story_seconds"], 1),
            # YouTube's "altered or synthetic content" label is for realistic made-up people and events:
            # on for the cinematic look, off for illustrations; stills.synthetic_media true/false overrides.
            "synthetic_media": bool((d.get("look") or {}).get("synthetic_media", False)
                                    if cfg.get("synthetic_media") is None else cfg.get("synthetic_media")),
        }
        languages = [str(x) for x in (cfg.get("languages") or [])]
        translated: dict[str, Any] = {}
        if languages and board.studio is not None:  # titles, descriptions and captions in other languages
            from .translate import localize
            translated = localize(board.studio, metadata["title"], description, d.get("cues") or [], languages,
                                  board.out)
            metadata.update(localizations=translated["localizations"], captions=translated["captions"])
        (board.out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        (board.out / "beats.json").write_text(shotlist.save(beats))
        return {"thumbnails": len(thumbs), "chapters": len(chapters),
                "thumbnail_checks": self.checks, "languages": sorted(translated.get("localizations", {})),
                "caption_languages": sorted(translated.get("captions", {}))}


def streams(path: Path) -> list[str]:
    result = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_type", "-of", "csv=p=0",
                             str(path)], capture_output=True, text=True)
    return [s.strip() for s in result.stdout.split() if s.strip()]


class QAAgent:
    """Checks the finished files before a person reviews them; never blocks, only reports."""

    name = "qa"
    critical = False

    def run(self, board: Board) -> dict[str, Any]:
        d, out = board.data, board.out
        issues = []
        video = out / "video.mp4"
        if not video.exists():
            return {"status": "fail", "issues": ["no video.mp4"]}
        seconds = probe(video)
        if abs(seconds - d["total"]) > 1.0:
            issues.append(f"video is {seconds:.1f} s, the programme {d['total']:.1f} s")
        kinds = streams(video)
        if "video" not in kinds or "audio" not in kinds:
            issues.append(f"streams: {kinds}")
        art = board.reports.get("illustrator", {})
        painted = art.get("pictures", 0)
        if d["beats"] and painted < 0.8 * len(d["beats"]):
            issues.append(f"only {painted} of {len(d['beats'])} beats have their own picture")
        if art.get("placeholders"):
            issues.append(f"{art['placeholders']} placeholder pictures")
        scares = {b.index for b in d["beats"] if b.scare}
        for i in art.get("doubted_beats", []):
            if i in scares:
                issues.append(f"the scare picture (beat {i}) passed no check: look at it")
        voice = board.reports.get("voice", {})
        if voice.get("lines_read_by_kokoro"):
            issues.append(f"{voice['lines_read_by_kokoro']} lines fell back to Kokoro")
        heard = voice.get("listener") or {}
        if heard.get("worst_match", 1.0) < 0.8:
            issues.append(f"a take matched its script only {heard['worst_match']:.0%} (listen to it)")
        issues += [f"script: {i}" for i in d.get("script_issues", [])]
        issues += [f"reference: {i}" for i in d.get("reference_issues", [])]
        review = board.reports.get("art_review", {})
        if review.get("reviewed") and isinstance(review.get("score"), int) and review["score"] < 6:
            issues.append(f"art review {review.get('score')}/10: "
                          + "; ".join(f["problem"][:80] for f in review.get("flagged", [])[:3]))
        final: dict[int, dict[str, Any]] = {}
        for t in board.reports.get("packager", {}).get("thumbnail_checks", []):
            final[t["thumbnail"]] = t  # the last check of each thumbnail is the one that stands
        for k, t in sorted(final.items()):
            if not t.get("ok"):
                issues.append(f"thumbnail {k}: {t.get('why', '')}")
        offset = float(d.get("offset") or 0.0)
        meant = [(b.t0 + offset, b.t0 + b.reveal + offset) for b in d["beats"] if getattr(b, "reveal", 0)]
        issues += picture_check(video, seconds, meant)
        lufs, peak = loudness(video)
        target = float(board.cfg.get("loudness_lufs", -14.0))
        if lufs is not None and abs(lufs - target) > 2.0:
            issues.append(f"loudness {lufs:.1f} LUFS (target {target:.0f})")
        if peak is not None and peak > -0.5:
            issues.append(f"true peak {peak:.1f} dBTP: may clip on some players")
        fell = int(getattr(getattr(board.studio, "fal", None), "fell_back", 0) or 0)
        if fell > int(board.cfg.get("max_fallbacks", 5)):
            issues.append(f"{fell} pictures or checks went to fal because Vertex failed (at fal's price)")
        google = traffic(getattr(board.studio, "fal", None))
        if google.get("stopped"):
            issues.append(f"Google stopped for this run: {google['stopped']}")
        refused = board.reports.get("illustrator", {}).get("google_refused") or {}
        if refused:
            issues.append(f"Google refused {len(refused)} picture(s) for their content ({', '.join(sorted(refused)[:5])}): "
                          "change those lines or pictures in the script; they are never asked for again")
        if int(google.get("busy_answers", 0)) > 10:
            issues.append(f"Google answered 'busy' {google['busy_answers']} times: lower stills.parallel")
        for warning in (board.reports.get("illustrator", {}).get("redo_warnings") or [])[:3]:
            issues.append(warning)
        screen = board.reports.get("screening", {})
        tried = str(screen.get("why", "")) and not str(screen.get("why", "")).startswith(("only Gemini", "no budget"))
        if screen and screen.get("watched") is False and tried:  # it tried to watch and failed
            issues.append(f"nobody watched the cut (screening: {str(screen['why'])[:100]})")
        review = board.reports.get("art_review", {})
        if review and review.get("reviewed") is False:
            issues.append("the art review did not run: the pictures were never compared side by side")
        for note in [n for n in screen.get("notes", []) if n.get("severity") == 3][:3]:
            where = f" (beat {note['beat']})" if note.get("beat") is not None else ""
            issues.append(f"screening at {note.get('at')}{where}: {note['note'][:120]}")
        if screen.get("redo_hint"):
            issues.append(f"screening: to repaint those pictures, run again with resume=<this run> "
                          f"redo={screen['redo_hint']}")
        pre = board.reports.get("preflight", {})
        if pre.get("warning"):
            issues.append(f"preflight: {pre['warning']}")
        if str(pre.get("checker", "")).startswith("failed"):
            issues.append("the checker did not answer before the run: pictures may have been used unchecked")
        for name in ("metadata.json", "captions.srt", "thumbnail_1.jpg", "contact_sheet.jpg"):
            if not (out / name).exists():
                issues.append(f"missing {name}")
        for name, report in board.reports.items():
            if isinstance(report, dict) and report.get("error"):
                issues.append(f"{name} failed: {report['error'][:120]}")
        return {"status": "pass" if not issues else "check", "issues": issues, "video_seconds": round(seconds, 1),
                "loudness_lufs": lufs, "true_peak_dbtp": peak, "vertex_fallbacks": fell}


def picture_check(video: Path, seconds: float, meant: list[tuple[float, float]] | None = None) -> list[str]:
    """The finished picture, watched by FFmpeg: no stretch of black (a missing or failed picture
    shows as one; scene dips and fades are well under the limit; `meant` are the dark build-ups of
    the scares, dark on purpose), and the right colour labels."""
    issues = []
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(video), "-map", "0:v:0", "-vf",
                             "scale=480:-2,blackdetect=d=1.5:pic_th=0.98:pix_th=0.06", "-an", "-f", "null", "-"],
                            capture_output=True, text=True)
    for start, length in re.findall(r"black_start:([\d.]+).*?black_duration:([\d.]+)", result.stderr):
        a, b = float(start), float(start) + float(length)
        if any(x - 0.5 <= a and b <= y + 0.5 for x, y in meant or []):
            continue  # a scare's build-up, dark on purpose
        if float(start) + float(length) < seconds - 0.5:  # the fade to black at the very end is meant
            issues.append(f"{float(length):.1f} s of black picture at {shotlist.stamp(float(start))}")
    tags = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                           "stream=color_space", "-of", "csv=p=0", str(video)], capture_output=True, text=True)
    if tags.stdout.strip() != "bt709":
        issues.append(f"colours labelled {tags.stdout.strip() or 'nothing'} (HD video should be bt709)")
    return issues[:5]


def loudnorm(path: Path, target: float) -> str:
    """FFmpeg's loudness filter, measured first: one pass alone lands a short video a couple of
    LUFS under the target (run 94: -16.1 for -14); with the file's own measurements it lands on it."""
    plain = f"loudnorm=I={target}:TP=-1.5:LRA=11"
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af",
                             plain + ":print_format=json", "-f", "null", "-"], capture_output=True, text=True)
    match = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", result.stderr)
    try:
        m = json.loads(match.group(0)) if match else {}
        values = [float(m[k]) for k in ("input_i", "input_tp", "input_lra", "input_thresh", "target_offset")]
    except (KeyError, TypeError, ValueError):
        return plain
    if any(v != v or abs(v) == float("inf") for v in values):  # silence measures as -inf
        return plain
    i, tp, lra, thresh, offset = values
    return (f"{plain}:measured_I={i}:measured_TP={tp}:measured_LRA={lra}:measured_thresh={thresh}"
            f":offset={offset}:linear=true")


def loudness(path: Path) -> tuple[float | None, float | None]:
    """Integrated loudness (LUFS) and true peak (dBTP) of a file's sound, by FFmpeg's EBU R128 meter."""
    result = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-map", "0:a:0",
                             "-af", "ebur128=peak=true", "-f", "null", "-"], capture_output=True, text=True)
    text = result.stderr[-3000:]
    lufs = re.findall(r"I:\s+(-?[\d.]+) LUFS", text)
    peak = re.findall(r"Peak:\s+(-?[\d.]+) dBFS", text)
    try:
        return (float(lufs[-1]) if lufs else None), (float(peak[-1]) if peak else None)
    except ValueError:
        return None, None
