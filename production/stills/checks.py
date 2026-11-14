"""The team's verifiers before and after the work.

- ScriptCheckAgent (first): the script is in a state to be produced: every scene and pause
  is found in the narration, quotes are closed, and the narration's length suits the plan.
  Code only, free; problems are reported, and the ones that would break timing stop the video.
- ArtReviewAgent (after the illustrator, before the edit): the pictures as one sequence. Each
  picture was checked alone by the critic; the vision model now sees them all together with
  the narration and flags what only shows side by side (a person who changes, a picture that
  breaks the story). The worst few are redrawn with its note. About $0.002 plus any redraw.
- thumbnail_ok (used by the packager): the thumbnail's words are spelled right and readable.
- sheet_ok (used by the art director): a character or location sheet shows what its design
  says, before every picture is painted from it. One wrong sheet is a wrong film.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from production.media import MediaError, data_uri

from .. import package
from ..paid import OverBudget, _small
from .board import Board, _json

log = logging.getLogger(__name__)

WORDS_PER_SECOND = 2.5  # the narrator's natural speed before voice_pace (about 150 words a minute)


class ScriptCheckAgent:
    name = "script_check"
    critical = True  # only for a script that cannot be timed at all; everything else is a warning

    def run(self, board: Board) -> dict[str, Any]:
        story, cfg = board.story, board.cfg
        narration = str(story.get("narration") or "").strip()
        scenes = [s for s in _json(story.get("scenes"), []) if isinstance(s, dict)]
        if not narration or not scenes:
            raise ValueError(f"story #{story.get('id')} has no narration or scene plan")
        issues: list[str] = []
        cursor = 0
        for i, sc in enumerate(scenes, 1):
            text = str(sc.get("text", "")).strip()
            at = narration.find(text, cursor) if text else -1
            if at < 0:
                issues.append(f"scene {i}'s text is not in the narration in order (it is placed by word count)")
            else:
                cursor = at + len(text)
            if not str(sc.get("visual", "")).strip():
                issues.append(f"scene {i} has no visual")
        for p in _json(story.get("pauses"), []):
            after = str((p or {}).get("after", "")).strip() if isinstance(p, dict) else ""
            if after and after not in narration:
                issues.append(f"pause after {after[:40]!r}: phrase not in the narration")
        if (narration.count('"') + narration.count("“") + narration.count("”")) % 2:
            issues.append("a quotation mark is not closed (a character's line may be read by the narrator)")
        words = len(narration.split())
        pace = float(cfg.get("voice_pace", 1.0)) or 1.0
        seconds = words / (WORDS_PER_SECOND * pace)
        target = float(cfg.get("target_seconds", 0) or 0)
        if target and not 0.8 * target <= seconds <= 1.2 * target:
            issues.append(f"the narration reads in about {seconds:.0f} s; the plan is {target:.0f} s "
                          f"(about {int(target * WORDS_PER_SECOND * pace)} words)")
        board.data["script_issues"] = issues
        return {"words": words, "estimated_seconds": round(seconds, 1), "target_seconds": target or None,
                "scenes": len(scenes), "issues": issues}


def eye(studio: Any, role: str = "vision") -> str:
    """The checker to ask: `role` (e.g. "vision_pro", a stronger model for the judgements that matter)
    when it is configured, else the everyday checker."""
    if studio is None:
        return ""
    if role != "vision" and studio.models.get(role):
        return role
    return "vision" if studio.models.get("vision") else ""


def ask_vision(studio: Any, prompt: str, image: Path | list[Path], task: str, system: str, width: int = 1536,
               role: str = "vision") -> dict[str, Any] | None:
    """One look by the vision model at one picture or several; its JSON verdict, or None when there is
    no model, budget or answer. `role`: "vision_pro" for a stronger model where it is configured."""
    role = eye(studio, role)
    pictures = [p for p in ([image] if isinstance(image, Path) else list(image or [])) if p is not None and p.exists()]
    if not role or not pictures:
        return None
    model = studio.models[role]
    try:
        usd = studio.budget.reserve(model)
    except OverBudget:
        return None
    try:
        result = studio.fal.run(model, dict(studio.args.get(role) or studio.args.get("vision", {}), prompt=prompt,
                                            system_prompt=system,
                                            image_urls=[data_uri(_small(p, width)) for p in pictures]))
    except MediaError as exc:
        studio.budget.settle(usd, task, model, ok=False)
        log.warning("%s: vision model unavailable: %s", task, str(exc)[:200])
        return None
    studio.budget.settle(usd, task, model, ok=True)
    text = str(result.get("output") or result.get("text") or result)
    match = re.search(r"\{.*\}", text, re.S)
    try:
        verdict = json.loads(match.group(0)) if match else None
    except ValueError:
        return None
    return verdict if isinstance(verdict, dict) else None


SYSTEM = ("You are the art director of Hollow Lines, a horror storytelling channel, looking at all the pictures "
          "of one video side by side before it is edited. Be specific and brief. Answer with JSON only.")


class ArtReviewAgent:
    """The pictures as a sequence: each was already checked alone; this checks them together (the
    same person looks the same, the pictures tell the story in order, the scare lands) and has the
    worst few redrawn with a note on what to correct. The critic checks every redraw.

    A long episode has about 90 pictures: on one contact sheet each is a few dozen pixels wide by
    the time the model reads it. So the review looks scene by scene (at most 12 pictures a sheet)
    and person by person (every picture of one character, beside their portrait), with the
    stronger checker where one is configured. Each picture is named by its beat number on the
    sheet and in the list, so a note points at the right picture."""

    name = "art_review"
    critical = False

    @staticmethod
    def groups(beats: list[Any], host: set[int], d: dict[str, Any], per_sheet: int = 12) -> list[tuple[str, list[Any], list[Path]]]:
        story = [b for b in beats if b.index not in host]
        out: list[tuple[str, list[Any], list[Path]]] = []
        by_scene: dict[int, list[Any]] = {}
        for b in story:
            by_scene.setdefault(b.scene, []).append(b)
        run: list[Any] = []
        for scene in sorted(by_scene):  # neighbouring short scenes share a sheet
            group = by_scene[scene]
            if run and len(run) + len(group) > per_sheet:
                out.append((f"scenes {run[0].scene}-{run[-1].scene}", run, []))
                run = []
            for k in range(0, len(group), per_sheet):
                part = group[k:k + per_sheet]
                if len(part) == per_sheet:
                    out.append((f"scene {scene}", part, []))
                else:
                    run += part
        if run:
            out.append((f"scenes {run[0].scene}-{run[-1].scene}", run, []))
        portraits: dict[str, Path] = d.get("portraits") or {}
        for name, face in portraits.items():
            theirs = [b for b in story if name in b.characters]
            if len(theirs) < 3:
                continue
            step = max(1, -(-len(theirs) // per_sheet))
            out.append((f"person {name}", theirs[::step][:per_sheet], [face]))
        return out

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        beats = d.get("beats", [])
        if not beats:
            return {"reviewed": False}
        host = set(d.get("host_beats") or ())
        fixes_allowed = max(int(cfg.get("art_review_fixes", 2)),
                            round(len(beats) * float(cfg.get("art_review_share", 0.04))))
        lines = {b.index: f"#{b.index}. {'(HOST) ' if b.index in host else ''}{str(b.text)[:140]}" for b in beats}
        cast = "; ".join(f"{c.get('name')}: {str(c.get('design', ''))[:160]}" for c in d.get("cast", [])
                         if isinstance(c, dict) and c.get("design"))
        sets = "; ".join(f"{x.get('name')}: {str(x.get('design', ''))[:200]}" for x in d.get("sets") or []
                         if isinstance(x, dict) and x.get("design"))
        looks = self.groups(beats, host, d)[: int(cfg.get("art_review_max_looks", 16))]
        flagged: dict[int, str] = {}
        scores: list[int] = []
        strongest = None
        reviewed = 0
        for n, (label, group, refs) in enumerate(looks):
            sheet = board.work / f"review_{n:02d}.jpg"
            package.contact_sheet(group, d["pictures"], sheet, columns=4, thumb=(480, 270))
            person = label.startswith("person ")
            focus = (f"Every picture here shows {label[7:]}; the second image is their close portrait. Does "
                     f"{label[7:]} look like the same person as the portrait in each picture (face, age, hair, "
                     "clothes)? List each picture where they do not." if person else
                     "Judge the pictures TOGETHER: does each named person look the same in every picture (face, "
                     "hair, age, clothes); does each place look the same in every picture set there (the same "
                     "vehicle or building, its colours and fixtures); do the pictures follow the story in order; "
                     "does a picture show something its line does not say, or contradict an earlier one; is "
                     "anything broken (warped faces or hands, text, panels); does a scare land; do they all look "
                     "like one dark horror film?")
            prompt = (f"Story: {board.story.get('title', '')}\nPeople: {cast or 'none named'}\n"
                      f"Places: {sets or 'as the lines say'}\n"
                      "The first image is a sheet of pictures from the video, in story order; each is labelled with "
                      "its number (#N). Each is on screen while its line is narrated:\n"
                      + "\n".join(lines[b.index] for b in group) + "\n\n"
                      "A threat the story never names may appear as a shadow or a shape: that is intended. Judge "
                      f"each picture against its OWN line. {focus} List only clear problems worth a redraw, worst "
                      "first. Reply exactly as JSON: "
                      '{"score": 1-10, "fix": [{"picture": N, "problem": "what is wrong and what it should show"}], '
                      '"strongest": N}  (N is the number on the sheet)')
            verdict = ask_vision(board.studio, prompt, [sheet, *refs], "art_review", SYSTEM, role="vision_pro")
            if verdict is None:
                continue
            reviewed += 1
            try:
                scores.append(max(1, min(10, int(verdict.get("score")))))
            except (TypeError, ValueError):
                pass  # no readable score: only its list of fixes counts
            if strongest is None and not person:
                strongest = verdict.get("strongest")
            known = {b.index for b in group}
            for f in verdict.get("fix", []) if isinstance(verdict.get("fix"), list) else []:
                try:
                    k, problem = int(f.get("picture")), str(f.get("problem", "")).strip()
                except (AttributeError, TypeError, ValueError):
                    continue
                if k in known and problem and k not in flagged:
                    flagged[k] = (f"{label}: " if person else "") + problem
        if not reviewed:
            return {"reviewed": False}
        redrawn = []
        repaint = d.get("repaint")
        by_index = {b.index: b for b in beats}
        for k, problem in list(flagged.items())[:fixes_allowed] if repaint is not None else []:
            try:
                path = repaint(by_index[k], problem)
            except Exception as exc:  # a failed redraw keeps the first picture
                log.warning("Redraw of picture %d failed: %s", k, str(exc)[:200])
                path = None
            if path is not None:
                d["pictures"][k] = path
                redrawn.append({"picture": k, "problem": problem[:200]})
        score = min(scores) if scores else None  # the weakest stretch is what a viewer remembers
        return {"reviewed": True, "looks": reviewed, "score": score, "scores": scores,
                "flagged": [{"picture": k, "problem": p[:200]} for k, p in flagged.items()],
                "redrawn": redrawn, "strongest_picture": strongest}


THUMB_SYSTEM = "You check YouTube thumbnails for a horror channel. Answer with JSON only."


def thumbnail_ok(studio: Any, image: Path, words: str) -> tuple[bool, str]:
    """Do the thumbnail's words read exactly as planned, large and clear, with nothing else written?"""
    want = f'the words "{words.upper()}" spelled exactly, large and readable at phone size' if words else "no text"
    verdict = ask_vision(studio, f"This thumbnail should have {want}, on a dark horror image with one clear focal "
                                 "point. Fail it ONLY if: the planned words are missing, misspelled, cut off or hard "
                                 "to read; other LARGE words compete with them (a small sign or label that belongs "
                                 "in the scene is fine); there is any logo, play button, channel name or brand-like "
                                 "text (a fake YouTube logo fails it); or a face or hand is clearly warped. Reply exactly as JSON: "
                                 '{"ok": true or false, "reason": "short reason"}', image, "thumbnail_check",
                         THUMB_SYSTEM, width=1024)
    if verdict is None or "ok" not in verdict:
        return True, "not checked"
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))[:200]


SHEET_SYSTEM = ("You check reference sheets for a horror film told in still pictures. Every picture of the film "
                "is painted from these sheets, so a wrong sheet makes every picture wrong. Answer with JSON only.")


def sheet_ok(studio: Any, image: Path, kind: str, name: str, design: str) -> tuple[bool, str]:
    """Does a reference sheet show what its design says? `kind`: "character", "portrait" (a
    close portrait painted from the character's sheet) or "location"."""
    # Only what would make every picture painted from the sheet wrong: run 104 repainted sheets for a
    # single bed instead of a double one, a blue tint and brown eyes ($0.43 for nothing a viewer sees).
    age = ("; a child who clearly looks older or younger than the age the design gives (a four-year-old "
           "painted as a child of eight is wrong)")
    if kind == "portrait":
        rules = ("it is not one single close portrait of one person; it is clearly a different person from the "
                 "design (a different age or sex, a different hair colour, clothes of a different kind); a face "
                 f"that the design keeps in shadow is shown lit in full{age}; other people are in it")
    elif kind == "character":
        rules = ("the views do not show one and the same person; the person is clearly someone else than the "
                 "design (a different age or sex, a different hair colour, clothes of a different kind); a face "
                 f"that the design keeps hidden is shown{age}; other people are in it")
    else:
        rules = ("it is clearly a different kind of place (a different kind of vehicle, building or room); its "
                 "views show two different places; people are in it")
    details = ("furniture sizes, colour tints, eye colour, small objects and accessories, how many views, "
               "the exact light") if kind != "portrait" else "eye colour, small accessories, the exact light"
    verdict = ask_vision(studio, f"This should be a {kind} reference sheet of {name}: {design[:500]}\n"
                                 f"Fail it ONLY if one of these is clearly true: {rules}; it contains written "
                                 "words or labels; it is bright and cheerful; anything is clearly broken or "
                                 f"melted. Never fail it for details: {details}. When you fail it, say in one "
                                 'sentence what is wrong and what it should show. Reply exactly as JSON: '
                                 '{"ok": true or false, "reason": "short reason"}', image, "reference_check",
                         SHEET_SYSTEM, width=1280)
    if verdict is None or "ok" not in verdict:
        return True, "not checked"
    return bool(verdict.get("ok")), str(verdict.get("reason", ""))[:200]
