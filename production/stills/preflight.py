"""The preflight agent: before anything is bought, the paid services this video needs answer.

A run used to discover an empty fal balance only when the narrator failed, then finish the
video in the free voice (two narrators in one episode). Now, first of all:
- the narrator reads one short line with the paid voice (a fraction of a cent); if it cannot,
  the run stops before a single picture is paid for (`require_paid_voice`, on by default);
- the checker answers one word, so a run never paints a whole episode with no eye on it;
- the episode's likely cost is worked out from the script and compared with the cap.
Without any paid key nothing is asked: the free path is unchanged.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

from production.media import MediaError

from ..paid import OverBudget
from .board import Board, _json
from .checks import WORDS_PER_SECOND

log = logging.getLogger(__name__)

LINE = "The lantern is lit."


class PreflightFailed(RuntimeError):
    pass


def _dot() -> str:
    """A tiny dark picture, so a vision model that needs an image gets one."""
    from PIL import Image

    data = io.BytesIO()
    Image.new("RGB", (32, 32), (12, 12, 12)).save(data, "PNG")
    return "data:image/png;base64," + base64.b64encode(data.getvalue()).decode()


def estimate(story: dict[str, Any], cfg: dict[str, Any], prices: Any) -> dict[str, Any]:
    """The episode's likely cost from its script: pictures (beats, references, retries, second
    takes, thumbnails), narration and checks. An estimate for the cap, not an invoice."""
    narration = str(story.get("narration") or "")
    words = len(narration.split())
    pace = float(cfg.get("voice_pace", 1.0)) or 1.0
    seconds = words / (WORDS_PER_SECOND * pace)
    beat = (float(cfg.get("beat_min_seconds", 5.0)) + float(cfg.get("beat_max_seconds", 11.0))) / 2
    beats = max(1, round(seconds / max(beat, 1.0)))
    cast = [c for c in _json(story.get("cast"), []) if isinstance(c, dict) and c.get("name")]
    people = len(cast) + 1  # and the host, unless his references are kept
    references = 1 + int(cfg.get("location_sheets", 3)) + people * (2 if cfg.get("portraits", True) else 1)
    takes = float((cfg.get("candidates") or {}).get("share", 0.0)) if (cfg.get("candidates") or {}).get(
        "enabled", False) else 0.0
    pictures = beats * (1 + takes) + references + int(cfg.get("thumbnails", 3))
    models = cfg.get("models", {})
    picture = prices.price(models.get("shot", ""))
    voice = prices.price(models.get("voice", ""), len(narration) / 1000.0)
    checks = (beats * 1.3 + references) * prices.price(models.get("vision", ""))
    retries = float(cfg.get("retry_share", 0.15)) * float(cfg.get("max_usd", 0) or 0)
    usd = pictures * picture + voice + checks
    return {"words": words, "seconds": round(seconds), "beats": beats, "pictures": round(pictures),
            "usd": round(usd, 2), "usd_with_retries": round(usd + retries, 2)}


class PreflightAgent:
    name = "preflight"
    critical = True  # a run that cannot speak in its own voice stops here, before money is spent

    def run(self, board: Board) -> dict[str, Any]:
        cfg, studio = board.cfg, board.studio
        if studio is None:
            return {"paid": False}
        report: dict[str, Any] = {"paid": True}
        guess = estimate(board.story, cfg, studio.budget)
        report["estimate"] = guess
        cap = float(cfg.get("max_usd", 0) or 0)
        if cap and guess["usd"] > cap:
            report["warning"] = (f"the episode should cost about ${guess['usd']:.2f}, over the ${cap:.2f} cap: "
                                 "the last pictures would borrow a neighbour's")
        board.data["preflight"] = report
        if studio.models.get("voice") and cfg.get("preflight_voice", True):
            host = str((cfg.get("host_voice") or {}).get("elevenlabs") or "Brian")
            model = studio.models["voice"]
            usd = 0.0
            try:  # asked every run, never from a cache: the balance may have run out since
                usd = studio.budget.reserve(model, len(LINE) / 1000.0)
                got = studio.fal.run(model, dict(studio.args.get("voice", {}), text=LINE, voice=host))
                if not isinstance(got, dict) or not any(got.get(k) for k in ("audio", "audio_url", "audio_file", "output")):
                    raise MediaError(f"no audio in the answer: {str(got)[:200]}")
                studio.budget.settle(usd, "preflight", model, ok=True, units=len(LINE) / 1000.0)
                report["voice"] = "ok"
            except (MediaError, OverBudget) as exc:
                if usd:
                    studio.budget.settle(usd, "preflight", model, ok=False)
                report["voice"] = f"failed: {str(exc)[:200]}"
                if cfg.get("require_paid_voice", True):
                    raise PreflightFailed(
                        f"the narrator ({studio.models['voice']}) could not read a test line: {str(exc)[:300]}. "
                        "Top up fal.ai (fal.ai/dashboard/billing), or set stills.require_paid_voice=false to "
                        "make the video in the free Kokoro voice. Nothing was paid for.") from exc
        if studio.models.get("vision"):
            model = studio.models["vision"]
            usd = 0.0
            try:
                usd = studio.budget.reserve(model)
                got = studio.fal.run(model, dict(studio.args.get("vision", {}), prompt="Reply with the single word OK.",
                                                 image_urls=[_dot()]))
                studio.budget.settle(usd, "preflight", model, ok=True)
                report["checker"] = "ok" if str(got.get("output", "")).strip() else "no answer"
            except OverBudget:
                report["checker"] = "no budget"
            except MediaError as exc:
                if usd:
                    studio.budget.settle(usd, "preflight", model, ok=False)
                report["checker"] = f"failed: {str(exc)[:200]}"
                google_only = str(model).startswith("vertex:") and not (cfg.get("vertex") or {}).get("fallback_to_fal", False)
                if google_only and cfg.get("require_google", True):
                    # Pictures and checks are made on Google only: a refused test means every picture
                    # would fail, so nothing more is bought (run 118: "Request rejected" on every request).
                    raise PreflightFailed(
                        f"Google Vertex AI refused the test check ({model}): {' '.join(str(exc).split())[:300]}. "
                        "Pictures are made on Google only, so the run stops here. Fix the Google Cloud project "
                        "(billing, the Vertex AI API, the key), or set stills.vertex.fallback_to_fal=true.") from exc
                log.warning("The checker did not answer: pictures will be used unchecked (%s)", str(exc)[:200])
        return report
