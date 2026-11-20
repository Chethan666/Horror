"""One approved story in, a finished stills video out, made by the agent team."""

from __future__ import annotations

import json
import logging
import shutil
import time
from pathlib import Path
from typing import Any

from ..paid import Budget, Studio
from ..vertex import traffic
from ..replay import Replay, ReplayRouter, cache_root
from .art import ArtDirectorAgent, IllustratorAgent
from .board import Board, Showrunner, _json
from .checks import ArtReviewAgent, ScriptCheckAgent
from .grade import ColoristAgent
from .continuity import ContinuityAgent
from .depth import DepthAgent
from .motion import MotionAgent
from .preflight import PreflightAgent
from .screening import ScreeningAgent
from .staging import StagingAgent
from .post import EditAgent, EditPlanAgent, PackagerAgent, QAAgent, SoundAgent
from .storyboard import StoryboardAgent
from .voices import CastingAgent, VoiceAgent

log = logging.getLogger(__name__)


def team() -> list[Any]:
    """The agents, in the order the showrunner runs them."""
    return [PreflightAgent(), ScriptCheckAgent(), CastingAgent(), VoiceAgent(), StoryboardAgent(), ContinuityAgent(),
            ArtDirectorAgent(), StagingAgent(), IllustratorAgent(), ArtReviewAgent(), ColoristAgent(), MotionAgent(), DepthAgent(),
            EditPlanAgent(), SoundAgent(), EditAgent(), ScreeningAgent(), PackagerAgent(),
            QAAgent()]


def produce(story: dict[str, Any], out_dir: Path, kokoro: Any, router: Any, cfg: dict[str, Any],
            fal: Any = None, conn: Any = None, keep_work: bool = False,
            redo: dict[int, str] | None = None) -> dict[str, Any]:
    """`redo`: beats a person asked to have repainted after watching ({beat: note}); with the
    cache of the run they watched (MEDIA_CACHE_DIR), only those pictures are paid for again."""
    started = time.time()
    out_dir = out_dir.resolve()  # FFmpeg runs inside the work folder: every path must be absolute
    work = out_dir / "work"
    work.mkdir(parents=True, exist_ok=True)
    root = cache_root(cfg, work)  # kept between runs when MEDIA_CACHE_DIR is set: a rerun pays nothing twice
    budget = Budget(float(cfg.get("max_usd", 0.0)), cfg.get("prices", {}), conn)
    if cfg.get("genre"):  # an episode that is not horror: every instruction re-toned (production/stills/genre.py)
        from .genre import GenreClient, GenreRouter
        fal = GenreClient(fal, cfg["genre"]) if fal is not None else None
        router = GenreRouter(router, cfg["genre"]) if router is not None else None
    answers = Replay(fal, root / "answers") if fal is not None else None
    studio = Studio(answers, budget, root / "media", cfg) if answers is not None else None
    if studio is not None:
        studio.stop_on_empty = False  # the preflight agent stops a run whose narrator cannot speak
    planner = ReplayRouter(router, root / "plans") if router is not None else None
    board = Board(story=story, out=out_dir, work=work, cfg=cfg, router=planner, studio=studio, budget=budget,
                  kokoro=kokoro, conn=conn)
    board.data.update(bible=_json(story.get("bible"), {}), packaging=_json(story.get("packaging"), {}),
                      redo=dict(redo or {}))
    reports = Showrunner(team()).run(board)
    d = board.data
    summary = {
        "mode": "stills", "video": str(d.get("video", "")), "seconds": round(d.get("total", 0.0), 1),
        "story_seconds": round(d.get("story_seconds", 0.0), 1), "beats": len(d.get("beats", [])),
        "point_of_view": d.get("pov"), "voices": d.get("casting"),
        "usd": round(budget.spent, 2), "usd_by_task": {k: round(v, 2) for k, v in sorted(budget.by_task.items())},
        "fal_balance_ran_out": bool(studio and studio.balance_ran_out),
        # pictures and checks Vertex could not make, sent to fal instead (at fal's price)
        "vertex_fallbacks": int(getattr(fal, "fell_back", 0) or 0) if fal is not None else 0,
        "google": traffic(studio.fal if studio is not None else fal),
        "answers_reused": {"planner": planner.replayed if planner else 0,
                           "paid_models": answers.replayed if answers else 0},
        "beat_times": d.get("beat_times", []),
        "qa": reports.get("qa", {}), "agents": reports,
        "review": "pending", "render_minutes": round((time.time() - started) / 60, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))
    if not keep_work:
        shutil.rmtree(work, ignore_errors=True)
    log.info("Made stills video: %s", {k: v for k, v in summary.items() if k != "agents"})
    return summary
