"""Depth parallax: a still that moves like a camera in a room, not a photo under a magnifier.

A slow push or pan moves a picture as one flat sheet. Here an open depth model (Depth Anything V2
Small, run with onnxruntime on the runner's CPU, free) estimates how far away each part of a
picture is; the nearest part becomes a foreground layer (its edges feathered), the rest a
background plate with the foreground's place filled from its surroundings. In the edit the
foreground moves a little more than the background, so the camera seems to move through the
scene. Dark horror frames hide the filled seams well.

Only wide and medium framings get it (a close-up has no depth to show), and only when it would
read: a foreground covering between `min_share` and `max_share` of the frame. Off by default
(`parallax.enabled`): turn it on, watch one episode, then keep it.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image, ImageFilter

from .board import Board

log = logging.getLogger(__name__)

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class DepthModel:
    """Depth Anything V2 Small (ONNX), downloaded once into the Hugging Face cache."""

    def __init__(self, repo: str, filename: str, session_factory: Callable[[str], Any] | None = None, side: int = 518):
        self.repo, self.filename, self.side = repo, filename, side
        self._factory = session_factory
        self._session = None

    def session(self) -> Any:
        if self._session is None:
            if self._factory is not None:
                self._session = self._factory("")
            else:
                import onnxruntime
                from huggingface_hub import hf_hub_download

                path = hf_hub_download(self.repo, self.filename)
                self._session = onnxruntime.InferenceSession(path, providers=["CPUExecutionProvider"])
        return self._session

    def __call__(self, picture: Path) -> np.ndarray:
        """Nearness from 0 (far) to 1 (near), at the picture's size."""
        with Image.open(picture) as im:
            rgb = im.convert("RGB")
            size = rgb.size
            small = np.asarray(rgb.resize((self.side, self.side), Image.BICUBIC), dtype=np.float32) / 255.0
        x = ((small - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        session = self.session()
        name = session.get_inputs()[0].name
        out = np.asarray(session.run(None, {name: x})[0], dtype=np.float32).squeeze()
        lo, hi = float(out.min()), float(out.max())
        near = (out - lo) / (hi - lo) if hi > lo else np.zeros_like(out)
        return np.asarray(Image.fromarray((near * 255).astype("uint8")).resize(size, Image.BICUBIC),
                          dtype=np.float32) / 255.0


def layers(picture: Path, near: np.ndarray, out_dir: Path, share: float = 0.3, feather: float = 0.012,
           min_share: float = 0.06, max_share: float = 0.55) -> tuple[Path, Path] | None:
    """(background plate, foreground RGBA) from a picture and its nearness map, or None when the
    picture has no clear foreground. `share`: the nearest part of the frame taken as the foreground."""
    with Image.open(picture) as im:
        rgb = im.convert("RGB")
    w, h = rgb.size
    cut = float(np.quantile(near, 1.0 - share))
    # the nearest `share` of the frame; when most of the map is one value (a flat far wall), only what is nearer
    hard = ((near >= cut) if cut > float(near.min()) + 1e-6 else (near > cut)).astype(np.float32)
    covered = float(hard.mean())
    if not min_share <= covered <= max_share:
        return None
    radius = max(1.0, feather * w)
    soft = np.asarray(Image.fromarray((hard * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(radius)),
                      dtype=np.float32) / 255.0
    # the background: where the foreground was, filled from its surroundings (normalised convolution)
    grown = np.asarray(Image.fromarray((hard * 255).astype("uint8")).filter(ImageFilter.MaxFilter(9))
                       .filter(ImageFilter.GaussianBlur(radius)), dtype=np.float32) / 255.0
    keep = 1.0 - np.clip(grown * 1.5, 0, 1)
    pixels = np.asarray(rgb, dtype=np.float32)
    blur = max(4.0, 0.03 * w)
    num = np.stack([np.asarray(Image.fromarray(np.clip(pixels[..., c] * keep, 0, 255).astype("uint8"))
                               .filter(ImageFilter.GaussianBlur(blur)), dtype=np.float32) for c in range(3)], -1)
    den = np.asarray(Image.fromarray((keep * 255).astype("uint8")).filter(ImageFilter.GaussianBlur(blur)),
                     dtype=np.float32)[..., None] / 255.0
    fill = num / np.maximum(den, 1e-3)
    plate = pixels * keep[..., None] + fill * (1.0 - keep[..., None])
    out_dir.mkdir(parents=True, exist_ok=True)
    bg, fg = out_dir / f"{picture.stem}_bg.png", out_dir / f"{picture.stem}_fg.png"
    Image.fromarray(np.clip(plate, 0, 255).astype("uint8")).save(bg)
    Image.fromarray(np.dstack([pixels, soft * 255]).clip(0, 255).astype("uint8"), "RGBA").save(fg)
    return bg, fg


class DepthAgent:
    name = "depth"
    critical = False  # without it, every picture moves as one flat sheet, as before

    def run(self, board: Board) -> dict[str, Any]:
        d, cfg = board.data, board.cfg
        settings = cfg.get("parallax") or {}
        if not settings.get("enabled", False) or str(cfg.get("motion", "drift")) != "drift":
            return {"enabled": False}
        framings = set(settings.get("framings") or ("wide", "medium"))
        host = set(d.get("host_beats") or ())
        clips = set(d.get("clips") or {})
        chosen = [b for b in d["beats"] if b.framing in framings and b.index not in host and b.index not in clips
                  and not b.scare and b.view == "film" and b.index in d.get("pictures", {})]
        model = d.get("depth_model") or DepthModel(str(settings.get("repo", "onnx-community/depth-anything-v2-small")),
                                                   str(settings.get("file", "onnx/model.onnx")))
        made: dict[int, tuple[Path, Path]] = {}
        skipped = 0
        for b in chosen:
            try:
                near = model(d["pictures"][b.index])
            except Exception as exc:  # noqa: BLE001 - no model, no download: no parallax, nothing else changes
                log.warning("No depth model (%s): no parallax", str(exc)[:200])
                return {"enabled": True, "made": 0, "error": str(exc)[:200]}
            got = layers(d["pictures"][b.index], near, board.work / "layers", float(settings.get("share", 0.3)))
            if got is None:
                skipped += 1
                continue
            made[b.index] = got
        d["layers"] = made
        return {"enabled": True, "made": len(made), "no_clear_foreground": skipped, "beats": sorted(made)}
