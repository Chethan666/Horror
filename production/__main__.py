"""Command line for pipeline 2.

    python -m production --config config.aws-open.toml produce            # next approved long story
    python -m production --config config.aws-open.toml produce --story 12 # a chosen story, status unchanged
    python -m production --config config.aws-open.toml produce --script production/samples/one_minute.json
    python -m production --config config.aws-open.toml check              # what each paid model expects (no spend)
    python -m production --config config.aws-open.toml produce --script production/samples/one_minute.json \
        --bakeoff all                                                      # pictures only, by each setup
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from story_factory import db
from story_factory.__main__ import apply_overrides
from story_factory.config import load_config
from story_factory.router import Router

from . import voice

log = logging.getLogger("production")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="production", description="Hollow Lines video pipeline")
    parser.add_argument("--config", help="TOML config file")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="list what each paid model on fal.ai expects (no spend)")
    h = sub.add_parser("host", help="paint candidates for the Storyteller's fixed sheet and portrait (paid, once)")
    h.add_argument("--out", default="out/host_refs")
    h.add_argument("--candidates", type=int, default=2)
    h.add_argument("--max-usd", type=float, default=1.5)
    h.add_argument("--loop", action="store_true", help="instead: candidates for the host's looping clip (Veo on "
                   "Vertex, from his fixed portrait; a few dollars once)")
    cp = sub.add_parser("compile", help="join finished episodes into one long compilation video (free)")
    cp.add_argument("--dirs", nargs="+", required=True, help="episode folders (video.mp4 and metadata.json each)")
    cp.add_argument("--out", default="out/compilation")
    cp.add_argument("--title", default="")
    th = sub.add_parser("theme", help="make candidates for the channel's theme with Lyria (paid, about $0.06 each)")
    th.add_argument("--out", default="out/theme")
    th.add_argument("--candidates", type=int, default=3)
    a = sub.add_parser("approve", help="a person has watched the video: approve it for publishing (or reject it)")
    a.add_argument("--story", type=int, required=True)
    a.add_argument("--reject", metavar="REASON", help="reject instead, with the reason")
    a.add_argument("--note", default="", help="what the reviewer thought (kept with the approval)")
    u = sub.add_parser("publish", help="upload an approved video to YouTube (private or scheduled)")
    u.add_argument("--story", type=int, required=True)
    u.add_argument("--dir", required=True, help="the video's folder (video.mp4, metadata.json, ...)")
    u.add_argument("--privacy", choices=("private", "unlisted"), default="private")
    u.add_argument("--at", help="schedule: an RFC 3339 time, e.g. 2026-11-20T17:00:00Z")
    sub.add_parser("stats", help="read every published video's YouTube stats into the performance table")
    p = sub.add_parser("produce", help="make the video for one approved story")
    p.add_argument("--story", type=int, help="a chosen story (its status is left unchanged)")
    p.add_argument("--script", help="a story JSON file instead of the database (stills mode; nothing is saved "
                                    "to the database); its optional \"stills\" object overrides settings")
    p.add_argument("--out", default="out", help="output folder")
    p.add_argument("--voice", choices=("kokoro", "silent"), default="kokoro",
                   help="silent: a placeholder tone, for testing the rest of the pipeline")
    p.add_argument("--no-direction-model", action="store_true",
                   help="no planner model calls (storyboard, casting and sound from the scene plan)")
    p.add_argument("--keep-work", action="store_true", help="keep the work folder (pictures, segments)")
    p.add_argument("--max-usd", type=float, help="spending cap for this video (default stills.max_usd)")
    p.add_argument("--bakeoff", metavar="SETUPS", help="no video: paint every picture with each setup in "
                   "stills.bakeoff.setups (comma-separated, or 'all') from the same references, side by side")
    p.add_argument("--redo", default="", metavar="BEATS",
                   help="repaint these beats with a note each, e.g. '12,31:no_text_on_the_wall' (with the "
                        "earlier run's MEDIA_CACHE_DIR, nothing else is paid for again)")
    c = sub.add_parser("cache", help="pack a media cache to keep it (pictures as JPEGs, the rest unchanged)")
    c.add_argument("action", choices=("pack",))
    c.add_argument("--dir", default=os.environ.get("MEDIA_CACHE_DIR", ""), help="the cache (MEDIA_CACHE_DIR)")
    c.add_argument("--out", required=True)
    lib = sub.add_parser("library", help="the sound library: build it from free sources, sync, or show it")
    lib.add_argument("action", choices=("build", "sync", "show"))
    lib.add_argument("--sources", default="sounds/sources.json")
    lib.add_argument("--out", default="build/library", help="where new files and library.json are written")
    lib.add_argument("--existing", help="the current library.json (default: downloaded from the release)")
    lib.add_argument("--only", default="", help="comma-separated source types to run (e.g. incompetech,opengameart)")
    lib.add_argument("--generate-music", action="store_true", help="also make the ACE-Step cues (paid, capped)")
    lib.add_argument("--max-usd", type=float, default=1.0, help="cap for --generate-music")
    lib.add_argument("--runs", nargs="*", default=[], help="sync: video output folders (each with sounds_used.json)")
    lib.add_argument("--fresh", action="store_true", help="build: ignore the current library and start over")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    cfg = apply_overrides(load_config(args.config), args.set)
    if args.command == "check":
        stills = cfg.get("stills", {})
        found = check(stills.get("models", {}), stills.get("model_args", {}))
        return check_vertex(stills) or found
    if args.command == "library":
        return library(args, cfg)
    if args.command == "cache":
        from .replay import pack
        print(json.dumps(pack(Path(args.dir), Path(args.out)), indent=2) if args.dir else "No cache folder given")
        return 0
    if args.command == "host" and args.loop:
        from .stills import host
        stills, client = media_client(cfg.get("stills", {}))
        if client is None:
            print("The host's loop is made by Veo on Vertex AI: set GEMINI_PLATFORM_API_KEY", file=sys.stderr)
            return 1
        model = str(stills.get("models", {}).get("clip") or "vertex:veo-3.1-fast-generate-001")
        report = host.make_loop(stills, client, Path(args.out), args.candidates, model)
        print(json.dumps(report, indent=2))
        return 0 if report["candidates"] else 1
    if args.command == "host":
        from .stills import host
        stills, client = media_client(cfg.get("stills", {}))
        report = host.make(stills, client, Path(args.out), args.candidates, args.max_usd)
        print(json.dumps(report, indent=2))
        return 0 if any("portrait" in c for c in report["candidates"]) else 1
    if args.command == "compile":
        from .compile import compile_episodes
        print(json.dumps(compile_episodes([Path(d) for d in args.dirs], Path(args.out), args.title), indent=2))
        return 0
    if args.command == "theme":
        from .stills import theme
        stills, client = media_client(cfg.get("stills", {}))
        if client is None:
            print("The theme is made by Lyria on Vertex AI: set GEMINI_PLATFORM_API_KEY", file=sys.stderr)
            return 1
        report = theme.make(stills, client, Path(args.out), args.candidates)
        print(json.dumps(report, indent=2))
        return 0 if report["candidates"] else 1
    conn = db.connect(cfg["database"])
    if args.command in ("approve", "publish", "stats"):
        return channel(args, conn)
    video_cfg = cfg.get("video", {})

    script_settings: dict = {}
    if args.command == "produce" and args.bakeoff and not (args.script or args.story):
        print("A bake-off needs --script or --story (it never takes a story from the queue)", file=sys.stderr)
        return 1
    if args.script:
        story = json.loads(Path(args.script).read_text())
        script_settings = story.pop("stills", {}) or {}
        story = {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in story.items()}
        row, claimed = story, False
    elif args.story:
        row = db.get_story(conn, args.story)
        if row is None:
            print(f"No story #{args.story}", file=sys.stderr)
            return 1
        claimed = False
    else:
        row = db.claim_next(conn, "long", worker=f"production-{os.environ.get('GITHUB_RUN_ID', 'local')}")
        if row is None:
            print("No approved long story waiting", file=sys.stderr)
            return 1
        claimed = True
    story = dict(row)
    print(f"Producing story #{story['id']}: {story['title']}", flush=True)

    narrator: voice.Voice
    if args.voice == "silent":
        narrator = voice.SilentVoice()
    else:
        narrator = voice.KokoroVoice(video_cfg.get("voice", "am_michael"), video_cfg.get("lang", "a"),
                                     float(video_cfg.get("speed", 0.92)))
    router = None if args.no_direction_model else Router(cfg, conn)
    out_dir = Path(args.out) / f"story-{story['id']}"
    stills_cfg = dict(cfg.get("stills", {}), **script_settings)
    if args.max_usd is not None:
        stills_cfg["max_usd"] = args.max_usd
    try:
        from .stills import pipeline as stills
        stills_cfg, fal = media_client(stills_cfg)
        if args.bakeoff:
            from .stills import bakeoff
            stills_cfg["max_usd"] = float(args.max_usd if args.max_usd is not None
                                          else stills_cfg.get("bakeoff", {}).get("max_usd", 7.0))
            chosen = None if args.bakeoff == "all" else [x.strip() for x in args.bakeoff.split(",") if x.strip()]
            summary = bakeoff.run(story, out_dir, narrator, router, stills_cfg, fal, conn, chosen)
        else:
            from .replay import parse_redo
            summary = stills.produce(story, out_dir, narrator, router, stills_cfg, fal, conn, args.keep_work,
                                     redo=parse_redo(args.redo))
    except Exception as exc:  # report, and hand a claimed story back to the queue
        log.exception("Production failed")
        info = {"error": str(exc)[:500], "run": os.environ.get("GITHUB_RUN_ID")}
        if args.script:
            print(f"[failed] script {args.script}: {exc}", file=sys.stderr)
            return 2
        db.update_story(conn, story["id"], production=info, **({"status": "approved"} if claimed else {}))
        print(f"[failed] story #{story['id']}: {exc}", file=sys.stderr)
        return 2
    summary.update(run=os.environ.get("GITHUB_RUN_ID"), run_number=os.environ.get("GITHUB_RUN_NUMBER"),
                   narrator_fallback=args.voice)
    if not args.script and not args.bakeoff:
        db.update_story(conn, story["id"], production=summary, **({"status": "produced"} if claimed else {}))
    print(json.dumps(summary, indent=2, default=str))
    return 0


def channel(args: argparse.Namespace, conn) -> int:
    """The human review gate, publishing, and stats."""
    from . import youtube

    if args.command == "stats":
        from . import retention

        rows = conn.execute("SELECT id, title, video_id, production FROM stories "
                            "WHERE video_id IS NOT NULL AND video_id != ''").fetchall()
        if not rows:
            print("No published videos yet")
            return 0
        stats = youtube.YouTube().stats([r["video_id"] for r in rows])
        notes: list[str] = []
        for r in rows:
            s = stats.get(r["video_id"])
            if not s:
                continue
            curve = s.pop("retention", None)
            db.record_performance(conn, r["id"], **s)
            print(f"story #{r['id']} {r['video_id']}: {s}")
            if curve:  # where viewers left, against the beats on screen (kept at production)
                made = json.loads(r["production"] or "{}") if isinstance(r["production"], str) else {}
                found, measured = retention.lessons(str(r["title"]), curve, float(made.get("seconds") or 0),
                                                    made.get("beat_times") or [])
                made["retention"] = dict(measured, measured_at=db.now(), curve=curve)
                db.update_story(conn, r["id"], production=made)
                notes += found
                for line in found:
                    print(f"  {line}")
        if notes:  # the writers see them with every new pitch, beside the judges' own lessons
            kept = [i for i in db.latest_learnings(conn) if not str(i).startswith(retention.PREFIX)]
            db.save_learnings(conn, kept + notes[:12])
        return 0
    row = db.get_story(conn, args.story)
    if row is None:
        print(f"No story #{args.story}", file=sys.stderr)
        return 1
    production = json.loads(row["production"] or "{}") if isinstance(row["production"], str) else {}
    if args.command == "approve":
        # The person's review is part of the record: who watched it, when, and what they thought (YouTube's
        # Partner Program asks for human creative input; a note per video shows it was given).
        production["review"] = "rejected" if args.reject else "approved"
        production["reviewed_at"] = db.now()
        note = args.reject or getattr(args, "note", "")
        if note:
            production["review_note"] = note
        db.update_story(conn, args.story, production=production)
        print(f"Story #{args.story}: {production['review']}")
        return 0
    if production.get("review") != "approved":
        print(f"Story #{args.story} is not approved yet (review: {production.get('review', 'none')}). "
              "Watch preview_60s.mp4 and the contact sheet, then run mode video-approve.", file=sys.stderr)
        return 1
    done = youtube.publish(youtube.YouTube(), Path(args.dir), args.privacy, args.at)
    production["youtube"] = done
    db.update_story(conn, args.story, production=production, video_id=done["video_id"], status="published",
                    published_at=db.now())
    print(json.dumps(done, indent=2))
    return 0


ALTERNATES = [  # listed alongside the configured models, to compare
    "fal-ai/nano-banana-pro", "fal-ai/nano-banana-pro/edit", "fal-ai/nano-banana-2", "fal-ai/nano-banana-2/edit",
    "fal-ai/bytedance/seedream/v4.5/text-to-image", "fal-ai/bytedance/seedream/v4.5/edit",
    "fal-ai/elevenlabs/tts/eleven-v3", "fal-ai/elevenlabs/sound-effects/v2", "fal-ai/ace-step-1.5",
    "fal-ai/elevenlabs/music",
    # image-to-video on fal, for the motion stage if Veo on Vertex is not used
    "fal-ai/veo3.1/fast/image-to-video", "fal-ai/kling-video/v2.5-turbo/pro/image-to-video",
]


def library(args: argparse.Namespace, cfg: dict) -> int:
    """Build, sync or show the sound library (production/library.py)."""
    from . import library_build
    from .library import Library, fetch_release_file, read_catalogue, release_base

    lib_cfg = dict(cfg.get("stills", {}).get("sound_library") or {})
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    existing = Path(args.existing) if args.existing else None
    if existing is None and not getattr(args, "fresh", False):
        base = release_base(lib_cfg)
        got = out.parent / "library-current.json"
        try:
            if not base:
                raise ValueError("no repository known (set GITHUB_REPOSITORY or stills.sound_library.repo)")
            got.write_bytes(fetch_release_file(f"{base}/library.json"))
            existing = got
        except Exception as exc:  # noqa: BLE001 - the first build starts an empty library
            print(f"No current library ({str(exc)[:120]}); starting a new one")
    if args.action == "show":
        sounds = read_catalogue(existing) if existing else []
        lib = Library(sounds, out, out)
        print(json.dumps({"total": len(lib.sounds), "by_kind": lib.summary()}, indent=2))
        for s in sorted(lib.sounds, key=lambda s: (s.kind, s.title))[:400]:
            print(f"{s.kind:8} {s.seconds:6.1f}s  {','.join(s.moods):14} {s.license:10} uses {s.uses:3}  {s.title}")
        return 0
    if args.action == "sync":
        print(json.dumps(library_build.sync([Path(r) for r in args.runs], existing, out), indent=2))
        return 0
    fal = None
    if args.generate_music and os.environ.get("FAL_KEY"):
        from .media import Fal
        fal = Fal()
    only = {x.strip() for x in args.only.split(",") if x.strip()} or None
    summary = library_build.build(Path(args.sources), out, existing, only, fal, args.max_usd)
    print(json.dumps(summary, indent=2))
    return 0


def media_client(stills: dict) -> tuple[dict, object | None]:
    """The stills config and the client for paid media: fal (FAL_KEY), with the Nano Banana 2.1
    roles, the checkers and the score on Google Cloud Vertex AI when a Google key is set, and the
    host's own voice on ElevenLabs when ELEVENLABS_API_KEY and `host_voice.elevenlabs_id` are set."""
    from production import elevenlabs, vertex
    from production.media import Fal, MediaError

    fal = Fal() if os.environ.get("FAL_KEY") else None
    stills = dict(stills)
    models = dict(stills.get("models", {}))
    others = {}
    if str((stills.get("host_voice") or {}).get("elevenlabs_id") or "") and elevenlabs.available():
        others[elevenlabs.PREFIX] = elevenlabs.ElevenLabs()
        print("The host's voice: the channel's own, on ElevenLabs", flush=True)
    else:
        models.pop("host_voice", None)  # no key or no voice of its own: the stock voice on fal
    stills["models"] = models
    client = None
    if vertex.available() and (stills.get("vertex") or {}).get("enabled", True):
        try:
            client = vertex.client()
        except MediaError as exc:
            log.warning("Vertex AI not used: %s", exc)
    if client is not None:
        moved, fallback = vertex.use_vertex(stills)
        on_google = sorted({str(m) for m in moved.get("models", {}).values() if str(m).startswith(vertex.PREFIX)})
        print(f"Paid pictures and checks on Google ({client.project}, {client.location}): {on_google}; "
              f"fal as their fallback: {'yes' if fal and fallback else 'no'}", flush=True)
        settings = stills.get("vertex") or {}
        vertex.set_safety(str(settings.get("safety", "BLOCK_MEDIUM_AND_ABOVE")))
        providers = vertex.Providers(fal, client, fallback, others)
        providers.max_requests = int(settings.get("max_requests", 600))
        providers.max_failures = int(settings.get("max_failures_in_a_row", 8))
        return moved, providers
    if others:
        return stills, vertex.Providers(fal, None, {}, others)
    return stills, fal


def check_vertex(stills: dict) -> int:
    """Vertex AI: the key signs in and each model is in the catalogue (free; nothing is generated)."""
    from production import vertex
    from production.media import MediaError

    if not vertex.available():
        print("GEMINI_PLATFORM_API_KEY / GCP_SERVICE_ACCOUNT_JSON: not set (pictures stay on fal)")
        return 0
    try:
        client = vertex.client()
        client.token()
    except MediaError as exc:
        print(f"Vertex AI: {exc}")
        return 1
    print("Google: key set")
    failed = 0
    models = sorted(set((stills.get("vertex") or {}).get("models", {}).values()))
    for model in models if not isinstance(client, vertex.VertexKey) else ():  # an API key cannot read it
        try:
            info = client.model_info(model)
            print(f"  vertex:{model}  found: {str(info.get('name', ''))[:120]} "
                  f"{str(info.get('launchStage', '') or info.get('versionState', ''))}")
        except MediaError as exc:
            print(f"  vertex:{model}  catalogue lookup failed: {str(exc)[:300]}")
    # Two tiny real calls prove the key, the API, the billing and each model: a word from the
    # checker (a fraction of a cent) and one small picture (about $0.03).
    text = next((m for m in models if "image" not in m and "banana" not in m), "")
    if text:
        try:
            got = client.run(f"vertex:{text}", {"prompt": "Reply with the single word OK.", "max_tokens": 5})
            print(f"  checker: {getattr(client, '_text_model', text)} answered {got.get('output', '')[:40]!r}")
        except MediaError as exc:
            failed += 1
            print(f"  checker {text}: CALL FAILED: {str(exc)[:400]}")
    picture = next((m for m in models if "banana" in m or "image" in m), "")
    if picture:
        try:
            import base64
            got = client.run(f"vertex:{picture}", {"prompt": "A dark, empty room at night, one red light. "
                                                             "A film still.", "aspect_ratio": "16:9",
                                                   "resolution": "1K"})
            size = len(base64.b64decode(got["images"][0]["url"].split(",", 1)[1]))
            print(f"  pictures: {picture} painted a test picture ({size // 1024} KB)")
        except MediaError as exc:
            failed += 1
            print(f"  pictures {picture}: CALL FAILED: {str(exc)[:400]}")
    # Models used only sometimes (the score, moving shots, the stronger checker): asked with an empty request,
    # which costs nothing: "found" when Google says the request is wrong, "not found" when the model is unknown.
    others = {str(m).split(":", 1)[1] for m in stills.get("models", {}).values() if str(m).startswith("vertex:")}
    others |= {str(v) for v in ((stills.get("vertex") or {}).get("roles") or {}).values()}
    others |= {"lyria-002"} if "fal-ai/ace-step-1.5" in (stills.get("vertex") or {}).get("models", {}) else set()
    for model in sorted(others - set(models)):
        method, where = ("predict", "us-central1") if "lyria" in model else (
            ("predictLongRunning", "us-central1") if "veo" in model else ("generateContent", None))
        try:
            client.http(client.url(model, method, where), method="POST", body={"instances": []} if method != (
                "generateContent") else {"contents": []}, timeout=60, headers=client._headers())
            print(f"  vertex:{model}  found")
        except MediaError as exc:
            text = str(exc)
            print(f"  vertex:{model}  {'NOT FOUND' if 'HTTP 404' in text else 'found'}"
                  f"{'' if 'HTTP 404' not in text and 'HTTP 400' in text else ': ' + text[:160]}")
    print(f"  project: {'found' if client.project else 'not found'}, location {client.location}")
    return 1 if failed else 0


def check(models: dict, model_args: dict | None = None) -> int:
    """Every paid model's inputs and choices, from fal's public descriptions. No spend."""
    from production.media import Fal

    print("FAL_KEY:", "set" if os.environ.get("FAL_KEY") else "MISSING (add it as a GitHub Actions secret)")
    fal = Fal(key=os.environ.get("FAL_KEY") or "not-needed-to-read-schemas")
    failed = 0
    # empty: switched off; "vertex:" and "elevenlabs:" models are not on fal (check_vertex asks Google)
    roles = {role: endpoint for role, endpoint in models.items() if endpoint and ":" not in endpoint}
    seen = set(roles.values())
    roles.update({f"(alt) {e}": e for e in ALTERNATES if e not in seen})
    for role, endpoint in roles.items():
        try:
            info = fal.schema(endpoint)
        except Exception as exc:  # noqa: BLE001 - a report, not a run
            failed += not role.startswith("(alt)")
            print(f"{role:<12} {endpoint}  NOT FOUND: {str(exc)[:160]}")
            continue
        price = fal.price(endpoint) if os.environ.get("FAL_KEY") else ""
        print(f"{role:<12} {endpoint}{'  [' + price + ']' if price else ''}\n    inputs: {', '.join(info['fields'])}\n"
              f"    required: {', '.join(info['required'])}")
        for field, choices in info.get("options", {}).items():
            print(f"    {field}: {json.dumps(choices)[:1500]}")
        args = (model_args or {}).get(role, {})
        unknown = set(args) - set(info["fields"]) if info["fields"] else set()
        if unknown:
            print(f"    WARNING: configured arguments the model doesn't list: {sorted(unknown)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
