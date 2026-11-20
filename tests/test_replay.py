"""Resumable runs: model answers kept on disk, the cache packed for keeping, the preflight agent."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    from PIL import Image
    HAVE = True
except ImportError:
    HAVE = False


class ReplayTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_a_text_answer_is_asked_once_and_media_is_never_kept_twice(self):
        from production.replay import Replay

        asked = []

        class Inner:
            fell_back = 3

            def run(self, endpoint, args):
                asked.append(endpoint)
                if endpoint == "vision":
                    return {"output": json.dumps({"ok": len(asked) == 1})}
                return {"images": [{"url": "data:image/png;base64,AAAA"}]}

        replay = Replay(Inner(), self.root / "answers")
        first = replay.run("vision", {"prompt": "p", "image_urls": ["data:image/png;base64,QUJD"]})
        again = replay.run("vision", {"prompt": "p", "image_urls": ["data:image/png;base64,QUJD"]})
        self.assertEqual(first, again)  # the second verdict would have differed: it is the first one
        self.assertEqual(asked, ["vision"])
        replay.run("vision", {"prompt": "another picture", "image_urls": []})
        self.assertEqual(len(asked), 2)
        replay.run("picture", {"prompt": "p"})
        replay.run("picture", {"prompt": "p"})
        self.assertEqual(asked.count("picture"), 2)  # pictures are cached by the studio, as files
        self.assertEqual(replay.fell_back, 3)  # the client's own counters show through
        self.assertEqual(replay.replayed, 1)

    def test_a_rerun_gets_the_same_plan_from_the_planner(self):
        from production.replay import ReplayRouter

        calls = []

        class Planner:
            def generate_json(self, task, system, prompt, schema):
                calls.append(task)
                return {"beats": [len(calls)]}

        router = SimpleNamespace(one=lambda role: Planner())
        first = ReplayRouter(router, self.root / "plans").one("planner").generate_json("storyboard", "s", "p", {})
        later = ReplayRouter(router, self.root / "plans").one("planner").generate_json("storyboard", "s", "p", {})
        self.assertEqual(first, later)
        self.assertEqual(calls, ["storyboard"])
        ReplayRouter(router, self.root / "plans").one("planner").generate_json("storyboard", "s", "other", {})
        self.assertEqual(len(calls), 2)

    def test_the_cache_folder_comes_from_the_environment(self):
        import os
        from unittest import mock

        from production.replay import cache_root

        self.assertEqual(cache_root({}, self.root / "work"), self.root / "work" / "cache")
        with mock.patch.dict(os.environ, {"MEDIA_CACHE_DIR": str(self.root / "kept")}):
            self.assertEqual(cache_root({}, self.root / "work"), (self.root / "kept").resolve())

    @unittest.skipUnless(HAVE, "numpy or pillow not installed")
    def test_packing_turns_large_pictures_into_jpegs_under_the_same_names(self):
        from production.replay import pack

        src = self.root / "cache"
        (src / "media").mkdir(parents=True)
        noise = (np.random.default_rng(1).random((900, 1600, 3)) * 255).astype("uint8")
        Image.fromarray(noise).save(src / "media" / "picture_shot-abc.png")
        (src / "plans").mkdir()
        (src / "plans" / "storyboard-x.json").write_text("{}")
        report = pack(src, self.root / "packed")
        self.assertEqual(report["files"], 2)
        self.assertEqual(report["pictures_shrunk"], 1)
        kept = self.root / "packed" / "media" / "picture_shot-abc.png"
        with Image.open(kept) as im:
            self.assertEqual(im.format, "JPEG")
            self.assertEqual(im.size, (1600, 900))
        self.assertLess(kept.stat().st_size, (src / "media" / "picture_shot-abc.png").stat().st_size)
        self.assertEqual((self.root / "packed" / "plans" / "storyboard-x.json").read_text(), "{}")

    def test_the_beats_to_repaint_are_read_from_the_overrides(self):
        from production.replay import parse_redo

        self.assertEqual(parse_redo("12,31:no_text_on_the_wall"), {12: "", 31: "no text on the wall"})
        self.assertEqual(parse_redo(""), {})
        self.assertEqual(parse_redo("x,4"), {4: ""})


@unittest.skipUnless(HAVE, "numpy or pillow not installed")
class PreflightTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def board(self, fal, **over):
        from production.paid import Budget, Studio
        from production.stills.board import Board
        from tests.test_stills import cfg, story

        c = cfg(**over)
        studio = Studio(fal, Budget(10.0, c["prices"]), self.root / "cache", c)
        return Board(story=story(), out=self.root, work=self.root / "work", cfg=c, studio=studio,
                     budget=studio.budget)

    def test_an_empty_fal_balance_stops_the_run_before_anything_is_bought(self):
        from production.media import MediaError
        from production.stills.preflight import PreflightAgent, PreflightFailed

        class Empty:
            def run(self, endpoint, args):
                raise MediaError("HTTP 403: User is locked. Reason: Exhausted balance.")

        board = self.board(Empty())
        with self.assertRaises(PreflightFailed) as caught:
            PreflightAgent().run(board)
        self.assertIn("Top up fal.ai", str(caught.exception))
        self.assertEqual(board.budget.spent, 0)
        # with the free voice allowed, the voice failure is let through, but a Google checker that does not
        # answer still stops the run: pictures are made on Google only
        board = self.board(Empty(), require_paid_voice=False)
        with self.assertRaises(PreflightFailed) as caught:
            PreflightAgent().run(board)
        self.assertIn("Google", str(caught.exception))
        self.assertTrue(board.data["preflight"]["voice"].startswith("failed"))

    def test_a_working_narrator_and_checker_pass_and_the_cost_is_estimated(self):
        from production.stills.preflight import PreflightAgent

        asked = []

        class Fine:
            def run(self, endpoint, args):
                asked.append((endpoint, args.get("text") or args.get("prompt")))
                if "gemini-3" in endpoint:  # the checker, on Google
                    return {"output": "OK"}
                return {"audio": {"url": "https://example.com/a.mp3"}}

        board = self.board(Fine())
        report = PreflightAgent().run(board)
        self.assertEqual((report["voice"], report["checker"]), ("ok", "ok"))
        self.assertEqual(asked[0], ("fal-ai/elevenlabs/tts/eleven-v3", "The lantern is lit."))
        self.assertGreater(report["estimate"]["usd"], 0)
        self.assertGreaterEqual(report["estimate"]["beats"], 1)
        self.assertNotIn("warning", report)
        board = self.board(Fine(), max_usd=0.01)  # a cap the episode cannot fit under is reported
        self.assertIn("over the $0.01 cap", PreflightAgent().run(board)["warning"])

    def test_without_a_paid_key_nothing_is_asked(self):
        from production.stills.board import Board
        from production.stills.preflight import PreflightAgent
        from tests.test_stills import cfg, story

        board = Board(story=story(), out=self.root, work=self.root, cfg=cfg())
        self.assertEqual(PreflightAgent().run(board), {"paid": False})


if __name__ == "__main__":
    unittest.main()
