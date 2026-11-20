"""The review's direction changes: tension, the reveal ladder, rationed and built scares, the
title card, takes, the art review's looks, continuity, staging, dread pacing, the stronger checker."""

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

try:
    import numpy as np
    from PIL import Image
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False

from production.shots import Shot


def beat(index, scene=0, **over):
    return Shot(index=index, scene=scene, t0=over.pop("t0", index * 6.0), t1=over.pop("t1", index * 6.0 + 6.0),
                text=over.pop("text", f"line {index}"), **over)


class StoryboardRules(unittest.TestCase):
    def test_scares_are_rationed_and_the_most_important_keep_theirs(self):
        from production.stills.storyboard import limit_scares

        beats = [beat(i, scare=True, importance=i % 5 + 1) for i in range(8)]
        self.assertEqual(limit_scares(beats, minutes=10, per_minute=0.35, most=3), 3)
        kept = [b.index for b in beats if b.scare]
        self.assertEqual(kept, [2, 3, 4])  # importance 3, 4 and 5: the most important three
        short = [beat(i, scare=True) for i in range(3)]
        self.assertEqual(limit_scares(short, minutes=1, per_minute=0.35, most=3), 1)  # never fewer than one
        self.assertEqual(sum(b.scare for b in short), 1)

    def test_the_threat_is_shown_more_only_as_the_climax_nears(self):
        from production.stills.storyboard import ladder

        beats = [beat(i, threat=4) for i in range(10)]
        beats[8].scare, beats[8].importance = True, 5  # the climax
        report = ladder(beats, set(), max_full=1)
        self.assertEqual(report["climax"], 8)
        self.assertEqual([b.threat for b in beats[:4]], [2, 2, 2, 2])  # before halfway to the climax
        self.assertEqual([b.threat for b in beats[4:8]], [3, 3, 3, 3])  # partly seen until the climax
        self.assertEqual(beats[8].threat, 4)  # fully seen once, at the climax
        self.assertEqual(beats[9].threat, 3)  # and not again
        self.assertGreater(report["lowered"], 0)

    def test_calm_scenes_hold_their_pictures_and_terrifying_ones_cut_fast(self):
        from production.shots import cut, mood_tension, scene_limits

        self.assertEqual(mood_tension("sheer terror"), 5)
        self.assertEqual(mood_tension("creeping dread"), 4)
        self.assertEqual(mood_tension("eerie"), 3)
        self.assertEqual(mood_tension("calm, ordinary evening"), 1)
        limits = scene_limits([{"mood": "calm"}, {"mood": "terror"}], 5.0, 11.0)
        self.assertEqual(limits[0], (6.0, 13.2))
        self.assertEqual(limits[1], (3.0, 6.6))
        chunks = [SimpleNamespace(t0=0.0, t1=12.0, text="a " * 30, start=0),
                  SimpleNamespace(t0=12.0, t1=24.0, text="b " * 30, start=100)]
        shots = cut(chunks, [0, 1], 24.0, 5.0, 11.0, limits)
        calm = [s for s in shots if s.scene == 0]
        scary = [s for s in shots if s.scene == 1]
        self.assertEqual(len(calm), 1)  # 12 s held as one picture
        self.assertGreaterEqual(len(scary), 2)  # the same length cut at least in two


class CameraRules(unittest.TestCase):
    def test_tension_speeds_the_move_and_a_reveal_jolts(self):
        from production.stills import camera

        calm = camera.path("push_in", 8.0, tension=1)
        tense = camera.path("push_in", 8.0, tension=5)
        self.assertLess(calm[1] - calm[0], tense[1] - tense[0])
        jolt = camera.path("jolt", 4.0)
        self.assertEqual(jolt[4], "jolt")
        self.assertGreater(jolt[0], 1.0)  # starts zoomed in, so the shake never shows an edge
        text = camera.move_filter(jolt, 96, (2700, 1520), (2400, 1350))
        self.assertIn("exp(-(on-1)/5)", text)
        self.assertEqual(camera.move_for(beat(1, framing="wide"), 1, after_scare=True), "pull_out")
        self.assertEqual(camera.move_for(beat(1, scare=True), 1, after_scare=True), "push_in_fast")

    def test_a_built_scare_is_two_shots_dark_then_the_reveal(self):
        from production.edit import Segment
        from production.stills.post import build_scares

        scare = beat(3, scare=True, reveal=2.0)
        plain = beat(4)
        story = [Segment("seg_003.mp4", 10.0, 16.0, Path("p.png"), camera="push_in_fast"),
                 Segment("seg_004.mp4", 16.0, 22.0, Path("q.png"), camera="push_in")]
        segs = build_scares(list(story), story, [scare, plain], 0.22)
        self.assertEqual([s.name for s in segs], ["seg_003_build.mp4", "seg_003.mp4", "seg_004.mp4"])
        self.assertEqual((segs[0].start, segs[0].end, segs[0].dim), (10.0, 12.0, 0.22))
        self.assertEqual((segs[1].start, segs[1].camera), (12.0, "jolt"))
        self.assertEqual(segs[2].dim, 1.0)

    @unittest.skipUnless(HAVE, "ffmpeg not installed")
    def test_the_build_up_renders_dark_and_the_reveal_bright(self):
        from production.edit import Segment
        from production.stills.post import render

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        Image.fromarray(np.full((360, 640, 3), 140, dtype="uint8")).save(root / "p.png")
        means = []
        for name, dim, move in (("a.mp4", 0.22, "push_in"), ("b.mp4", 1.0, "jolt")):
            seg = Segment(str(root / name), 0.0, 1.0, root / "p.png", camera=move, dim=dim)
            render(seg, root, {"grain": 0.0, "vignette": 0.0}, 0, (640, 360))
            import subprocess
            raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(root / name), "-frames:v", "1", "-f", "rawvideo",
                                  "-pix_fmt", "gray", "-"], capture_output=True).stdout
            means.append(float(np.frombuffer(raw, dtype=np.uint8).mean()))
        self.assertLess(means[0], means[1] * 0.45)


class TitleCards(unittest.TestCase):
    def test_each_story_keeps_its_own_card_and_stories_differ(self):
        from production.stills.post import TITLE_LAYOUTS, title_layout

        a = title_layout({"id": 12, "title": "The Night Bus"}, {})
        self.assertEqual(a, title_layout({"id": 12, "title": "The Night Bus"}, {}))
        seen = {title_layout({"id": i, "title": f"T{i}"}, {}) for i in range(40)}
        self.assertGreater(len(seen), 2)
        self.assertTrue(seen <= set(TITLE_LAYOUTS))
        self.assertEqual(title_layout({"id": 1}, {"title_variety": False}), TITLE_LAYOUTS[0])


@unittest.skipUnless(HAVE, "video libraries or ffmpeg not installed")
class Takes(unittest.TestCase):
    def test_a_key_picture_is_painted_twice_and_the_checker_keeps_the_better_take(self):
        from production.paid import Budget, Studio
        from production.stills.images import Critic, Painter
        from tests.test_stills import FakeFal, cfg, painted

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)

        class Picky(FakeFal):
            def run(self, endpoint, args):
                if "any-llm" in endpoint and "takes of the same frame" in args.get("prompt", ""):
                    self.calls.append((endpoint, {"prompt": args["prompt"]}))
                    return {"output": json.dumps({"best": 2, "why": "her fear reads"})}
                return super().run(endpoint, args)

        fal = Picky(root)
        c = cfg()
        studio = Studio(fal, Budget(5.0, c["prices"]), root / "cache", c)
        painter = Painter(studio, c, root / "art", Critic(studio, "dark"))
        out, step = painter.paint("the reveal", [painted(root / "ref.png")], root / "beat.png", "k",
                                  c["chains"]["scare"], check="the reveal", takes=2)
        self.assertEqual(step, "fal:shot")
        self.assertEqual(painter.counts, {"fal:shot (best of 2)": 1})
        pictures = [e for e, _ in fal.calls if e == "google/nano-banana-2.1/edit"]
        self.assertEqual(len(pictures), 2)  # two takes, both paid
        self.assertTrue(any("takes of the same frame" in a.get("prompt", "") for _, a in fal.calls))
        self.assertTrue(out.exists())


class ArtReviewLooks(unittest.TestCase):
    def test_a_long_episode_is_reviewed_scene_by_scene_and_person_by_person(self):
        from production.stills.checks import ArtReviewAgent

        beats = [beat(i, scene=i // 9, characters=["Mara"] if i % 3 == 0 else []) for i in range(90)]
        looks = ArtReviewAgent.groups(beats, {0, 89}, {"portraits": {"Mara": Path("mara.png")}})
        scenes = [g for label, g, refs in looks if not label.startswith("person")]
        people = [(label, g, refs) for label, g, refs in looks if label.startswith("person")]
        self.assertTrue(all(len(g) <= 12 for g in scenes))
        self.assertEqual(sum(len(g) for g in scenes), 88)  # every story picture, once
        self.assertEqual(people[0][0], "person Mara")
        self.assertLessEqual(len(people[0][1]), 12)
        self.assertEqual(people[0][2], [Path("mara.png")])  # the portrait goes with the sheet


class ContinuityAndStaging(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_a_change_the_story_makes_is_held_until_it_ends(self):
        from production.stills.board import Board
        from production.stills.continuity import ContinuityAgent

        class Planner:
            def generate_json(self, task, system, prompt, schema):
                assert task == "continuity"
                return {"facts": [{"fact": "Mara's left hand is wrapped in a bloody cloth", "from": 2, "until": 4},
                                  {"fact": "x", "from": 1, "until": 2},  # too short to be a fact
                                  {"fact": "The hallway light is off", "from": 3, "until": 99}]}

        board = Board(story={}, out=self.root, work=self.root, cfg={},
                      router=SimpleNamespace(one=lambda role: Planner()))
        board.data.update(beats=[beat(i) for i in range(6)], host_beats={0}, cast=[])
        report = ContinuityAgent().run(board)
        self.assertEqual(report["facts"], 2)
        sheet = board.data["continuity"]
        self.assertEqual(sheet[2], ["Mara's left hand is wrapped in a bloody cloth"])
        self.assertEqual(sheet[4], ["Mara's left hand is wrapped in a bloody cloth", "The hallway light is off"])
        self.assertEqual(sheet[5], ["The hallway light is off"])  # held to the last picture
        self.assertNotIn(1, sheet)

    def test_a_fact_about_a_place_stays_there_and_one_about_a_person_travels(self):
        from production.stills.board import Board
        from production.stills.continuity import ContinuityAgent

        class Planner:
            def generate_json(self, task, system, prompt, schema):
                return {"facts": [{"fact": "Room 6's door is nailed shut with planks", "from": 1, "until": 5},
                                  {"fact": "June holds her phone in both hands", "from": 1, "until": 5}]}

        board = Board(story={}, out=self.root, work=self.root, cfg={},
                      router=SimpleNamespace(one=lambda role: Planner()))
        beats = [beat(i) for i in range(6)]
        for b in beats:
            b.scene = 0 if b.index < 3 else 1
        board.data.update(beats=beats, host_beats=set(), cast=[{"name": "June"}],
                          scenes=[{"location": "the motel, outside"}, {"location": "the motel office"}])
        report = ContinuityAgent().run(board)
        sheet = board.data["continuity"]
        self.assertIn("Room 6's door is nailed shut with planks", sheet[2])
        self.assertEqual(sheet[4], ["June holds her phone in both hands"])  # the planks stay outside
        self.assertEqual(report["kept_in_their_place"], 3)

    def test_continuity_reaches_the_prompt_and_the_check(self):
        from production.stills.art import IllustratorAgent

        b = beat(3, picture="Mara opens the door.", characters=["Mara"], threat=2)
        text = IllustratorAgent().prompt(b, "HOUSE.", {"Mara": "a woman"}, [], False, {}, facts=["The hallway light is off"])
        self.assertIn("Continuity: The hallway light is off.", text)
        self.assertIn("only a dark shape deep in the shadows", text)
        check = IllustratorAgent.check_text(b, {"Mara": "a woman"}, ["The hallway light is off"])
        self.assertIn("threat shown too clearly", check)
        self.assertIn("must not contradict what the story has established: The hallway light is off", check)
        from production.stills.images import serious
        self.assertTrue(serious("threat shown too clearly: the creature is fully lit"))

    @unittest.skipUnless(HAVE, "pillow not installed")
    def test_the_cinematographer_restages_only_what_does_not_fit_the_set(self):
        from production.paid import Budget, Studio
        from production.stills.board import Board
        from production.stills.staging import StagingAgent

        Image.new("RGB", (640, 360), (30, 30, 30)).save(self.root / "sheet.png")
        asked = []

        class Eye:
            def run(self, endpoint, args):
                asked.append((endpoint, args.get("model"), len(args.get("image_urls") or [])))
                return {"output": json.dumps({"pictures": [
                    {"index": 2, "picture": "From the back seat, down the aisle to the driver's cab, Dana stands.",
                     "why": "the set has no window on the left"},
                    {"index": 99, "picture": "not a beat in this place at all, ignored"}]})}

        c = {"models": {"vision": "fal-ai/any-llm/vision", "vision_pro": "fal-ai/any-llm/vision"},
             "model_args": {"vision": {"model": "flash"}, "vision_pro": {"model": "pro"}}}
        studio = Studio(Eye(), Budget(1.0, {}), self.root / "cache", c)
        board = Board(story={}, out=self.root, work=self.root, cfg={}, studio=studio)
        beats = [beat(i, scene=1, picture=f"picture {i}") for i in range(1, 4)]
        board.data.update(beats=beats, host_beats=set(), sets=[
            {"name": "the night bus", "scenes": [1], "design": "a cream bus", "sheet": self.root / "sheet.png"}])
        report = StagingAgent().run(board)
        self.assertEqual(report["rewritten"][0]["beat"], 2)
        self.assertTrue(beats[1].picture.startswith("From the back seat"))
        self.assertEqual(beats[0].picture, "picture 1")
        self.assertEqual(asked, [("fal-ai/any-llm/vision", "pro", 1)])  # the stronger checker, with the sheet


class DreadPacing(unittest.TestCase):
    def test_dread_is_read_slower_with_longer_silences(self):
        from production.voice import pause_for, speed_for

        scenes = [{"mood": "calm", "delivery": "steady"}, {"mood": "creeping dread", "delivery": "steady"}]
        spans = [(0, 100), (100, 200)]
        self.assertEqual(speed_for(10, spans, scenes), 1.0)
        self.assertLess(speed_for(150, spans, scenes), 1.0)
        self.assertEqual(pause_for(10, spans, scenes), 1.0)
        self.assertGreater(pause_for(150, spans, scenes), 1.0)


class StrongerChecker(unittest.TestCase):
    def test_the_pro_role_has_its_own_vertex_model_and_falls_back_to_fal(self):
        from production.vertex import PREFIX, use_vertex

        cfg = {"models": {"vision": "fal-ai/any-llm/vision", "vision_pro": "fal-ai/any-llm/vision",
                          "music": "vertex:lyria-002"},
               "vertex": {"models": {"fal-ai/any-llm/vision": "gemini-3.6-flash"},
                          "roles": {"vision_pro": "gemini-3.6-pro"}, "prices": {"gemini-3.6-pro": 0.01},
                          "fallback_to_fal": True}}
        moved, fallback = use_vertex(cfg)
        self.assertEqual(moved["models"]["vision"], PREFIX + "gemini-3.6-flash")
        self.assertEqual(moved["models"]["vision_pro"], PREFIX + "gemini-3.6-pro")
        self.assertEqual(moved["models"]["music"], "vertex:lyria-002")  # already on Vertex: left alone
        self.assertEqual(fallback[PREFIX + "gemini-3.6-pro"], ["fal-ai/any-llm/vision"])

    def test_a_missing_pro_model_tries_the_other_pro_models_first(self):
        from production.media import MediaError
        from production.vertex import VertexKey

        tried = []

        def http(url, method="GET", body=None, timeout=0, headers=None):
            model = url.split("/models/")[1].split(":")[0]
            tried.append(model)
            if model in ("gemini-3.6-pro", "gemini-3.5-pro"):
                raise MediaError(f"POST {url}: HTTP 404: model not found")
            return {"candidates": [{"content": {"parts": [{"text": "OK"}]}}]}

        v = VertexKey(key="k", project="123", http=http)
        self.assertEqual(v.run("vertex:gemini-3.6-pro", {"prompt": "hi"})["output"], "OK")
        self.assertEqual(tried, ["gemini-3.6-pro", "gemini-3.5-pro", "gemini-3.1-pro"])
        v.run("vertex:gemini-3.6-pro", {"prompt": "again"})
        self.assertEqual(tried[-1], "gemini-3.1-pro")  # the one that answered is remembered


if __name__ == "__main__":
    unittest.main()
