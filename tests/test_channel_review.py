"""The review's channel changes: depth parallax, the screening, the retention curve, translations,
compilations and localized uploads."""

import json
import shutil
import subprocess
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


class Depth(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def picture(self):
        pic = np.full((360, 640, 3), 30, dtype=np.uint8)
        pic[:, ::40] = 90
        pic[120:300, 250:390] = (180, 150, 120)
        Image.fromarray(pic).save(self.root / "p.png")
        near = np.zeros((360, 640), dtype=np.float32)
        near[120:300, 250:390] = 1.0
        return self.root / "p.png", near

    @unittest.skipUnless(HAVE, "pillow or numpy not installed")
    def test_the_model_is_given_a_normalised_square_and_its_answer_is_scaled_back(self):
        from production.stills.depth import DepthModel

        seen = {}

        class Session:
            def get_inputs(self):
                return [SimpleNamespace(name="pixel_values")]

            def run(self, outputs, feeds):
                seen["shape"] = feeds["pixel_values"].shape
                ramp = np.tile(np.linspace(0, 5, 518, dtype=np.float32), (518, 1))
                return [ramp[None]]

        path, _ = self.picture()
        near = DepthModel("repo", "file", session_factory=lambda _: Session())(path)
        self.assertEqual(seen["shape"], (1, 3, 518, 518))
        self.assertEqual(near.shape, (360, 640))
        self.assertLess(near[:, :20].mean(), 0.1)  # far on the left
        self.assertGreater(near[:, -20:].mean(), 0.9)  # near on the right

    @unittest.skipUnless(HAVE, "pillow or numpy not installed")
    def test_a_picture_splits_into_a_filled_background_and_a_feathered_foreground(self):
        from production.stills.depth import layers

        path, near = self.picture()
        bg, fg = layers(path, near, self.root / "layers")
        with Image.open(fg) as im:
            alpha = np.asarray(im)[..., 3]
        self.assertGreater(alpha[210, 320], 240)  # the figure is in front
        self.assertLess(alpha[20, 20], 5)  # the wall is not
        with Image.open(bg) as im:
            plate = np.asarray(im.convert("L"), dtype=np.float32)
        self.assertLess(plate[210, 320], 120)  # where the figure stood, the background shows through
        flat = np.zeros((360, 640), dtype=np.float32)
        self.assertIsNone(layers(path, flat, self.root / "layers2"))  # no clear foreground: no parallax

    @unittest.skipUnless(HAVE, "ffmpeg not installed")
    def test_a_parallax_shot_renders_with_the_film_finish(self):
        from production.edit import Segment
        from production.stills.depth import layers
        from production.stills.post import render

        path, near = self.picture()
        seg = Segment(str(self.root / "x.mp4"), 0.0, 1.0, path, camera="push_in", layers=layers(path, near, self.root))
        render(seg, self.root, {"grain": 0.01, "vignette": 0.2}, 0, (640, 360), {})
        probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=width,height,nb_frames,color_space",
                                "-of", "csv=p=0", str(self.root / "x.mp4")], capture_output=True, text=True).stdout
        self.assertEqual(probe.strip(), "640,360,bt709,24")

    def test_parallax_is_off_by_default(self):
        from production.stills.board import Board
        from production.stills.depth import DepthAgent
        from story_factory.config import DEFAULTS

        board = Board(story={}, out=self.root, work=self.root, cfg=DEFAULTS["stills"])
        self.assertEqual(DepthAgent().run(board), {"enabled": False})


class Screening(unittest.TestCase):
    def test_times_are_read_and_pinned_to_beats(self):
        from production.stills.screening import beat_at, seconds_of

        self.assertEqual(seconds_of("3:05"), 185.0)
        self.assertEqual(seconds_of("1:02:03"), 3723.0)
        self.assertEqual(seconds_of(42), 42.0)
        self.assertIsNone(seconds_of(""))
        times = [{"beat": 4, "start": 10.0, "end": 16.0}, {"beat": 5, "start": 16.0, "end": 22.0}]
        self.assertEqual(beat_at(17.0, times), 5)
        self.assertIsNone(beat_at(80.0, times))

    @unittest.skipUnless(HAVE, "ffmpeg not installed")
    def test_the_cut_is_watched_by_gemini_and_its_notes_point_at_beats(self):
        from production.paid import Budget, Studio
        from production.stills.board import Board
        from production.stills.screening import ScreeningAgent

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=24:duration=6",
                        "-f", "lavfi", "-i", "sine=frequency=220:duration=6", "-shortest", str(root / "video.mp4")],
                       check=True)
        asked = []

        class Gemini:
            def run(self, endpoint, args):
                asked.append((endpoint, args["video_url"][:22], args["prompt"]))
                return {"output": json.dumps({"score": 6, "hook": "slow", "leave_at": "0:04", "notes": [
                    {"at": "0:03", "kind": "picture", "severity": 3, "note": "she is calm while the line screams"},
                    {"at": "0:05", "kind": "drags", "severity": 2, "note": "holds too long"}],
                    "scares": [{"at": "0:04", "landed": False, "why": "the sting came early"}]})}

        studio = Studio(Gemini(), Budget(5.0, {"vertex:gemini-3.6-pro": 0.01}), root / "cache",
                        {"models": {"vision": "vertex:gemini-3.6-flash", "vision_pro": "vertex:gemini-3.6-pro"}})
        board = Board(story={"title": "T"}, out=root, work=root, cfg={}, studio=studio)
        board.data.update(video=root / "video.mp4", total=6.0, offset=1.0, titles=["The Night Bus"],
                          beat_times=[{"beat": 1, "start": 1.0, "end": 3.5}, {"beat": 2, "start": 3.5, "end": 6.0}])
        report = ScreeningAgent().run(board)
        self.assertTrue(report["watched"])
        self.assertEqual(asked[0][:2], ("vertex:gemini-3.6-pro", "data:video/mp4;base64,"))
        self.assertIn("The Night Bus", asked[0][2])
        self.assertEqual([n["beat"] for n in report["notes"]], [1, 2])
        self.assertEqual(report["redo_hint"], "1")  # a serious picture fault: redo=1
        self.assertEqual(report["leave_beat"], 2)
        self.assertFalse(report["scares"][0]["landed"])

    def test_without_google_the_cut_is_not_watched(self):
        from production.stills.board import Board
        from production.stills.screening import ScreeningAgent

        studio = SimpleNamespace(models={"vision": "other:checker"})
        board = Board(story={}, out=Path("."), work=Path("."), cfg={}, studio=studio)
        board.data["video"] = Path(__file__)
        self.assertFalse(ScreeningAgent().run(board)["watched"])


class Retention(unittest.TestCase):
    def test_the_steepest_drops_are_found_on_the_beats_and_written_as_lessons(self):
        from production.retention import PREFIX, drops, lessons

        curve = [(i / 100, 1.0 - 0.002 * i) for i in range(101)]
        curve[40] = (0.40, curve[39][1] - 0.12)  # a cliff at 40%
        curve = [(r, w - (0.12 if r > 0.40 else 0)) for r, w in curve]
        beats = [{"beat": 7, "start": 230.0, "end": 245.0, "line": "She waited in the kitchen.", "tension": 2}]
        found = drops(curve, 600.0, beats)
        self.assertEqual(found[0]["beat"], 7)
        self.assertAlmostEqual(found[0]["at"], 234.0, delta=1)
        notes, measured = lessons("The Night Bus", curve, 600.0, beats)
        self.assertTrue(notes[0].startswith(f"{PREFIX} \"The Night Bus\":"))
        self.assertIn("0:30", notes[0])
        self.assertIn("beat 7", notes[1])
        self.assertIn("She waited", notes[1])
        self.assertGreater(measured["still_watching_at_30s"], 0.9)

    def test_the_curve_is_read_from_youtube_analytics(self):
        from production import youtube

        def http(url, method="GET", headers=None, data=None, timeout=0):
            assert "dimensions=elapsedVideoTimeRatio" in url and "audienceWatchRatio" in url
            return 200, {}, json.dumps({"rows": [[0.01, 1.0], [0.02, 0.95]]}).encode()

        yt = youtube.YouTube.__new__(youtube.YouTube)
        yt.http, yt.token = http, "t"
        self.assertEqual(yt.retention("abc"), [(0.01, 1.0), (0.02, 0.95)])


class Languages(unittest.TestCase):
    def test_titles_and_captions_are_translated_and_keep_their_timing(self):
        from production.paid import Budget, Studio
        from production.stills.translate import localize
        from production.timing import Cue

        class Gemini:
            def run(self, endpoint, args):
                if "caption line" in args["prompt"]:
                    lines = json.loads(args["prompt"].split("\n", 1)[1].rsplit("\nReply", 1)[0])["lines"]
                    return {"output": json.dumps({"lines": [f"[es] {x}" for x in lines]})}
                return {"output": json.dumps({"title": "El último autobús", "description": "Una historia."})}

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        studio = Studio(Gemini(), Budget(1.0, {}), root / "c", {"models": {"vision": "vertex:gemini-3.6-flash"}})
        cues = [Cue(0.0, 2.0, "The bell rang."), Cue(2.5, 4.0, "Nobody was there.")]
        got = localize(studio, "The Last Bus", "A story.", cues, ["es"], root)
        self.assertEqual(got["localizations"]["es"]["title"], "El último autobús")
        srt = (root / "captions_es.srt").read_text()
        self.assertIn("[es] Nobody was there.", srt)
        self.assertIn("00:00:02,500 --> 00:00:04,000", srt)
        fal_only = Studio(Gemini(), Budget(1.0, {}), root / "c2", {"models": {"vision": "other:checker"}})
        self.assertEqual(localize(fal_only, "T", "D", cues, ["es"], root), {"localizations": {}, "captions": {}})

    def test_the_upload_carries_the_translations_and_every_caption_track(self):
        from production import youtube

        calls = []

        def http(url, method="GET", headers=None, data=None, timeout=0):
            calls.append((method, url, data))
            if "uploadType=resumable" in url:
                return 200, {"Location": "https://upload/put"}, b""
            if url == "https://upload/put":
                return 200, {}, json.dumps({"id": "vid1"}).encode()
            return 200, {}, b"{}"

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        (root / "video.mp4").write_bytes(b"v")
        (root / "captions.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhi\n")
        (root / "captions_es.srt").write_text("1\n00:00:00,000 --> 00:00:01,000\nhola\n")
        (root / "metadata.json").write_text(json.dumps({"title": "T", "description": "D", "tags": [],
                                                        "localizations": {"es": {"title": "T es", "description": "D es"}},
                                                        "captions": {"es": "captions_es.srt"}}))
        yt = youtube.YouTube.__new__(youtube.YouTube)
        yt.http, yt.token = http, "t"
        done = youtube.publish(yt, root)
        first = calls[0]
        self.assertIn("part=snippet,status,localizations", first[1])
        body = json.loads(first[2])
        self.assertEqual(body["localizations"]["es"]["title"], "T es")
        self.assertEqual(body["snippet"]["defaultLanguage"], "en")
        self.assertIn("captions_es", done["steps"])
        self.assertTrue(any(b"hola" in (d or b"") for _, _, d in calls))


@unittest.skipUnless(HAVE, "ffmpeg not installed")
class Compilations(unittest.TestCase):
    def test_episodes_are_joined_at_their_end_screens_with_a_card_between(self):
        from production.compile import CARD_SECONDS, compile_episodes

        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        folders = []
        for k, title in enumerate(("The Night Bus", "Last Stop")):
            folder = root / f"ep{k}"
            folder.mkdir()
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i",
                            "testsrc=size=640x360:rate=24:duration=6", "-f", "lavfi", "-i",
                            "sine=frequency=330:duration=6", "-c:v", "libx264", "-g", "48", "-pix_fmt", "yuv420p",
                            "-c:a", "aac", "-ar", "48000", "-ac", "2", "-shortest", str(folder / "video.mp4")],
                           check=True)
            (folder / "metadata.json").write_text(json.dumps({"title": title, "end_screen_starts": 4.0,
                                                              "sound_credits": ["Track by Kevin MacLeod"]}))
            folders.append(folder)
        report = compile_episodes(folders, root / "out", "Two Stories")
        self.assertEqual(report["stories"], 2)
        self.assertAlmostEqual(report["seconds"], 8.0 + CARD_SECONDS, delta=0.1)
        seconds = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
                                        "csv=p=0", report["video"]], capture_output=True, text=True).stdout)
        self.assertAlmostEqual(seconds, 8.0 + CARD_SECONDS, delta=0.6)
        meta = json.loads((root / "out" / "metadata.json").read_text())
        self.assertIn("0:00 The Night Bus", meta["description"])
        self.assertIn(f"0:0{int(4 + CARD_SECONDS)} Last Stop", meta["description"])
        self.assertEqual(meta["description"].count("Kevin MacLeod"), 1)


if __name__ == "__main__":
    unittest.main()
