"""The review's sound changes: a stereo mix, an original score from Lyria, the channel's theme,
and the host's own voice on ElevenLabs."""

import base64
import io
import json
import shutil
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import soundfile
    HAVE = bool(shutil.which("ffmpeg"))
except ImportError:
    HAVE = False


def tone(path, seconds=2.0, hz=220, rate=44100, stereo=False):
    t = np.arange(int(rate * seconds)) / rate
    wave = (0.3 * np.sin(2 * np.pi * hz * t)).astype(np.float32)
    soundfile.write(path, np.stack([wave, wave * 0.5], axis=1) if stereo else wave, rate)
    return path


class Placement(unittest.TestCase):
    def test_effects_are_placed_and_moving_ones_cross_the_field(self):
        from production.score import pan, position

        self.assertEqual(position("a whisper behind her", 1)[0], "close")
        kind, start, end = position("slow footsteps in the hall", 2)
        self.assertEqual(kind, "moving")
        self.assertEqual((start, end), (-0.6, 0.6))
        self.assertEqual(position("a door on the left creaks", 3)[1], -0.6)
        self.assertEqual(position("a clock ticks", 4)[0], "near")
        mono = np.ones(1000, dtype=np.float32)
        left = pan(mono, -1.0)
        self.assertGreater(left[:, 0].mean(), 0.99)
        self.assertLess(abs(left[:, 1]).mean(), 1e-6)
        centre = pan(mono, 0.0)
        np.testing.assert_allclose(centre[:, 0], centre[:, 1], atol=1e-6)
        self.assertAlmostEqual(float(centre[0, 0] ** 2 + centre[0, 1] ** 2), 1.0, places=5)  # equal loudness
        moving = pan(mono, -1.0, 1.0)
        self.assertGreater(moving[0, 0], moving[0, 1])
        self.assertLess(moving[-1, 0], moving[-1, 1])

    def test_the_mix_check_works_on_a_stereo_mix(self):
        from production.stills.post import voice_over_bed

        rate = 1000
        voice = np.zeros(4000, dtype=np.float32)
        voice[1000:2000] = 0.5
        bed = np.full((4000, 2), 0.25, dtype=np.float32)
        mixed, report = voice_over_bed(voice[:, None] + bed, voice, [(1.0, 2.0)], rate, 12.0)
        self.assertEqual(mixed.shape, (4000, 2))
        self.assertAlmostEqual(report["voice_over_bed_db"], 6.0, delta=0.1)
        self.assertAlmostEqual(report["after_db"], 12.0, delta=0.1)

    def test_pieces_of_music_are_crossfaded_into_one_cue(self):
        from production.score import _joined, _loop

        a = np.ones((3000, 2), dtype=np.float32)
        b = np.ones((3000, 2), dtype=np.float32) * 0.5
        joined = _joined([a, b], 1000, seconds=1.0)
        self.assertEqual(joined.shape, (5000, 2))
        self.assertAlmostEqual(float(joined[0, 0]), 1.0)
        self.assertAlmostEqual(float(joined[-1, 0]), 0.5)
        self.assertEqual(_loop(a[:100], 250).shape, (250, 2))
        self.assertEqual(_loop(np.ones(100, dtype=np.float32), 250).shape, (250,))


@unittest.skipUnless(HAVE, "audio libraries or ffmpeg not installed")
class Stereo(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def test_the_mix_is_stereo_with_the_voice_in_the_centre_and_effects_to_the_sides(self):
        from production.score import Score
        from production.shots import Shot

        rate = 8000
        score = Score(None, self.root / "w", rate)
        voice = np.zeros(rate * 12, dtype=np.float32)
        voice[rate:rate * 3] = 0.3 * np.sin(np.arange(rate * 2) / 3)
        shots = [Shot(index=0, scene=0, t0=0, t1=6, text="a", sound="slow footsteps in the hall"),
                 Shot(index=1, scene=0, t0=6, t1=12, text="b", scare=True, tension=5, sound="a door on the left")]
        out = score.mix(voice, [(1.0, 3.0)], shots, 0.0, [{"mood": "dread", "location": "the hall"}], [], 12.0)
        self.assertEqual(out.shape, (rate * 12, 2))
        self.assertGreater(float(np.abs(out[:, 0] - out[:, 1]).max()), 0.001)  # not the same in both ears
        self.assertEqual(score.report["stereo"], {"moving": 1, "side": 1})  # footsteps cross, the door is left

    def test_a_stereo_file_is_decoded_as_two_channels(self):
        from production.speech import decode

        path = tone(self.root / "s.wav", stereo=True)
        both = decode(path, 22050, channels=2)
        self.assertEqual(both.ndim, 2)
        self.assertEqual(both.shape[1], 2)
        self.assertGreater(float(np.abs(both[:, 0]).mean()), float(np.abs(both[:, 1]).mean()))
        self.assertEqual(decode(path, 22050).ndim, 1)


@unittest.skipUnless(HAVE, "audio libraries or ffmpeg not installed")
class OriginalScore(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, ignore_errors=True)

    def studio(self, fal, models):
        from production.paid import Budget, Studio

        return Studio(fal, Budget(5.0, {"vertex:lyria-002": 0.06}), self.root / "cache", {"models": models})

    def test_each_cue_is_two_lyria_pieces_with_the_storys_motif(self):
        from production.score import Score

        piece = tone(self.root / "piece.wav", 3.0, rate=48000, stereo=True)
        asked = []

        class Lyria:
            def run(self, endpoint, args):
                asked.append((endpoint, args))
                return {"audio": {"url": piece.as_uri()}}

        studio = self.studio(Lyria(), {"music": "vertex:lyria-002"})
        score = Score(studio, self.root / "w", 24000)
        score.cue_names, score.motif, score.seed = ["dread"], "a music box winding down", 11
        score.prefer_made = {"music_dread"}
        cues = score.music(60)
        self.assertEqual(len(asked), 2)
        self.assertTrue(all(e == "vertex:lyria-002" for e, _ in asked))
        self.assertIn("a music box winding down", asked[0][1]["prompt"])
        self.assertIn("vocals", asked[0][1]["negative_prompt"])
        self.assertNotEqual(asked[0][1]["seed"], asked[1][1]["seed"])  # two different pieces
        self.assertEqual(cues["dread"].ndim, 2)
        self.assertGreater(len(cues["dread"]), 24000 * 4)  # two 3 s pieces, crossfaded
        self.assertEqual([g["name"] for g in score.generated], ["music_dread", "music_dread_2"])
        self.assertAlmostEqual(studio.budget.spent, 0.12, places=3)

    def test_if_lyria_fails_the_librarys_track_plays(self):
        from production.media import MediaError
        from production.score import Score

        track = tone(self.root / "track.wav", 2.0, stereo=True)

        class Down:
            def run(self, endpoint, args):
                raise MediaError("HTTP 404: model not found")

        score = Score(self.studio(Down(), {"music": "vertex:lyria-002"}), self.root / "w", 24000)
        score.cue_names, score.prefer_made, score.chosen = ["dread"], {"music_dread"}, {"music_dread": track}
        cues = score.music(60)
        self.assertIn("dread", cues)
        self.assertEqual(score.generated, [])

    def test_the_theme_plays_under_the_ritual_line(self):
        from production.score import Score
        from production.shots import Shot

        rate = 8000
        score = Score(None, self.root / "w", rate)
        score.theme = np.ones((rate, 2), dtype=np.float32) * 0.2
        score.theme_at = [4.0]
        voice = np.zeros(rate * 10, dtype=np.float32)
        shots = [Shot(index=0, scene=0, t0=0, t1=10, text="The lantern is lit.")]
        out = score.mix(voice, [], shots, 0.0, [{"mood": "calm", "location": ""}], [], 10.0)
        self.assertEqual(score.report["theme"], 1)
        before, during = np.abs(out[int(2.0 * rate):int(3.0 * rate)]).mean(), np.abs(out[int(4.0 * rate):int(4.5 * rate)]).mean()
        self.assertGreater(during, before)

    def test_theme_candidates_are_trimmed_and_faded(self):
        from production.stills import theme

        wav = io.BytesIO()
        soundfile.write(wav, np.full((48000 * 12, 2), 0.5, dtype=np.float32), 48000, format="WAV")
        data = base64.b64encode(wav.getvalue()).decode()

        class Lyria:
            def run(self, endpoint, args):
                assert endpoint == "vertex:lyria-002" and "music box" in args["prompt"]
                return {"audio": {"url": f"data:audio/wav;base64,{data}"}}

        report = theme.make({"theme_file": "production/assets/theme/lantern.wav"}, Lyria(), self.root, 2)
        self.assertEqual(report["candidates"], ["theme_1.wav", "theme_2.wav"])
        samples, rate = soundfile.read(self.root / "theme_1.wav")
        self.assertAlmostEqual(len(samples) / rate, 7.0, delta=0.05)
        self.assertLess(abs(samples[-1]).max(), 0.01)  # faded out


class VertexMusic(unittest.TestCase):
    def test_lyria_is_asked_through_predict_and_moves_to_a_location_that_serves_it(self):
        from production.media import MediaError
        from production.vertex import VertexKey

        urls = []

        def http(url, method="GET", body=None, timeout=0, headers=None):
            urls.append(url)
            if "/locations/global/" in url:
                raise MediaError(f"POST {url}: HTTP 404: Publisher model lyria-002 not found")
            assert url.endswith("lyria-002:predict") and body["instances"][0]["negative_prompt"] == "vocals"
            return {"predictions": [{"bytesBase64Encoded": "UklGRg==", "mimeType": "audio/wav"}]}

        v = VertexKey(key="k", project="123", http=http)
        got = v.run("vertex:lyria-002", {"prompt": "a cello drone", "negative_prompt": "vocals", "seed": 3})
        self.assertEqual(got["audio"]["url"], "data:audio/wav;base64,UklGRg==")
        self.assertIn("us-central1-aiplatform.googleapis.com/v1/projects/123/locations/us-central1/", urls[-1])
        v.run("vertex:lyria-002", {"prompt": "again", "negative_prompt": "vocals"})
        self.assertNotIn("/locations/global/", urls[-1])  # the location that served it is kept

    def test_veo_is_a_long_job_asked_and_then_fetched(self):
        from production.vertex import VertexKey

        calls = []

        def http(url, method="GET", body=None, timeout=0, headers=None):
            calls.append(url.rsplit(":", 1)[1])
            if url.endswith(":predictLongRunning"):
                instance = body["instances"][0]
                assert instance["image"]["mimeType"] == "image/png" and body["parameters"]["durationSeconds"] == 8
                assert body["parameters"]["generateAudio"] is False
                return {"name": "projects/123/locations/us-central1/publishers/google/models/veo/operations/7"}
            if len(calls) < 3:
                return {"name": "op", "done": False}
            return {"done": True, "response": {"videos": [{"bytesBase64Encoded": "AAAA", "mimeType": "video/mp4"}]}}

        v = VertexKey(key="k", project="123", http=http)
        v.POLL_SECONDS = 0.0
        got = v.run("vertex:veo-3.1-fast-generate-001", {"prompt": "she turns", "image_url": "data:image/png;base64,QQ==",
                                                        "duration": "8"})
        self.assertEqual(got["video"]["url"], "data:video/mp4;base64,AAAA")
        self.assertEqual(calls, ["predictLongRunning", "fetchPredictOperation", "fetchPredictOperation"])


class OwnVoice(unittest.TestCase):
    def test_the_hosts_voice_is_read_by_elevenlabs_api_with_a_name_looked_up_once(self):
        from production.elevenlabs import ElevenLabs

        seen = []

        def http(url, method="GET", body=None, headers=None, timeout=0):
            seen.append((method, url.split("?")[0], body))
            assert headers["xi-api-key"] == "secret"
            if url.endswith("/voices"):
                return json.dumps({"voices": [{"name": "Brian - deep narrator", "voice_id": "nPczCjzI2devNBz1zQrb"}]}).encode()
            return b"ID3audio"

        el = ElevenLabs(key="secret", http=http)
        got = el.run("elevenlabs:eleven_v3", {"text": "The lantern is lit.", "voice": "Brian", "stability": 0.5})
        self.assertEqual(base64.b64decode(got["audio"]["url"].split(",", 1)[1]), b"ID3audio")
        self.assertEqual(seen[1][1], "https://api.elevenlabs.io/v1/text-to-speech/nPczCjzI2devNBz1zQrb")
        self.assertEqual(seen[1][2], {"text": "The lantern is lit.", "model_id": "eleven_v3",
                                      "voice_settings": {"stability": 0.5}})
        el.run("elevenlabs:eleven_v3", {"text": "again", "voice": "AbCdEfGhIjKlMnOpQrSt"})  # an ID is used as it is
        self.assertEqual([m for m, u, _ in seen].count("GET"), 1)

    def test_other_services_are_reached_by_their_prefix(self):
        from production.vertex import Providers

        class One:
            def __init__(self, name):
                self.name = name

            def run(self, endpoint, args):
                return {"by": self.name}

        router = Providers(One("fal"), None, {}, {"elevenlabs:": One("elevenlabs")})
        self.assertEqual(router.run("elevenlabs:eleven_v3", {})["by"], "elevenlabs")
        self.assertEqual(router.run("fal-ai/elevenlabs/tts/eleven-v3", {})["by"], "fal")

    def test_without_a_key_the_host_keeps_the_stock_voice(self):
        import os
        from unittest import mock

        from production.__main__ import media_client
        from story_factory.config import DEFAULTS

        stills = dict(DEFAULTS["stills"], host_voice={"elevenlabs": "Brian", "kokoro": "am_michael",
                                                      "elevenlabs_id": "AbCdEfGhIjKlMnOpQrSt"})
        clean = {k: "" for k in ("ELEVENLABS_API_KEY", "FAL_KEY", "GEMINI_PLATFORM_API_KEY",
                                 "GCP_SERVICE_ACCOUNT_JSON", "GCP_ACCESS_TOKEN")}
        with mock.patch.dict(os.environ, clean):
            cfg, client = media_client(stills)
        self.assertNotIn("host_voice", cfg["models"])
        self.assertIsNone(client)


if __name__ == "__main__":
    unittest.main()
