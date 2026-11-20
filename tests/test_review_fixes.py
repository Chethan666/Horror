"""Fixes from watching run 113 (the 2-minute test episode, Room Six)."""

import unittest

from production import edit
from production import shots as shotlist
from production.shots import Shot


def story(n: int = 20, seconds: float = 5.0) -> list[Shot]:
    return [Shot(index=i, scene=i // 2, t0=i * seconds, t1=(i + 1) * seconds, text=f"line {i}") for i in range(n)]


class ColdOpen(unittest.TestCase):
    def test_the_twist_near_the_end_never_opens_the_video(self):
        beats = story()
        beats[15].scare = True  # the reveal: "and the name was mine", at three quarters of the story
        beats[15].importance = 5
        beats[9].importance = 4  # the figure behind her, in the middle
        cold = edit.pick_cold_open(beats, [{"key_moment": True}] * 10, 6.0, 14.0)
        self.assertNotIn(15, cold)
        self.assertEqual(cold[0], 9)

    def test_the_hosts_closing_frame_does_not_count_as_the_story(self):
        beats = story()
        beats[11].importance = 5
        host = {0, 18, 19}  # the outro's scene would otherwise be the "final scene" excluded
        cold = edit.pick_cold_open(beats, [{}] * 10, 6.0, 14.0, host=host)
        self.assertTrue(cold and all(i not in host for i in cold))
        self.assertLess(beats[cold[0]].t0, 0.65 * 100)

    def test_the_threat_seen_outranks_a_scare_flag_on_an_empty_place(self):
        beats = story()
        beats[5].scare, beats[5].threat, beats[5].tension, beats[5].importance = True, 2, 5, 5  # a boarded door
        beats[11].threat, beats[11].tension, beats[11].importance = 3, 5, 5  # the figure over her shoulder
        self.assertEqual(edit.pick_cold_open(beats, [{"key_moment": True}] * 10, 4.0, 14.0)[0], 11)

    def test_a_short_story_still_gets_a_cold_open(self):
        beats = story(6)
        cold = edit.pick_cold_open(beats, [{}] * 3, 6.0, 14.0)
        self.assertTrue(cold)


class IndoorAir(unittest.TestCase):
    def test_no_rain_or_fog_across_an_office_or_a_screen(self):
        self.assertEqual(shotlist.outdoor_air(["rain", "flicker"], "June frozen at the desk", "the motel office"),
                         ["flicker"])
        self.assertEqual(shotlist.outdoor_air(["fog"], "a figure in the dark", "", "screen"), [])

    def test_weather_stays_outdoors(self):
        self.assertEqual(shotlist.outdoor_air(["rain", "fog"], "an empty two-lane road at night"), ["rain", "fog"])
        self.assertTrue(shotlist.outside("Wide exterior: a motel in fog, one lit window"))



class Sentences(unittest.TestCase):
    def test_titles_and_times_do_not_end_a_sentence(self):
        from production import timing

        text = "Mr. Pell came in at six. Room six. Two fourteen a.m. And the name was mine. At 3 a.m. she woke."
        self.assertEqual([c.text for c in timing.split_chunks(text)],
                         ["Mr. Pell came in at six.", "Room six.", "Two fourteen a.m.", "And the name was mine.",
                          "At 3 a.m. she woke."])



class Run118(unittest.TestCase):
    def test_the_climax_is_where_the_threat_is_shown_and_it_may_be_seen(self):
        from production.stills.storyboard import ladder

        beats = story(20)
        beats[12].scare, beats[12].threat, beats[12].importance = True, 2, 5  # "I looked at the camera"
        beats[15].threat, beats[15].importance, beats[15].tension = 3, 5, 5  # "leaning over my shoulder"
        report = ladder(beats, set())
        self.assertEqual(report["climax"], 15)
        self.assertEqual(beats[15].threat, 4)  # no longer refused as "shown too clearly"

    def test_the_cold_open_teases_and_never_shows_the_full_reveal(self):
        beats = story(20)
        beats[8].threat, beats[8].tension, beats[8].importance = 2, 5, 4
        beats[9].threat, beats[9].tension, beats[9].importance = 4, 5, 5  # the reveal, same scene as 8
        cold = edit.pick_cold_open(beats, [{}] * 10, 6.0, 14.0)
        self.assertNotIn(9, cold)
        self.assertEqual(cold[0], 8)

    def test_a_repaint_can_name_its_beat_by_the_words_of_its_line(self):
        from production.replay import parse_redo, redo_targets

        beats = story(6)
        beats[4].text = "And behind me, close enough to breathe on my neck"
        redo = parse_redo("@close_enough_to_breathe:show_the_figure,2,@no_such_words,9")
        self.assertEqual(redo, {"close enough to breathe": "show the figure", 2: "", "no such words": "", 9: ""})
        targets, warnings = redo_targets(redo, beats)
        self.assertEqual(targets, {4: "show the figure", 2: ""})
        self.assertEqual(len(warnings), 2)


class GoogleOnly(unittest.TestCase):
    def test_pictures_and_checks_have_no_fal_fallback_and_elevenlabs_makes_the_score(self):
        from production.vertex import use_vertex
        from story_factory.config import DEFAULTS

        cfg, fallback = use_vertex(DEFAULTS["stills"])
        self.assertEqual(fallback, {})
        for role in ("reference", "sheet", "shot", "shot_alt", "hero", "thumbnail", "vision", "vision_pro"):
            self.assertTrue(cfg["models"][role].startswith("vertex:"), role)
        for role in ("voice", "sfx", "music"):
            self.assertIn("elevenlabs", cfg["models"][role], role)

    def test_a_score_cue_is_one_billed_minute(self):
        from story_factory.config import DEFAULTS

        self.assertEqual(DEFAULTS["stills"]["music_max_seconds"], 60)



class GoodStanding(unittest.TestCase):
    """Staying in good standing with Google (the project was blocked on 19 November 2026)."""

    def test_google_keeps_its_default_safety_level(self):
        from production import vertex

        self.assertTrue(all(c["threshold"] == "BLOCK_MEDIUM_AND_ABOVE" for c in vertex.SAFETY))
        vertex.set_safety("BLOCK_OFF_EVERYTHING")  # not a level: ignored
        self.assertTrue(all(c["threshold"] == "BLOCK_MEDIUM_AND_ABOVE" for c in vertex.SAFETY))

    def test_a_terms_block_stops_every_google_request_at_once(self):
        from production.media import MediaError
        from production.vertex import Providers

        class Blocked:
            asked = 0

            def run(self, endpoint, args):
                self.asked += 1
                raise MediaError('HTTP 403: { "error": { "message": "Request rejected in accordance with our terms '
                                 'of use and policies" } }')

        google = Blocked()
        p = Providers(None, google, {})
        for _ in range(5):
            with self.assertRaises(MediaError):
                p.run("vertex:nb", {"prompt": "x"})
        self.assertEqual(google.asked, 1)
        self.assertIn("terms", p.stopped)

    def test_repeated_failures_and_the_request_cap_stop_google(self):
        from production.media import MediaError
        from production.vertex import Providers

        class Broken:
            asked = 0

            def run(self, endpoint, args):
                self.asked += 1
                raise MediaError("HTTP 500: internal")

        google = Broken()
        p = Providers(None, google, {})
        p.max_failures = 3
        for _ in range(6):
            with self.assertRaises(MediaError):
                p.run("vertex:nb", {"prompt": "x"})
        self.assertEqual(google.asked, 3)

        class Fine:
            def run(self, endpoint, args):
                return {"output": "ok"}

        q = Providers(None, Fine(), {})
        q.max_requests = 2
        q.run("vertex:c", {})
        q.run("vertex:c", {})
        with self.assertRaises(MediaError):
            q.run("vertex:c", {})

    def test_a_picture_refused_for_its_content_is_never_asked_for_again(self):
        import tempfile
        from pathlib import Path

        from production.stills.images import Painter
        from production.vertex import PolicyRefusal

        class Budget:
            cap = 0.0

            @staticmethod
            def price(model, units=1.0):
                return 0.05

        class Studio:
            models = {"shot": "vertex:nb", "shot_alt": "vertex:nb"}
            budget = Budget()
            asked = 0

        studio = Studio()
        with tempfile.TemporaryDirectory() as root:
            painter = Painter(studio, {}, Path(root), critic=None)
            painter.available = lambda step: True

            def refuse(step, text, refs, raw, key, retry=0):
                studio.asked += 1
                raise PolicyRefusal("vertex nb: no picture, content refused or empty (finish: IMAGE_SAFETY)")

            painter._one = refuse
            out = Path(root) / "beat_005.png"
            self.assertEqual(painter.paint("x", [], out, "k", ["model:shot", "model:shot_alt"], check=None),
                             (None, "refused"))
            self.assertEqual(studio.asked, 1)  # not handed to the second painter
            fix = Path(root) / "beat_005_fix1234.png"  # the art review or a redo asking again
            self.assertEqual(painter.paint("y", [], fix, "k2", ["model:shot"], check=None), (None, "refused"))
            self.assertEqual(studio.asked, 1)
            self.assertIn("beat_005", painter.refused)

    def test_prompts_are_said_as_a_film_would_say_them(self):
        from production.stills.wording import soften
        from story_factory.config import DEFAULTS

        style = DEFAULTS["stills"]["looks"]["cinematic"]["style"]
        self.assertEqual(soften(style), style)  # the house look, "never gory" and the blood-red accent, untouched
        self.assertEqual(soften("close enough to breathe on my neck"), "standing close behind")
        self.assertEqual(soften("She screams as he grabs her"), "She gasps as he reaches for her")
        self.assertEqual(soften("a blood droplet on the page, never gory"), "a dark red droplet on the page, never gory")

    def test_a_child_is_never_in_a_frightening_frame(self):
        from production.stills.storyboard import keep_children_safe

        beats = story(4)
        beats[1].characters, beats[1].threat, beats[1].scare = ["Lily"], 2, True
        beats[2].picture, beats[2].tension = "the baby monitor on the table, its light red", 5  # no child in it
        beats[3].characters, beats[3].tension = ["Lily"], 2  # calm: unchanged
        changed = keep_children_safe(beats, [{"name": "Lily", "look": "a four-year-old girl"}], set())
        self.assertEqual(changed, [1])
        self.assertEqual((beats[1].threat, beats[1].scare), (0, False))
        self.assertIn("nothing threatening shares the frame", beats[1].picture)


    def test_without_a_google_sign_in_no_picture_goes_to_fal(self):
        import os
        from unittest import mock

        from production.__main__ import media_client
        from story_factory.config import DEFAULTS

        env = {k: v for k, v in os.environ.items()
               if k not in ("GEMINI_PLATFORM_API_KEY", "GCP_SERVICE_ACCOUNT_JSON", "ELEVENLABS_API_KEY")}
        env["FAL_KEY"] = "test"
        with mock.patch.dict(os.environ, env, clear=True):
            stills, _ = media_client(dict(DEFAULTS["stills"]))
        left = {k: v for k, v in stills["models"].items() if v}
        self.assertTrue(left and all("elevenlabs" in v for v in left.values()), left)


if __name__ == "__main__":
    unittest.main()
