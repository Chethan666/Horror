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

    def test_the_stronger_checker_on_fal_is_asked_with_reasoning(self):
        from story_factory.config import DEFAULTS

        self.assertTrue(DEFAULTS["stills"]["model_args"]["vision_pro"]["reasoning"])

    def test_a_video_to_watch_is_never_sent_to_fals_picture_checker(self):
        from production.media import MediaError
        from production.vertex import Providers

        class Fal:
            asked = []

            def run(self, endpoint, args):
                self.asked.append(endpoint)
                return {}

        class Refused:
            def run(self, endpoint, args):
                raise MediaError("HTTP 403: Request rejected")

        fal = Fal()
        p = Providers(fal, Refused(), {"vertex:pro": ["fal-ai/any-llm/vision"]})
        with self.assertRaises(MediaError):
            p.run("vertex:pro", {"prompt": "watch", "video_url": "data:video/mp4;base64,AA"})
        self.assertEqual(fal.asked, [])
        p.run("vertex:pro", {"prompt": "look", "image_urls": ["data:image/png;base64,AA"]})
        self.assertEqual(fal.asked, ["fal-ai/any-llm/vision"])



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
        on, back = use_vertex(dict(DEFAULTS["stills"], vertex=dict(DEFAULTS["stills"]["vertex"], fallback_to_fal=True)))
        self.assertTrue(back)  # the old behaviour, when asked for

    def test_a_score_cue_is_one_billed_minute(self):
        from story_factory.config import DEFAULTS

        self.assertEqual(DEFAULTS["stills"]["music_max_seconds"], 60)


if __name__ == "__main__":
    unittest.main()
