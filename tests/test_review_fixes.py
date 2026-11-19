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


if __name__ == "__main__":
    unittest.main()
