import unittest
from preprocess import task_segments


class EventTests(unittest.TestCase):
    def test_both_classes_and_terminal_types(self):
        rows = task_segments([(0, 1000), (512, 769), (1280, 7691), (3840, 7692),
                              (5000, 1000), (5512, 770), (6280, 7701), (6792, 7693)])
        self.assertEqual([r["label"] for r in rows], [0, 1])
        self.assertEqual(rows[1]["end"] - rows[1]["start"], 512)

    def test_short_trial_retained_in_qa(self):
        rows = task_segments([(1, 769), (769, 7691), (1000, 7693)])
        self.assertLess(rows[0]["end"] - rows[0]["start"], 512)

    def test_label_mismatch_fails(self):
        with self.assertRaises(ValueError):
            task_segments([(1, 769), (769, 7701), (1500, 7702)])

    def test_missing_end_fails(self):
        with self.assertRaises(ValueError):
            task_segments([(1, 769), (769, 7691), (1500, 1000)])

    def test_new_task_before_end_fails(self):
        with self.assertRaises(ValueError):
            task_segments([(1, 769), (769, 7691), (1500, 769)])

    def test_cue_cannot_cross_trial_boundary(self):
        with self.assertRaises(ValueError):
            task_segments([(10, 769), (20, 1000), (1000, 7691), (1512, 7693)])

    def test_duplicate_cue_fails(self):
        with self.assertRaises(ValueError):
            task_segments([(10, 769), (20, 769), (1000, 7691), (1512, 7693)])


if __name__ == "__main__":
    unittest.main()
