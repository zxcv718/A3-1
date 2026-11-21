import unittest

import numpy as np

from evaluate import compare_outline


class OutlineEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.paper = np.array([[30, 10], [150, 25], [135, 180], [20, 170]], np.float32)

    def test_comparison_matches_cyclically_shifted_corners(self):
        iou, corner_error = compare_outline(
            np.roll(self.paper, 2, axis=0), self.paper, (200, 200)
        )
        self.assertAlmostEqual(iou, 1.0)
        self.assertAlmostEqual(corner_error, 0.0)

    def test_background_rectangle_has_a_poor_overlap(self):
        background = np.array([[0, 0], [199, 0], [199, 199], [0, 199]], np.float32)
        iou, corner_error = compare_outline(background, self.paper, (200, 200))
        self.assertLess(iou, 0.90)
        self.assertGreater(corner_error, 0.05)


if __name__ == "__main__":
    unittest.main()
