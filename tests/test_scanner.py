from pathlib import Path
import unittest

import cv2
import numpy as np

from scanner import (
    ScanSettings,
    detect_edges,
    order_corners,
    preprocess,
    scan,
    warp_document,
)


def sample_page():
    image = np.full((360, 480, 3), 35, np.uint8)
    corners = np.array([[135, 65], [360, 95], [400, 295], [100, 315]], np.int32)
    cv2.fillConvexPoly(image, corners, (240, 240, 240))
    cv2.putText(
        image, "SCAN", (160, 210), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (20, 20, 20), 4
    )
    return image, corners


class ScannerTest(unittest.TestCase):
    def test_preprocess_converts_bgr_and_spreads_a_bright_pixel(self):
        image = np.zeros((5, 5, 3), dtype=np.uint8)
        image[2, 2] = (0, 0, 255)

        gray, blurred = preprocess(image, 3)

        self.assertEqual(gray.shape, (5, 5))
        self.assertEqual(blurred.shape, gray.shape)
        self.assertEqual(gray.dtype, np.uint8)
        self.assertEqual(blurred.dtype, np.uint8)
        self.assertEqual(gray[2, 2], 76)
        self.assertEqual(blurred[2, 2], 19)
        self.assertEqual(gray[2, 1], 0)
        self.assertGreater(blurred[2, 1], 0)

    def test_detect_edges_marks_the_brightness_boundary(self):
        blurred = np.full((24, 32), 20, dtype=np.uint8)
        blurred[:, 16:] = 240

        edges = detect_edges(blurred, 25, 80)

        self.assertEqual(edges.shape, blurred.shape)
        self.assertEqual(edges.dtype, np.uint8)
        self.assertEqual(set(np.unique(edges)), {0, 255})
        rows, columns = np.nonzero(edges)
        np.testing.assert_array_equal(rows, np.arange(24))
        self.assertTrue(np.all((15 <= columns) & (columns <= 16)))

    def test_orders_shuffled_points_even_when_two_sums_match(self):
        points = np.array([[100, 25], [175, 100], [100, 175], [25, 100]])
        np.testing.assert_array_equal(order_corners(points[[3, 2, 0, 1]]), points)

    def test_rejects_duplicate_and_collinear_corners(self):
        for points in (
            [(2, 2), (2, 2), (12, 2), (2, 12)],
            [(2, 2), (5, 5), (8, 8), (12, 12)],
        ):
            with self.subTest(points=points), self.assertRaises(ValueError):
                order_corners(np.array(points))

    def test_warps_known_corners_to_matching_rectangle_corners(self):
        image, corners = sample_page()
        for marker, point in enumerate(corners, start=1):
            cv2.circle(image, tuple(point), 3, (marker * 40,) * 3, -1)
        warped = warp_document(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), corners)
        self.assertEqual(warped.shape, (252, 301))
        self.assertEqual(
            [
                int(warped[0, 0]),
                int(warped[0, -1]),
                int(warped[-1, -1]),
                int(warped[-1, 0]),
            ],
            [40, 80, 120, 160],
        )

    def test_scan_finds_a_page_and_produces_binary_result(self):
        image, corners = sample_page()
        result = scan(image)
        if result.corners is None or result.scanned is None:
            self.fail("사다리꼴 문서의 네 모서리와 스캔본이 필요합니다.")
        np.testing.assert_allclose(result.corners, corners, atol=4)
        self.assertEqual(set(np.unique(result.scanned)), {0, 255})
        self.assertFalse(np.array_equal(result.original, result.overlay))

    def test_no_candidate_returns_current_stages_without_a_scan(self):
        result = scan(np.full((320, 480, 3), 127, np.uint8))
        self.assertIsNone(result.corners)
        self.assertIsNone(result.scanned)
        self.assertEqual(result.edges.shape, (320, 480))

    def test_initial_settings_detect_a_faint_boundary_in_a_real_photo(self):
        path = (
            Path(__file__).resolve().parents[1]
            / "evaluation"
            / "images"
            / "front_02_letter001_frame_0072.jpeg"
        )
        image = cv2.imread(str(path))
        if image is None:
            self.fail(f"평가 사진을 읽을 수 없습니다. {path}")
        self.assertIsNotNone(scan(image).scanned)

    def test_invalid_blur_and_thresholds_are_rejected(self):
        for options in (
            {"blur": 4},
            {"canny_low": 150, "canny_high": 150},
            {"canny_high": 256},
        ):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ScanSettings(**options)


if __name__ == "__main__":
    unittest.main()
