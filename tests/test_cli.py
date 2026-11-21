from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from test_scanner import sample_page


MAIN = Path(__file__).resolve().parents[1] / "main.py"


class CommandLineTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)

    def invoke(self, *arguments):
        return subprocess.run(
            [sys.executable, str(MAIN), "--no-gui", *map(str, arguments)],
            cwd=self.directory,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_headless_scan_accepts_a_korean_filename_and_saves_png_and_jpg(self):
        image, _ = sample_page()
        source = self.directory / "촬영한 종이.png"
        cv2.imencode(".png", image)[1].tofile(source)
        for extension in ("png", "jpg"):
            target = self.directory / f"스캔본.{extension}"
            result = self.invoke(source, "--output", target)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(
                cv2.imdecode(np.fromfile(target, np.uint8), cv2.IMREAD_GRAYSCALE)
            )

    def test_missing_and_corrupt_input_are_reported(self):
        corrupt = self.directory / "손상.png"
        corrupt.write_text("invalid image")
        for path in (self.directory / "missing.png", corrupt):
            with self.subTest(path=path):
                result = self.invoke(path)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn("읽을 수 없", result.stderr)

    def test_no_candidate_creates_no_scan_but_can_save_histogram(self):
        source = self.directory / "blank.png"
        cv2.imwrite(str(source), np.zeros((128, 128, 3), np.uint8))
        histogram = self.directory / "histogram.png"
        result = self.invoke(source, "--histogram", histogram)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("문서 후보", result.stderr)
        self.assertFalse((self.directory / "blank_scanned.png").exists())
        self.assertTrue(histogram.is_file())

    def test_refuses_to_overwrite_input(self):
        source = self.directory / "page.png"
        cv2.imwrite(str(source), sample_page()[0])
        original = source.read_bytes()
        result = self.invoke(source, "--output", source)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("입력 파일", result.stderr)
        self.assertEqual(source.read_bytes(), original)

    def test_reports_an_unwritable_output(self):
        source = self.directory / "page.png"
        cv2.imwrite(str(source), sample_page()[0])
        target = self.directory / "folder.png"
        target.mkdir()
        result = self.invoke(source, "--output", target)
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("저장할 수 없", result.stderr)


if __name__ == "__main__":
    unittest.main()
