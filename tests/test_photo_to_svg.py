"""Regression tests: synthesize images; never store fixtures in Git."""
from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("engraving_regression", ROOT / "photo_to_svg.py")
photo = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = photo
spec.loader.exec_module(photo)


class OriginalStyleRegression(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        image = np.full((330, 320, 3), 250, np.uint8)
        cv2.ellipse(image, (161, 166), (54, 119), -13, 0, 360, (15, 20, 185), cv2.FILLED)
        cv2.ellipse(image, (149, 162), (12, 103), -11, 0, 360, (51, 67, 222), -1)
        cv2.line(image, (140, 92), (153, 238), (110, 142, 246), 3)
        self.source = self.folder / "cornicello.png"
        photo.write_png(self.source, image)

    def test_original_command_and_transparency(self):
        ns = photo.parse_args([
            "--input", str(self.source), "--output", str(self.folder / "out"),
            "--preset", "balanced", "--style", "accent",
            "--accent-local-density", "4.2", "--outline-style", "cut",
            "--outline-cut-density", "1.6", "--outline-cut-length", "11",
            "--export-transparent", "--svg-background"
        ])
        result = photo.process_one(self.source, ns, photo.validated(ns))
        self.assertGreater(result["accents"], 10)
        self.assertGreater(result["outer"], 0)
        out = self.folder / "out"
        self.assertTrue((out / "cornicello.svg").is_file())
        self.assertTrue((out / "cornicello_art.png").is_file())
        self.assertTrue((out / "cornicello_art_transparent.png").is_file())
        svg = (out / "cornicello.svg").read_text(encoding="utf-8")
        self.assertIn('id="accent-cuts"', svg)
        self.assertIn('id="outer-cuts"', svg)
        rgba = cv2.imread(str(out / "cornicello_art_transparent.png"), cv2.IMREAD_UNCHANGED)
        self.assertEqual(rgba.shape[2], 4)
        self.assertTrue(np.all(rgba[0, 0] == 0))
        self.assertTrue(np.any(rgba[:, :, 3] > 0))

    def test_defaults(self):
        ns = photo.parse_args([])
        self.assertEqual(ns.style, "accent")
        self.assertEqual(ns.accent_local_density, 4.2)
        self.assertEqual(ns.outline_style, "cut")
        self.assertTrue(ns.export_transparent)
        self.assertEqual(ns.background, "#181818")
        self.assertEqual(ns.stroke, "#798388")


if __name__ == "__main__":
    unittest.main()
