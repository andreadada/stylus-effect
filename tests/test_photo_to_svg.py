"""Generate synthetic fixtures in memory; no sample images are committed."""
import tempfile
import unittest
from pathlib import Path
import xml.etree.ElementTree as ET

import cv2
import numpy as np

import photo_to_svg as engrave


class EngravingTests(unittest.TestCase):
    def setUp(self):
        self.args = engrave.build_parser().parse_args([])
        self.image = np.full((280, 240, 3), 250, np.uint8)
        cv2.ellipse(self.image, (115, 145), (48, 98), 21, 0, 360, (27, 42, 170), -1)
        cv2.ellipse(self.image, (98, 127), (7, 66), 23, 0, 360, (80, 110, 235), -1)

    def test_default_style(self):
        self.assertEqual(self.args.style, 'accent')
        self.assertEqual(self.args.accent_local_density, 4.2)
        self.assertEqual(self.args.outline_style, 'cut')
        self.assertTrue(self.args.export_transparent)
        legacy = engrave.build_parser().parse_args([
            '--input', 'input/cornicello.jpg', '--output', 'output',
            '--preset', 'balanced', '--style', 'accent',
            '--accent-local-density', '4.2', '--outline-style', 'cut',
            '--outline-cut-density', '1.6', '--outline-cut-length', '11',
            '--export-transparent', '--svg-background',
        ])
        engrave.validate(legacy)
        self.assertTrue(legacy.export_transparent)
        self.assertTrue(legacy.svg_background)

    def test_white_segment_and_select_one_component(self):
        mask = engrave.make_mask(self.image, self.args)
        self.assertEqual(mask.shape, (280, 240))
        self.assertGreater(cv2.countNonZero(mask), 8000)
        self.assertEqual(mask[0, 0], 0)

    def test_larger_local_density_stays_same_zone_count(self):
        a = engrave.build_parser().parse_args(['--accent-local-density', '1'])
        b = engrave.build_parser().parse_args(['--accent-local-density', '5'])
        ma = engrave.make_mask(self.image, a)
        first = engrave._accent_marks(self.image, ma, a, np.random.default_rng(a.seed))
        second = engrave._accent_marks(self.image, ma, b, np.random.default_rng(b.seed))
        self.assertGreaterEqual(len(second), len(first))
        self.assertLess(len(second), 200)

    def test_deterministic(self):
        a, _ = engrave.render_artwork(self.image, self.args)
        b, _ = engrave.render_artwork(self.image, self.args)
        self.assertEqual(len(a.layers), len(b.layers))
        for x, y in zip(a.layers, b.layers):
            self.assertEqual(len(x[1]), len(y[1]))
            self.assertTrue(all(np.array_equal(p, q) for p, q in zip(x[1], y[1])))

    def test_transparent_and_svg(self):
        art, _ = engrave.render_artwork(self.image, self.args)
        raster = engrave.transparent_png(art, self.args)
        self.assertEqual(raster.shape, (280, 240, 4))
        self.assertEqual(raster[0, 0, 3], 0)
        self.assertGreater(raster[:, :, 3].max(), 0)
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'test.svg'
            engrave.export_svg(target, art, self.args)
            doc = ET.parse(target)
            self.assertEqual(doc.getroot().tag.split('}')[-1], 'svg')
            self.assertFalse(any(e.tag.endswith('rect') for e in doc.getroot()))

    def test_process_creates_files(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / 'input.png'
            engrave.write_png(source, self.image)
            args = engrave.build_parser().parse_args(['--input', str(source), '--output', str(base / 'out')])
            result = engrave.process(source, args)
            self.assertTrue(result['svg'].is_file())
            self.assertTrue(result['png'].is_file())
            self.assertTrue((base / 'out' / 'input_preview.png').is_file())
            self.assertTrue((base / 'out' / 'input_art_transparent.png').is_file())
            self.assertTrue((base / 'out' / 'input_art.png').is_file())
            self.assertTrue((base / 'out' / 'input_transparent.png').is_file())

    def test_disable_transparent_export(self):
        with tempfile.TemporaryDirectory() as d:
            base = Path(d)
            source = base / 'input.png'
            engrave.write_png(source, self.image)
            args = engrave.build_parser().parse_args([
                '--input', str(source), '--output', str(base / 'out'),
                '--no-export-transparent',
            ])
            result = engrave.process(source, args)
            self.assertIsNone(result['png'])
            self.assertFalse((base / 'out' / 'input_art_transparent.png').exists())
            self.assertFalse((base / 'out' / 'input_transparent.png').exists())
            self.assertTrue((base / 'out' / 'input_preview.png').exists())

    def test_validation(self):
        for flag, value in [('--accent-local-density', '9'), ('--outline-jitter', '-1'), ('--stroke', 'red')]:
            args = engrave.build_parser().parse_args([flag, value])
            with self.assertRaises(ValueError):
                engrave.validate(args)


if __name__ == '__main__':
    unittest.main()