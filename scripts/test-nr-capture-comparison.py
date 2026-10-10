#!/usr/bin/env python3
"""CPU-only comparison tests; tiny invented fixtures are not model evidence."""
import hashlib
import importlib.util
import json
import pathlib
import tempfile
import unittest

spec = importlib.util.spec_from_file_location('compare',
    pathlib.Path(__file__).with_name('compare-nr-frame-captures.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='nr-capture-compare-test-')
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def fixture(self, name, value=0, mutate=None):
        root = self.root/name
        root.mkdir()
        record = {'schema': 'netal.nr.live-frame-capture.v1',
                  'gpu_completed': True, 'model_identity': 'TEST-NOT-A-MODEL',
                  'arithmetic': 'test', 'reset_only': True,
                  'game_extent': [1, 1], 'model_extent': [1, 1],
                  'model_content': [0, 0, 1, 1], 'files': []}
        for plane in sorted(module.PLANES):
            raw = bytes([value])*4
            (root/plane).write_bytes(raw)
            record['files'].append({'file': plane, 'width': 1, 'height': 1,
                'bytes_per_pixel': 4, 'metal_pixel_format': 92, 'bytes': 4,
                'sha256': hashlib.sha256(raw).hexdigest()})
        if mutate:
            mutate(record)
        path = root/'capture.json'
        path.write_text(json.dumps(record))
        return path

    def test_equal_is_not_acceptance(self):
        r = module.compare(self.fixture('a'), self.fixture('b'))
        self.assertTrue(r['all_plane_bytes_equal'])
        self.assertTrue(r['execution_contract_metadata_equal'])
        self.assertFalse(r['visual_acceptance'])
        self.assertFalse(r['rtx_parity_acceptance'])
        self.assertFalse(r['live_game_acceptance'])

    def test_differences_reported_not_hidden(self):
        r = module.compare(self.fixture('a'), self.fixture('b', 1))
        self.assertFalse(r['all_plane_bytes_equal'])
        for p in r['planes'].values():
            self.assertEqual(p['mismatching_bytes'], 4)
            self.assertEqual(p['first_mismatching_byte'], 0)

    def test_missing_metadata_is_not_comparable(self):
        r = module.compare(self.fixture('a', mutate=lambda r: r.pop('model_identity')),
                           self.fixture('b', mutate=lambda r: r.pop('model_identity')))
        self.assertFalse(r['execution_contract_metadata_equal'])

    def test_different_color_contract_is_not_comparable(self):
        r = module.compare(self.fixture('a'), self.fixture('b', mutate=lambda r:r.update(color_mode='bounded-reinhard-srgb',color_reconstruction='inverse-reinhard-srgb')))
        self.assertTrue(r['all_plane_bytes_equal'])
        self.assertFalse(r['execution_contract_metadata_equal'])

    def test_hash_mismatch_rejected(self):
        p = self.fixture('a')
        (p.parent/'game-input.bin').write_bytes(b'bad!')
        with self.assertRaisesRegex(ValueError, 'SHA256'):
            module.read_capture(p)

    def test_missing_plane_rejected(self):
        p = self.fixture('a', mutate=lambda r: r['files'].pop())
        with self.assertRaisesRegex(ValueError, 'four planes'):
            module.read_capture(p)

    def test_duplicate_and_path_traversal_rejected(self):
        for i, name in enumerate(('game-input.bin', '../capture.json')):
            p = self.fixture(str(i), mutate=lambda r: r['files'][-1].update(file=name))
            with self.assertRaisesRegex(ValueError, 'unknown/duplicate'):
                module.read_capture(p)

    def test_nonterminal_capture_rejected(self):
        p = self.fixture('a', mutate=lambda r: r.update(gpu_completed=False))
        with self.assertRaisesRegex(ValueError, 'completion'):
            module.read_capture(p)

    def test_bool_dimension_rejected(self):
        p = self.fixture('a', mutate=lambda r: r['files'][0].update(width=True))
        with self.assertRaisesRegex(ValueError, 'non-integer'):
            module.read_capture(p)


if __name__ == '__main__':
    unittest.main()
