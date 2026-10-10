#!/usr/bin/env python3
"""Synthetic inventory controls, not model evidence."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('availability', Path(__file__).with_name('audit-nr-evidence-availability.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Availability(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.artifact = self.root / 'synthetic.bin'
        self.artifact.write_bytes(b'synthetic inventory test only')
        self.receipt = self.root / 'receipt.json'
        self.data = {'pins': {str(self.artifact): m.sha(self.artifact)}, 'result': 'test-only'}
        self.save()

    def save(self):
        data = dict(self.data)
        data['receipt_root_sha256'] = m.canonical(data)
        self.receipt.write_text(json.dumps(data))

    def test_present(self):
        self.assertTrue(m.inspect(self.receipt)['all_pins_available_and_matching'])

    def test_missing_is_not_pass(self):
        self.artifact.unlink()
        result = m.inspect(self.receipt)
        self.assertFalse(result['all_pins_available_and_matching'])
        self.assertEqual(result['counts']['missing'], 1)

    def test_changed_is_not_pass(self):
        self.artifact.write_bytes(b'changed')
        self.assertEqual(m.inspect(self.receipt)['counts']['mismatched'], 1)

    def test_unreadable_not_missing(self):
        with patch.object(m, 'sha', side_effect=PermissionError):
            self.assertEqual(m.inspect(self.receipt)['counts']['unreadable'], 1)

    def test_bad_seal_rejected(self):
        self.receipt.write_text(self.receipt.read_text().replace('test-only', 'tampered'))
        with self.assertRaisesRegex(ValueError, 'seal mismatch'):
            m.inspect(self.receipt)

    def test_empty_inventory_rejected(self):
        self.data['pins'] = {}
        self.save()
        with self.assertRaisesRegex(ValueError, 'no pinned'):
            m.inspect(self.receipt)

    def test_relative_path_rejected(self):
        self.data['pins'] = {'synthetic.bin': m.sha(self.artifact)}
        self.save()
        with self.assertRaisesRegex(ValueError, 'relative pin'):
            m.inspect(self.receipt)

    def test_invalid_digest_rejected(self):
        self.data['pins'][str(self.artifact)] = 'x' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'invalid sha256'):
            m.inspect(self.receipt)


if __name__ == '__main__':
    unittest.main()
