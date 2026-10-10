#!/usr/bin/env python3
"""Read-only verifier controls using restored constants, not model execution."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('recovery', Path(__file__).with_name('verify-nr-recovered-fixtures.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PEER = m.ROOT.parent / 'NeuralStudio-nr-fable'
STAGE = PEER / 'build/nr_body/desc-v31t/production-a5-a/decoder-chain/stage'


class RecoveredConstants(unittest.TestCase):
    def test_all_148(self):
        report = m.audit(PEER, STAGE)
        self.assertEqual(report['matching'], 148)
        self.assertFalse(report['full_model_parity'])
        self.assertFalse(report['live_model_executed'])

    def test_missing_is_not_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            report = m.audit(Path(directory))
        self.assertEqual(report['matching'], 0)
        self.assertEqual(report['result'], 'incomplete_or_mismatched')

    def test_bad_stage_inventory_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'exactly 48'):
                m.audit(PEER, Path(directory))

    def test_changed_constant_is_not_pass(self):
        original = m.sha
        def altered(path):
            return '0' * 64 if path == STAGE / 'weights-133.bin' else original(path)
        with patch.object(m, 'sha', side_effect=altered):
            report = m.audit(PEER, STAGE)
        self.assertEqual(report['matching'], 147)
        self.assertEqual(report['result'], 'incomplete_or_mismatched')

    def test_racing_regeneration_rejected(self):
        original = m.sha
        calls = 0
        def altered(path):
            nonlocal calls
            if path == STAGE / 'weights-133.bin':
                calls += 1
                if calls > 1:
                    return '0' * 64
            return original(path)
        with patch.object(m, 'sha', side_effect=altered):
            with self.assertRaisesRegex(ValueError, 'changed during review'):
                m.audit(PEER, STAGE)


if __name__ == '__main__':
    unittest.main()
