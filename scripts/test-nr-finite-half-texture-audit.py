#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('audit', Path(__file__).with_name('audit-nr-finite-half-texture.py'))
m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
ROOT = Path(__file__).resolve().parents[1]
CAPTURES = [ROOT / 'build/nr_tex_finite_half' / n for n in ('a5-a','a5-b','5a-a','5a-b')]
# Frozen pure function used by the accepted audit, not a moving peer checkout.
CANDIDATE = ROOT / 'build/nr_tex_finite_half/audit-v1/candidate.py'

@unittest.skipUnless(CANDIDATE.exists() and (CAPTURES[-1]/'samples.bin').exists(), 'captured fixtures unavailable')
class TextureAudit(unittest.TestCase):
    def test_complete_capture_and_preserved_candidate_failure(self):
        r, _, _, _ = m.audit(CAPTURES, CANDIDATE)
        self.assertEqual(r['queries'], 1381)
        self.assertEqual(r['finite_half_patterns_in_image'], 63488)
        self.assertEqual(r['candidate_unequal_values'], 4986)
        self.assertFalse(r['candidate_exact_on_dataset'])
        self.assertFalse(r['model_parity'])
    def rejected(self, filename, edit):
        original = m.read
        def changed(p):
            r = original(p)
            if p == CAPTURES[-1] / filename:
                r = copy.deepcopy(r); edit(r)
            return r
        with patch.object(m, 'read', side_effect=changed):
            with self.assertRaises(ValueError): m.audit(CAPTURES, CANDIDATE)
    def test_duplicate_pid(self): self.rejected('run.json', lambda r: r.__setitem__('pid', 10908))
    def test_poison(self): self.rejected('run.json', lambda r: r.__setitem__('poison', 165))
    def test_coverage(self): self.rejected('run.json', lambda r: r.__setitem__('exact_write_coverage', False))
    def test_input_mutation(self): self.rejected('run.json', lambda r: r.__setitem__('input_immutable', False))
    def test_incomplete_queries(self): self.rejected('run.json', lambda r: r.__setitem__('sample_queries', 303))
    def test_substituted_image(self): self.rejected('5a-b-receipt.json', lambda r: r['files']['input.rgba16le'].__setitem__('sha256', '0'*64))
    def test_substituted_samples(self): self.rejected('5a-b-receipt.json', lambda r: r['files']['samples.bin'].__setitem__('sha256', '0'*64))
    def test_wrong_binary(self): self.rejected('build.json', lambda r: r.__setitem__('binary_sha256', '0'*64))

if __name__ == '__main__': unittest.main(verbosity=2)
