#!/usr/bin/env python3
"""Read-only regression controls: corrupted metadata must not pass parity audit."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-inmemory-matrix.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')
MATRIX=PEER/'build/nr_full/matrix-v1'

class ByteComparison(unittest.TestCase):
    def test_equal_and_single_byte_counterexample(self):
        with tempfile.TemporaryDirectory(prefix='nr-audit-unit-') as td:
            a,b=Path(td)/'a',Path(td)/'b'
            for p in (a,b):
                with p.open('wb') as f:f.truncate(16588800)
            audit.identical(a,b)
            with b.open('r+b') as f:f.seek(16588799);f.write(b'\x01')
            with self.assertRaisesRegex(ValueError,'bytes differ'):audit.identical(a,b)
    def test_wrong_extent(self):
        with tempfile.TemporaryDirectory(prefix='nr-audit-unit-') as td:
            a,b=Path(td)/'a',Path(td)/'b';a.write_bytes(b'\0');b.write_bytes(b'\0')
            with self.assertRaisesRegex(ValueError,'extent'):audit.identical(a,b)

@unittest.skipUnless((MATRIX/'summary.json').is_file(),'retained local six-run fixture unavailable')
class RetainedMatrixControls(unittest.TestCase):
    def rejects(self,edit):
        original=audit.read
        target=MATRIX/'dense-5a-1/receipt.json'
        def changed(path):
            r=original(path)
            if path==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(audit,'read',side_effect=changed):
            with self.assertRaises(ValueError):audit.audit(PEER,MATRIX)
    def test_reference_fixture_stays_unpromoted(self):
        result=audit.audit(PEER,MATRIX)
        self.assertEqual(result['result'],'observed_final_parity');self.assertFalse(result['strict_promotion']);self.assertFalse(result['goal_complete'])
    def test_failed_body_with_matching_pixels(self):self.rejects(lambda r:r['body'].__setitem__('code',2))
    def test_failed_encoder_with_matching_pixels(self):self.rejects(lambda r:r['encoder'].__setitem__('code',2))
    def test_texture_hole_with_matching_pixels(self):self.rejects(lambda r:r['body'].__setitem__('final_pixels_not_written',1))
    def test_texture_multiwrite_with_matching_pixels(self):self.rejects(lambda r:r['body'].__setitem__('final_pixels_written_more_than_once',1))
    def test_zero_pixel_count(self):self.rejects(lambda r:r['body'].__setitem__('final_pixels',0))
    def test_changed_input(self):self.rejects(lambda r:r['body'].__setitem__('input_texture_immutable',False))
    def test_wrong_image_digest(self):self.rejects(lambda r:r.__setitem__('input_sha256','0'*64))
    def test_changed_producer_region(self):self.rejects(lambda r:r.__setitem__('producer_regions_unchanged_by_body',False))
    def test_wrong_region_offset(self):self.rejects(lambda r:r['regions']['entry'].__setitem__('offset',0))
    def test_wrong_region_size(self):self.rejects(lambda r:r['regions']['entry'].__setitem__('bytes',1105919))
    def test_wrong_model_identity(self):self.rejects(lambda r:r['encoder'].__setitem__('bundle_root','0'*64))
    def test_missing_encoder_checks(self):self.rejects(lambda r:r['encoder'].__setitem__('coverage_values_checked',0))
    def test_wrong_poison(self):self.rejects(lambda r:r.__setitem__('poison',165))

if __name__=='__main__':unittest.main(verbosity=2)
