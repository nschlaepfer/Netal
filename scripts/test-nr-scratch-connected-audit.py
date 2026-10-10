#!/usr/bin/env python3
"""Read-only connected audit regression and negative receipt controls."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('scratch',Path(__file__).with_name('audit-nr-scratch-connected.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')
TARGET=PEER/'build/nr_full/bwc-scratch-v1/p1-a5-forward/receipt.json'
@unittest.skipUnless(TARGET.exists(),'connected fixture unavailable')
class Connected(unittest.TestCase):
    def test_positive(self):
        r=m.audit(PEER);self.assertEqual(len(r['runs']),16);self.assertTrue(r['correctness_promotion_supported']);self.assertFalse(r['performance_promotion_supported'])
    def rejected(self,edit):
        original=m.read
        def altered(p):
            r=original(p)
            if p==TARGET:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER)
    def test_old_binary(self):self.rejected(lambda r:r.__setitem__('executable_sha256','0'*64))
    def test_wrong_descriptor(self):self.rejected(lambda r:r.__setitem__('body_descriptor_sha256','0'*64))
    def test_zero_coverage(self):self.rejected(lambda r:r['frames'][0]['encoder'].__setitem__('coverage_values_checked',0))
    def test_ignored_failures(self):self.rejected(lambda r:r['frames'][0]['body'].__setitem__('check_failures',['coverage']))
    def test_unchanged_pixels_wrong_regions(self):self.rejected(lambda r:r['frames'][0]['regions']['entry'].__setitem__('sha256','0'*64))
    def test_missing_constant_rehash(self):self.rejected(lambda r:r['frames'][0]['body'].__setitem__('constants_rehashed_after_frame',152))
    def test_model_hash_changed(self):self.rejected(lambda r:r['frames'][0]['body']['loaded'].__setitem__(next(iter(r['frames'][0]['body']['loaded'])),'0'*64))
    def test_missing_post_identity(self):self.rejected(lambda r:r['post_run_file_identity'].pop(next(iter(r['post_run_file_identity']))))
if __name__=='__main__':unittest.main(verbosity=2)
