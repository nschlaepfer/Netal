#!/usr/bin/env python3
"""CPU-only mutations of copied receipt dictionaries; original evidence untouched."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('refresh',Path(__file__).with_name('audit-nr-reset-reference-refresh.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
REPO=Path(__file__).resolve().parents[1]
MATRIX=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt/build/nr_full/bwc-v1')
BLACK=REPO/'build/nr_color_encoder_reference/reference-1-v1';CHECKER=REPO/'build/nr_color_encoder_reference/reference-2-retry-v2'
@unittest.skipUnless((CHECKER/'retrieval.json').exists(),'retained reference fixture unavailable')
class Refresh(unittest.TestCase):
    def test_positive(self):
        r=m.audit(MATRIX,BLACK,CHECKER);self.assertEqual(r['boundary_hashes_matched'],48);self.assertFalse(r['new_gpu_execution']);self.assertFalse(r['full_goal_complete'])
    def rejected(self,edit,target=None):
        target=target or MATRIX/'p1-a5-forward/receipt.json';original=m.read
        def altered(p):
            r=original(p)
            if p==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(MATRIX,BLACK,CHECKER)
    def test_bad_live_region(self):self.rejected(lambda r:r['frames'][0]['regions']['entry'].__setitem__('sha256','0'*64))
    def test_body_failure_with_matching_pixels(self):self.rejected(lambda r:r['frames'][0]['body'].__setitem__('check_failures',['constant changed']))
    def test_skipped_encoder_coverage(self):self.rejected(lambda r:r['frames'][0]['encoder'].__setitem__('coverage_values_checked',0))
    def test_missing_final_write(self):self.rejected(lambda r:r['frames'][0]['body'].__setitem__('final_pixels_not_written',1))
    def test_output_used_as_runtime_input(self):self.rejected(lambda r:r['frames'][0]['reference'].__setitem__('cpu_comparison_only',False))
    def test_lost_region_lifetime(self):self.rejected(lambda r:r['frames'][0].__setitem__('producer_regions_unchanged_by_body',False))
    def test_duplicate_process(self):self.rejected(lambda r:r.__setitem__('pid',46431))
    def test_rtx_input_immutable_lie(self):self.rejected(lambda r:r.__setitem__('input_arrays_immutable',False),BLACK/'identities/a5-a/capture-receipt.json')
    def test_rtx_changed_runtime(self):self.rejected(lambda r:r.__setitem__('runtime_file_sha256','0'*64),BLACK/'identities/a5-a/capture-receipt.json')
if __name__=='__main__':unittest.main(verbosity=2)
