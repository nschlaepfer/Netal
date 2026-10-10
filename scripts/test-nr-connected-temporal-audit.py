#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
import numpy as np

def module(name,file):
    s=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
connected=module('connected','audit-nr-connected-temporal.py')
counter=module('counter','audit-nr-history-copy-counterexample.py')
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')

class BitComparison(unittest.TestCase):
    def test_signed_zero_not_equal(self):
        self.assertEqual(counter.classify(np.array([0],dtype='<u2'),np.array([0x8000],dtype='<u2')),(1,True))
    def test_nonzero_change_not_misclassified(self):
        self.assertEqual(counter.classify(np.array([0x3c00],dtype='<u2'),np.array([0x3c01],dtype='<u2')),(1,False))
    def test_matching_bits(self):
        a=np.array([0,0x8000,0x3c00,0x7e00],dtype='<u2');self.assertEqual(counter.classify(a,a)[0],0)
    def test_different_extent(self):
        with self.assertRaises(ValueError):counter.classify(np.zeros(1,dtype='<u2'),np.zeros(2,dtype='<u2'))

@unittest.skipUnless((PEER/'build/nr_full/temporal-seq-v3/a5-a5/receipt.json').exists(),'native evidence unavailable')
class CapturedFailure(unittest.TestCase):
    def test_changing_images(self):
        r=connected.audit(PEER,6,0,False,True)
        self.assertTrue(r['changing_images'])
        self.assertEqual(r['encoder_RTX_boundary_hashes_checked'],72)
        self.assertFalse(r['production_promotion'])
    def test_full_fractional_motion(self):
        r=connected.audit(PEER,6,0,True)
        self.assertTrue(r['fractional_motion'])
        self.assertEqual(r['encoder_RTX_boundary_hashes_checked'],72)
        self.assertEqual(len(r['runs']),12)
        self.assertFalse(r['production_promotion'])
    def test_deferred_commit_clean_original(self):
        r=connected.audit(PEER,6)
        self.assertEqual(r['encoder_RTX_boundary_hashes_checked'],72)
        self.assertFalse(r['full_goal_complete'])
    def test_black_temporal(self):
        r=connected.audit(PEER,6,1);self.assertEqual(r['case'],1);self.assertEqual(len(r['runs']),12)
    def test_checker_temporal(self):
        r=connected.audit(PEER,6,2);self.assertEqual(r['case'],2);self.assertEqual(len(r['runs']),12)
    def test_repaired_v4_has_actual_history_continuity(self):
        r=connected.audit(PEER,4)
        self.assertEqual(len(r['runs']),12)
        self.assertEqual(r['frames_per_process'],3)
        self.assertEqual(r['result'],'pass_bounded_two_frame_parity_and_third_call_continuity')
        self.assertFalse(r['production_promotion'])
        self.assertFalse(r['full_goal_complete'])
    def test_all_eight_false_positives_reproduced(self):
        r=counter.audit(PEER);self.assertEqual(len(r['runs']),8);self.assertFalse(r['temporal_history_parity']);self.assertTrue(r['display_output_parity'])
    def test_v2_cannot_pass_from_claimed_history_hash(self):
        with self.assertRaisesRegex(ValueError,'retained post/copy bytes: .*frame-1/tex_out_156.bin; actual=9fa7'):
            connected.audit(PEER,2)
    def test_v3_cannot_pass_from_claimed_history_hash(self):
        with self.assertRaisesRegex(ValueError,'retained post/copy bytes: .*frame-1/tex_out_156.bin; actual=9fa7'):
            connected.audit(PEER,3)

if __name__=='__main__':unittest.main(verbosity=2)
