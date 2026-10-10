#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-sixteen-native.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt');TARGET=PEER/'build/nr_full/temporal-sixteen-halffma-v1/replica-1-pilot/receipt.json'
@unittest.skipUnless((PEER/'build/nr_full/temporal-sixteen-halffma-v1/replica-4-5a-a5/receipt.json').exists(),'retained native fixture unavailable')
class Sixteen(unittest.TestCase):
    def test_positive(self):
        r=m.audit(PEER);self.assertEqual(len(r['runs']),64);self.assertEqual(r['encoder_boundary_hashes_checked'],384);self.assertFalse(r['full_goal_complete'])
    def rejected(self,edit):
        original=m.read
        def altered(p):
            r=original(p)
            if p==TARGET:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER)
    def test_context_recreated(self):self.rejected(lambda r:r.__setitem__('context_preparations',2))
    def test_counter_restarted_mid_sequence(self):self.rejected(lambda r:r['frames'][4]['body'].__setitem__('frames_since_reset',1))
    def test_late_history_substitution(self):self.rejected(lambda r:r['frames'][7]['body'].__setitem__('history_sha256_after','0'*64))
    def test_wrong_poison(self):self.rejected(lambda r:r['frames'][3].__setitem__('poison',165))
    def test_diagnostic_ingress(self):self.rejected(lambda r:r['frames'][12]['encoder'].__setitem__('diagnostics_dir','archived'))
if __name__=='__main__':unittest.main(verbosity=2)
