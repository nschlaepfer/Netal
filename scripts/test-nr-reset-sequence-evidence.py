#!/usr/bin/env python3
"""Offline receipt mutation tests; never alter captured model evidence."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('seq',Path(__file__).with_name('retrieve-nr-reset-sequence.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=Path(__file__).resolve().parents[1]/'build/nr_reset_sequence/pilot-reset11-a5-compressed-v2'
@unittest.skipUnless((D/'verification.json').exists(),'complete pilot fixture unavailable')
class Evidence(unittest.TestCase):
    def test_positive(self):
        r=m.audit(D);self.assertEqual(r['result'],'audited_observation');self.assertTrue(r['repeat_output_equal']);self.assertTrue(r['graph_equal']);self.assertFalse(r['temporal_acceptance'])
    def rejected(self,name,edit):
        original=m.read
        def altered(p):
            r=original(p)
            if p==D/name:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(D)
    def test_modified_runtime(self):self.rejected('capture-receipt.json',lambda r:r.__setitem__('runtime_file_sha256','0'*64))
    def test_missing_repoison(self):self.rejected('capture-receipt.json',lambda r:r.__setitem__('output_repoisoned_each_evaluation',False))
    def test_mutated_input(self):self.rejected('capture-receipt.json',lambda r:r.__setitem__('input_arrays_immutable',False))
    def test_wrong_handle_contract(self):self.rejected('capture-receipt.json',lambda r:r.__setitem__('same_feature_handle',False))
    def test_wrong_first_reset(self):self.rejected('job-receipt.json',lambda r:r.__setitem__('resets',[0,1]))
    def test_failed_execution(self):self.rejected('job-receipt.json',lambda r:r.__setitem__('exit_code',1))
if __name__=='__main__':unittest.main(verbosity=2)
