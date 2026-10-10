#!/usr/bin/env python3
"""Mutate copied, resealed history receipts to test semantic acceptance gates."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('history',Path(__file__).with_name('verify-nr-history-matrix.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=Path(__file__).resolve().parents[1]/'build/nr_reset_history/receipts-v1'
TARGET=D/'history-repeat-reset10-5a-b/history-repeat-reset10-5a-b-reference.json'
@unittest.skipUnless(TARGET.exists(),'retained history fixture unavailable')
class History(unittest.TestCase):
    def test_positive(self):
        r=m.audit(D);self.assertEqual(r['fresh_processes'],4);self.assertTrue(r['history_continuity_proven']);self.assertFalse(r['mac_temporal_parity'])
    def rejected(self,edit):
        original=m.read
        def altered(path):
            r=original(path)
            if path==TARGET:
                r=copy.deepcopy(r);edit(r);r.pop('receipt_root_sha256');r['receipt_root_sha256']=m.canonical(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(AssertionError):m.audit(D)
    def test_wrong_reset(self):self.rejected(lambda r:r.__setitem__('resets',[1,1]))
    def test_wrong_poison(self):self.rejected(lambda r:r.__setitem__('poison','a5'))
    def test_wrong_process(self):self.rejected(lambda r:r.__setitem__('pid',15268))
    def test_unaudited_payloads(self):self.rejected(lambda r:r.__setitem__('all_inventory_payloads_rehashed_on_rig',False))
    def test_changed_history(self):self.rejected(lambda r:r['history_sha256'].__setitem__('frame-1-history-before.rgba16le','0'*64))
    def test_wrong_boundary(self):self.rejected(lambda r:r['boundaries'][3]['ports'].__setitem__('block22-middle','0'*64))
    def test_wrong_snapshot_binding(self):self.rejected(lambda r:r['boundaries'][2]['snapshot'].__setitem__('delta_sha256','0'*64))
    def test_wrong_sampler(self):self.rejected(lambda r:r['samplers'][0].__setitem__('filter_mode',0))
if __name__=='__main__':unittest.main(verbosity=2)
