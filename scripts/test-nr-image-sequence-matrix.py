#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('matrix',Path(__file__).with_name('verify-nr-image-sequence-matrix.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=Path(__file__).resolve().parents[1]/'build/nr_image_sequence/receipts-v1'
TARGET=D/'mixed-reset100-5a-b/mixed-reset100-5a-b-reference.json'
@unittest.skipUnless(TARGET.exists(),'retained fixtures unavailable')
class Matrix(unittest.TestCase):
    def test_positive(self):self.assertEqual(m.audit(D)['evaluations'],12)
    def rejected(self,edit):
        original=m.read
        def altered(path):
            r=original(path)
            if path==TARGET:
                r=copy.deepcopy(r);edit(r);r.pop('receipt_root_sha256');r['receipt_root_sha256']=m.canonical(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(D)
    def test_wrong_case(self):self.rejected(lambda r:r.__setitem__('color_case',1))
    def test_wrong_reset(self):self.rejected(lambda r:r.__setitem__('resets',[1,1,1]))
    def test_wrong_poison(self):self.rejected(lambda r:r.__setitem__('poison','a5'))
    def test_duplicate_process(self):self.rejected(lambda r:r.__setitem__('pid',35800))
    def test_missing_payload_audit(self):self.rejected(lambda r:r.__setitem__('all_inventory_payloads_rehashed_on_rig',False))
    def test_third_history(self):self.rejected(lambda r:r['history_sha256'].__setitem__('frame-2-history-after.rgba16le','0'*64))
    def test_third_encoder(self):self.rejected(lambda r:r['boundaries'][5]['ports'].__setitem__('block22-middle','0'*64))
    def test_snapshot_binding(self):self.rejected(lambda r:r['boundaries'][4]['snapshot'].__setitem__('delta_sha256','0'*64))
    def test_graph_substitution(self):self.rejected(lambda r:r['graph_roots'].__setitem__(2,'0'*64))
    def test_sampler_substitution(self):self.rejected(lambda r:r['samplers'][0].__setitem__('filter_mode',0))
    def test_wrong_image_order(self):self.rejected(lambda r:r.__setitem__('color_cases',[0,1,2]))
    def test_wrong_input_sequence(self):self.rejected(lambda r:r['input_hashes'].__setitem__(1,r['input_hashes'][0]))
if __name__=='__main__':unittest.main(verbosity=2)
