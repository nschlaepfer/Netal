#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
spec=importlib.util.spec_from_file_location('domain',Path(__file__).with_name('verify-nr-ex2-half-domain.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
D=Path(__file__).resolve().parents[1]/'build/nr_ex2_half';CAPTURES=[D/n for n in ('a5-a','a5-b','5a-a','5a-b')]
@unittest.skipUnless((CAPTURES[-1]/'values.bin').exists(),'full-domain fixture unavailable')
class Domain(unittest.TestCase):
    def test_complete_domain(self):
        table,r=m.audit(CAPTURES);self.assertEqual(len(table),524288);self.assertEqual(r['finite_argument_rows_verified'],63488);self.assertFalse(r['model_executed'])
    def rejected(self,edit):
        original=m.read
        def altered(p):
            r=original(p)
            if p==CAPTURES[-1]/'run.json':r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(AssertionError):m.audit(CAPTURES)
    def test_duplicate_pid(self):self.rejected(lambda r:r.__setitem__('pid',14640))
    def test_wrong_poison(self):self.rejected(lambda r:r.__setitem__('poison',165))
    def test_missing_coverage(self):self.rejected(lambda r:r.__setitem__('coverage_once',False))
    def test_bad_canary(self):self.rejected(lambda r:r.__setitem__('canaries_intact',False))
    def test_wrong_scale(self):self.rejected(lambda r:r.__setitem__('scale_bits','0x3f800000'))
    def test_incomplete_domain(self):self.rejected(lambda r:r.__setitem__('rows',65535))
    def test_model_data_claim(self):self.rejected(lambda r:r.__setitem__('model_executed',True))
if __name__=='__main__':unittest.main(verbosity=2)
