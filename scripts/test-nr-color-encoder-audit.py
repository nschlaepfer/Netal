#!/usr/bin/env python3
"""CPU negative controls for the separate color encoder acceptance auditor."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('verify-nr-color-encoder.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)

class Gates(unittest.TestCase):
    def setUp(self):
        self.good={'result':'pass','pid':123,'poison':165,'coverage_errors':0,'canary_errors':0,'immutable_errors':0}
    def test_valid_memory(self):audit.memory(self.good,165,123)
    def test_failed_status(self):
        r=copy.deepcopy(self.good);r['result']='fail'
        with self.assertRaises(ValueError):audit.memory(r,165)
    def test_each_memory_failure(self):
        for field in ('coverage_errors','canary_errors','immutable_errors'):
            with self.subTest(field=field):
                r=copy.deepcopy(self.good);r[field]=1
                with self.assertRaises(ValueError):audit.memory(r,165)
    def test_missing_gate(self):
        r=copy.deepcopy(self.good);del r['coverage_errors']
        with self.assertRaises(KeyError):audit.memory(r,165)
    def test_wrong_poison(self):
        with self.assertRaises(ValueError):audit.memory(self.good,90)
    def test_wrong_producer(self):
        with self.assertRaises(ValueError):audit.memory(self.good,165,124)
    def test_invalid_pid(self):
        r=copy.deepcopy(self.good);r['pid']=0
        with self.assertRaises(ValueError):audit.memory(r,165)
    def test_raw_coverage(self):
        with tempfile.TemporaryDirectory(prefix='nr-coverage-unit-') as td:
            p=Path(td)/'coverage.bin';np.ones(8,dtype='<u4').tofile(p);audit.coverage(p,8)
            for bad in (0,2,0xa5a5a5a5):
                v=np.ones(8,dtype='<u4');v[3]=bad;v.tofile(p)
                with self.assertRaises(ValueError):audit.coverage(p,8)
    def test_coverage_truncated(self):
        with tempfile.TemporaryDirectory(prefix='nr-coverage-unit-') as td:
            p=Path(td)/'coverage.bin';np.ones(7,dtype='<u4').tofile(p)
            with self.assertRaises(ValueError):audit.coverage(p,8)
    def test_sealed_record_tampering(self):
        with tempfile.TemporaryDirectory(prefix='nr-seal-unit-') as td:
            p=Path(td)/'record.json';r={'result':'pass','case':3};r['receipt_root_sha256']=audit.canonical(r)
            p.write_text(json.dumps(r));self.assertEqual(audit.sealed(p)['case'],3)
            r['case']=0;p.write_text(json.dumps(r))
            with self.assertRaises(ValueError):audit.sealed(p)
    def test_missing_seal(self):
        with tempfile.TemporaryDirectory(prefix='nr-seal-unit-') as td:
            p=Path(td)/'record.json';p.write_text('{"result":"pass"}')
            with self.assertRaises(KeyError):audit.sealed(p)

if __name__=='__main__':unittest.main(verbosity=2)
