#!/usr/bin/env python3
"""CPU-only negative controls for prefix identity and delta parsing."""
import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

def load(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
ids=load('identities','audit-nr-prefix-identities.py');delta=load('deltas','build-rtx-oracle-manifest.py')
class IdentityTests(unittest.TestCase):
    def setUp(self):self.lines=['runtime.process_id=123','runtime.color_case=3','runtime.output_poison=a5']
    def test_valid(self):self.assertEqual(ids.identity(self.lines,3,'a5'),123)
    def test_missing(self):
        with self.assertRaises(ValueError):ids.identity(self.lines[:-1],3,'a5')
    def test_duplicate(self):
        with self.assertRaises(ValueError):ids.identity(self.lines+self.lines[:1],3,'a5')
    def test_wrong_case(self):
        with self.assertRaises(ValueError):ids.identity(self.lines,0,'a5')
    def test_wrong_poison(self):
        with self.assertRaises(ValueError):ids.identity(self.lines,3,'5a')
    def test_nonpositive_pid(self):
        with self.assertRaises(ValueError):ids.identity(['runtime.process_id=0']+self.lines[1:],3,'a5')
class DeltaTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nr-prefix-test-');self.addCleanup(self.tmp.cleanup);self.path=Path(self.tmp.name)/'snapshot.delta'
        self.header=struct.pack('<8sIIQII',b'NSDLTA1\0',2,3,32,65536,1)
        self.payload=self.header+struct.pack('<II',0,4)+b'ABCD'
    def parse(self,data):self.path.write_bytes(data);return delta.parse_snapshot_delta(self.path,2,None)
    def test_valid(self):
        state,r=self.parse(self.payload);self.assertEqual(state,bytearray(b'ABCD'+b'\0'*28));self.assertEqual(r['allocation_id'],'allocation_003')
    def test_bad_magic(self):
        with self.assertRaises(delta.OracleError):self.parse(b'BADMAGIC'+self.payload[8:])
    def test_wrong_launch(self):
        with self.assertRaises(delta.OracleError):self.parse(self.payload[:8]+struct.pack('<I',1)+self.payload[12:])
    def test_outside_allocation(self):
        with self.assertRaises(delta.OracleError):self.parse(self.header+struct.pack('<II',1,4)+b'ABCD')
    def test_truncated(self):
        with self.assertRaises(delta.OracleError):self.parse(self.payload[:-1])
    def test_trailing(self):
        with self.assertRaises(delta.OracleError):self.parse(self.payload+b'X')
if __name__=='__main__':unittest.main()
