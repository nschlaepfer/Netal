#!/usr/bin/env python3
import copy,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-sampler-edge-rtx.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
R=Path(__file__).resolve().parents[1]/'build/nr_sampler_edges/rtx-v1';F=R.parent/'input-v1';H='49b11a8750ff8e60b0754997b84df1adb978877465b2951a3cf5fc0fca3475ad'
class AuditTests(unittest.TestCase):
    def reject(self,edit,path=None):
        target=path or R/'job-receipt.json';old=m.read
        def changed(p):
            r=old(p)
            if p==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=changed):
            with self.assertRaises(ValueError):m.audit(R,F,H)
    def test_positive_scope(self):
        r=m.audit(R,F,H);self.assertEqual(len(r['runs']),4);self.assertFalse(r['rtx_mac_parity'])
    def test_missing_run(self):self.reject(lambda r:r['runs'].pop())
    def test_wrong_driver(self):self.reject(lambda r:r.update(gpu='another driver'))
    def test_source_drift(self):self.reject(lambda r:r['sources_after'].update({'nr_sampler_edge_inputs.h':'0'*64}))
    def test_wrong_texture_identity(self):self.reject(lambda r:r['artifacts']['a5-a/input-after.rgba16le'].update(sha256='0'*64))
    def test_duplicate_pid(self):
        pid=m.read(R/'a5-a/run.json')['pid'];self.reject(lambda r:r.update(pid=pid),R/'a5-b/run.json')
    def test_wrong_poison(self):self.reject(lambda r:r.update(poison=90),R/'a5-a/run.json')
    def test_guard_failure(self):self.reject(lambda r:r.update(guards_intact=False),R/'a5-a/run.json')
if __name__=='__main__':unittest.main()
