#!/usr/bin/env python3
"""CPU-only inventory controls; synthetic receipts never enter model execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('pair',Path(__file__).with_name('audit-nr-live-checked-pair.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class InventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nr-inventory-test-')
        self.addCleanup(self.tmp.cleanup)
        r=Path(self.tmp.name)
        self.plan=m.paired_plan(r/'baseline',r/'resident',3,2,r/'base-complement',r/'resident-complement')
        for tag,b,c,poisons in self.plan:
            for d in (b,c):
                p=d/tag/'receipt.json';p.parent.mkdir(parents=True)
                p.write_text(json.dumps({'frames_in_process':3,'frames':[{'frame':i,'poison':v} for i,v in enumerate(poisons)]}))
        self.target=self.plan[-1][2]/self.plan[-1][0]/'receipt.json'

    def mutate(self,fn):
        r=json.loads(self.target.read_text());fn(r);self.target.write_text(json.dumps(r))
        with self.assertRaises(ValueError):m.preflight_plan(self.plan,3)

    def test_complete_union(self):
        m.preflight_plan(self.plan,3)
        self.assertEqual([{p[3][i] for p in self.plan} for i in range(3)],[{165,90}]*3)
    def test_base_scope_gap_visible(self):
        self.assertEqual({p[3][0] for p in self.plan[:2]},{165})
    def test_missing_complement_receipt(self):
        self.target.unlink()
        with self.assertRaises(ValueError):m.preflight_plan(self.plan,3)
    def test_missing_frame(self):self.mutate(lambda r:r['frames'].pop())
    def test_duplicate_frame(self):self.mutate(lambda r:r['frames'][2].update(frame=1))
    def test_wrong_poison(self):self.mutate(lambda r:r['frames'][0].update(poison=165))
    def test_wrong_declared_count(self):self.mutate(lambda r:r.update(frames_in_process=2))
    def test_empty_inventory(self):self.mutate(lambda r:r.update(frames=[],frames_in_process=0))
    def test_unexpected_replica(self):
        (self.plan[0][1]/'checked-extra').mkdir()
        with self.assertRaises(ValueError):m.preflight_plan(self.plan,3)
    def test_one_sided_complement(self):
        with self.assertRaises(ValueError):m.paired_plan(Path('a'),Path('b'),3,2,Path('c'))
    def test_undeclared_replica_count(self):
        with self.assertRaises(ValueError):m.paired_plan(Path('a'),Path('b'),3,3)
    def test_no_zero_frames(self):
        with self.assertRaises(ValueError):m.paired_plan(Path('a'),Path('b'),0,2)
    def test_complement_cannot_overwrite_base(self):
        with self.assertRaises(ValueError):m.paired_plan(Path('a'),Path('b'),3,2,Path('a'),Path('d'))

if __name__=='__main__':unittest.main()
