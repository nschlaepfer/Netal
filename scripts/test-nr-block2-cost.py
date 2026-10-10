#!/usr/bin/env python3
"""Synthetic statistics controls; these are not GPU performance evidence."""
import copy
import importlib.util
import unittest
from pathlib import Path
from unittest.mock import patch

spec=importlib.util.spec_from_file_location('cost',Path(__file__).with_name('audit-nr-block2-cost.py'))
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class CostStatistics(unittest.TestCase):
    def setUp(self):
        self.rows=[{'pair':p,'frame':i,'class':'reset-first' if i==0 else 'temporal-short',
                    'a':100 if i==0 else (10 if i==1 else 90),
                    'b':90 if i==0 else (5 if i==1 else 85)}
                   for p in range(1,5) for i in range(3)]
    def run_stats(self):return m.paired_costs(self.rows,'a','b')
    def test_process_mean_before_ratio(self):
        s=self.run_stats()['temporal-short']
        self.assertEqual(len(s['pairs']),4)
        self.assertEqual(s['baseline_median_ms'],50)
        self.assertEqual(s['candidate_median_ms'],45)
        self.assertAlmostEqual(s['paired_median_reduction_percent'],10)
    def test_reset_separate(self):self.assertEqual(self.run_stats()['reset-first']['baseline_median_ms'],100)
    def test_missing_frame(self):
        self.rows.pop()
        with self.assertRaises(ValueError):self.run_stats()
    def test_duplicate_frame(self):
        self.rows[-1]=copy.deepcopy(self.rows[-2])
        with self.assertRaises(ValueError):self.run_stats()
    def test_invalid_individual_costs(self):
        for bad in (float('nan'),float('inf'),0,-1):
            with self.subTest(bad=bad):
                self.rows[1]['a']=bad
                with self.assertRaises(ValueError):self.run_stats()
    def test_slower_candidate_not_hidden(self):
        for r in self.rows:r['b']=r['a']*2
        s=self.run_stats()['temporal-short']
        self.assertEqual(s['paired_median_reduction_percent'],-100)
        self.assertEqual(s['pairs_candidate_faster'],0)
    def test_pin_inventory(self):
        self.assertEqual(set(m.RECIPE_PINS),{'baseline','candidate'})
        pins=[p for arm in m.RECIPE_PINS.values() for p in arm]
        self.assertEqual(len(set(pins)),8)
        self.assertTrue(all(len(p)==64 and int(p,16)>=0 for p in pins))

PEER=Path(__file__).resolve().parents[1].parent/'NeuralStudio-nr-fable'
RUN=PEER/'build/nr_full/block2cost-v1'

@unittest.skipUnless((RUN/'baseline-p1/process-1/receipt.json').exists(),'requires retained live benchmark')
class LiveAuditControls(unittest.TestCase):
    def reject(self,edit,reason,arm='baseline'):
        target=RUN/f'{arm}-p1/process-1/receipt.json'
        original=m.read
        def changed(path):
            r=original(path)
            if Path(path)==target:
                r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=changed):
            with self.assertRaisesRegex(ValueError,reason):m.audit(PEER,RUN)
    def test_missing_frame(self):self.reject(lambda r:r['frames'].pop(),'native frame inventory')
    def test_missing_write(self):self.reject(lambda r:r['frames'][0]['body'].update(final_pixels_not_written=1),'full output coverage')
    def test_stale_history(self):self.reject(lambda r:r['frames'][1]['body'].update(history_sha256_before='0'*64),'own previous history')
    def test_hidden_omission(self):self.reject(lambda r:r['frames'][0]['body'].update(checks_skipped=[]),'declared production checks')
    def test_missing_provenance(self):
        def edit(r):
            loaded=r['frames'][0]['body']['loaded'];loaded.pop(next(k for k in loaded if k.startswith('file:')))
        self.reject(edit,'complete pinned loaded inventories')
    def test_wrong_baseline_path(self):
        def edit(r):
            phases=r['frames'][0]['encoder_telemetry']['phases']
            next(p for p in phases if p['label']=='block2/allocation_and_poison')['label']='block2/allocation_and_guards'
        self.reject(edit,'expected initialization path')
    def test_missing_candidate_fill(self):
        def edit(r):
            phases=r['frames'][0]['encoder_telemetry']['phases']
            next(p for p in phases if p['label']=='block2/gpu_fill_encoding')['label']='block2/unobserved'
        self.reject(edit,'expected initialization path','candidate')

if __name__=='__main__':unittest.main()
