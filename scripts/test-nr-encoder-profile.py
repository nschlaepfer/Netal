#!/usr/bin/env python3
"""CPU-only negative controls for the timing summary, not model acceptance."""
import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('profile_summary',Path(__file__).with_name('analyze-nr-encoder-profile.py'))
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
builder_spec=importlib.util.spec_from_file_location('profile_builder',Path(__file__).with_name('build-nr-profiled-encoder.py'))
builder=importlib.util.module_from_spec(builder_spec);builder_spec.loader.exec_module(builder)

class PhaseTests(unittest.TestCase):
    def test_repeated_phases_sum(self):
        self.assertEqual(module.summarize_phases([
            {'label':'block1/allocation','wall_ms':1.25},
            {'label':'block1/wait','wall_ms':5},
            {'label':'block1/allocation','wall_ms':2.5}]),
            {'block1/allocation':3.75,'block1/wait':5})
    def test_legacy_no_phases(self):self.assertEqual(module.summarize_phases([]),{})
    def test_zero_duration(self):self.assertEqual(module.summarize_phases([{'label':'a','wall_ms':0}]),{'a':0})
    def test_invalid_durations(self):
        for value in (-1,float('nan'),float('inf'),-float('inf'),'1'):
            with self.subTest(value=value),self.assertRaises(ValueError):module.summarize_phases([{'label':'a','wall_ms':value}])
    def test_invalid_labels(self):
        for label in ('',None,3):
            with self.subTest(label=label),self.assertRaises(ValueError):module.summarize_phases([{'label':label,'wall_ms':1}])

class FillTests(unittest.TestCase):
    def test_exact_source_transform(self):
        self.assertEqual(builder.single_pass_fill('prefix '+builder.OLD_FILL+' suffix'),'prefix '+builder.NEW_FILL+' suffix')
    def test_missing_or_duplicate_implementation_rejected(self):
        for source in ('',builder.OLD_FILL*2,builder.NEW_FILL):
            with self.subTest(source=source),self.assertRaises(ValueError):builder.single_pass_fill(source)
    def test_all_poison_values_and_boundary_sizes(self):
        # Three disjoint intervals partition [0, payload+2*guard). The two
        # algorithms must agree for every prior byte, including zero payload.
        guard=256
        for size in (0,1,2,255,256,257,511,512,1023,4096):
            for poison in range(256):
                old=bytearray([0xc3])*(size+2*guard)
                old[guard:guard+size]=bytes([poison])*size
                new=bytearray([poison^255])*(size+2*guard)
                new[:guard]=bytes([0xc3])*guard
                new[guard+size:]=bytes([0xc3])*guard
                new[guard:guard+size]=bytes([poison])*size
                self.assertEqual(old,new,(size,poison))

if __name__=='__main__':unittest.main()
