#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest

s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-resident-v13-checkpoints.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)


class CheckpointContract(unittest.TestCase):
    def setUp(self):
        self.expected={(i,'pool'):f'{i:064x}' for i in range(133,157)}
        self.expected[25,'out']='a'*64
    def test_complete(self):self.assertEqual(m.require_all_checkpoints(self.expected,dict(self.expected)),([],[]))
    def test_no_partial_success(self):
        with self.assertRaises(AssertionError):m.require_all_checkpoints(self.expected,{(25,'out'):'a'*64})
    def test_missing_one_decoder_checkpoint(self):
        actual=dict(self.expected);actual.pop((155,'pool'))
        with self.assertRaises(AssertionError):m.require_all_checkpoints(self.expected,actual)
    def test_changed_checkpoint(self):
        actual=dict(self.expected);actual[155,'pool']='0'*64
        with self.assertRaises(AssertionError):m.require_all_checkpoints(self.expected,actual)
    def test_extra_observation_does_not_hide_missing(self):
        actual={(25,'out'):'a'*64,(58,'extra'):'b'*64}
        with self.assertRaises(AssertionError):m.require_all_checkpoints(self.expected,actual)
    def test_extra_observation_allowed(self):
        actual=dict(self.expected);actual[58,'extra']='b'*64
        self.assertEqual(m.require_all_checkpoints(self.expected,actual),([],[]))


if __name__=='__main__':unittest.main(verbosity=2)
