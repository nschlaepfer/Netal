#!/usr/bin/env python3
"""CPU regression audit tests; tampered copies are in memory, never on disk."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-half-fma-regression.py'))
m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
REPO=Path(__file__).resolve().parents[1]
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')
MATRIX=PEER/'build/nr_full/regression-halffma-v1/images3'
TARGET=MATRIX/'pilot/receipt.json'


class Contracts(unittest.TestCase):
    def test_all_pinned_contracts(self):
        for name in m.REFERENCES:
            r,n,resets,images,motion,fractional=m.configuration(REPO,name)
            self.assertEqual(len(images),n);self.assertEqual(len(resets),n)
            self.assertEqual(fractional,name in ('uniform3','spatial3'))
    def test_dense_is_not_original(self):
        r,n,resets,images,*_=m.configuration(REPO,'dense')
        self.assertEqual(n,1);self.assertEqual(resets,[1]);self.assertNotEqual(images[0],m.ORIGINAL)
        self.assertEqual(r['outputs'],['153fa0d408035d0f88bf483bedb87ca26e250b8bf6e7927f71c2f799dc03c534'])
    def test_unknown_contract_rejected(self):
        with self.assertRaises(ValueError):m.configuration(REPO,'dense-reset3')


@unittest.skipUnless((MATRIX/'replica-4/receipt.json').exists(),'completed retained matrix unavailable')
class RetainedAudit(unittest.TestCase):
    def test_positive(self):
        r=m.audit(PEER,MATRIX,'images3')
        self.assertEqual(len(r['runs']),12);self.assertEqual(r['encoder_boundary_hashes_checked'],72)
        self.assertFalse(r['production_promotion']);self.assertFalse(r['full_goal_complete'])
    def rejected(self,edit,target=TARGET):
        original=m.read
        def altered(path):
            r=original(path)
            if path==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER,MATRIX,'images3')
    def test_context_recreated(self):self.rejected(lambda r:r.__setitem__('context_preparations',2))
    def test_wrong_second_image(self):self.rejected(lambda r:r['frames'][1].__setitem__('input_file',r['frames'][0]['input_file']))
    def test_wrong_history(self):self.rejected(lambda r:r['frames'][1]['body'].__setitem__('history_sha256_before','0'*64))
    def test_wrong_boundary(self):self.rejected(lambda r:r['frames'][1]['regions']['skip_c32'].__setitem__('sha256','0'*64))
    def test_empty_reference(self):self.rejected(lambda r:r['frames'][1]['reference'].__setitem__('sha256',''))
    def test_wrong_poison(self):self.rejected(lambda r:r['frames'][1].__setitem__('poison',165))
    def test_missing_coverage(self):
        def edit(r):
            launch=next(x for x in r['launches'] if x['kind']!='ptx' and x['launch']!=57 and 'coverage' in x)
            del launch['coverage']
        self.rejected(edit,MATRIX/'pilot/diag/frame-1/chain-run.json')
    def test_extra_omission(self):self.rejected(lambda r:r['frames'][1]['body']['checks_skipped'].append('history comparison'))


if __name__=='__main__':unittest.main(verbosity=2)
