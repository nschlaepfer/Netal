#!/usr/bin/env python3
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-resident-live.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')
MATRIX=PEER/'build/nr_full/gate1-v27t/images3';TARGET=MATRIX/'replica-1-pilot/receipt.json'


class BackendContracts(unittest.TestCase):
    def test_v17_zero_pinned(self):self.assertEqual(m.backend_contract(17,False)[0::3],('v31t',12))
    def test_v13_unchanged(self):self.assertEqual(m.backend_contract(13,True)[0::3],('v27t-sp',10))
    def test_unknown_rejected(self):
        with self.assertRaises(ValueError):m.backend_contract(18,False)
    def test_reviewed_v17_spatial(self):self.assertEqual(m.backend_contract(17,True)[0::3],('v31t-sp',12))

@unittest.skipUnless((MATRIX/'replica-4/receipt.json').exists(),'completed fixture unavailable')
class ResidentLive(unittest.TestCase):
    def rejected(self,edit,target=TARGET):
        original=m.read
        def changed(path):
            r=original(path)
            if path==target:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=changed):
            with self.assertRaises(ValueError):m.audit(PEER,MATRIX,'images3')
    def test_positive_scope(self):
        r=m.audit(PEER,MATRIX,'images3');self.assertEqual(len(r['runs']),12)
        self.assertFalse(r['strong_live_intermediate_acceptance']);self.assertFalse(r['production_promotion'])
    def test_wrong_image(self):self.rejected(lambda r:r['frames'][1].__setitem__('input_file',r['frames'][0]['input_file']))
    def test_wrong_history(self):self.rejected(lambda r:r['frames'][1]['body'].__setitem__('history_sha256_before','0'*64))
    def test_wrong_port(self):self.rejected(lambda r:r['frames'][1]['regions']['skip_c32'].__setitem__('sha256','0'*64))
    def test_context_recreated(self):self.rejected(lambda r:r.__setitem__('context_preparations',2))
    def test_missing_reference(self):self.rejected(lambda r:r['frames'][1]['reference'].__setitem__('sha256',''))
    def test_incomplete_validator(self):self.rejected(lambda r:r['frames'][1]['resident_evidence'].__setitem__('validators_run',125))
    def test_extra_command_buffer(self):self.rejected(lambda r:r['frames'][1]['inventory'].__setitem__('command_buffers',2))
    def test_hidden_omission(self):self.rejected(lambda r:r['frames'][1]['body'].__setitem__('checks_skipped',[]))
    def test_preimage_mislabel(self):self.rejected(lambda r:r['frames'][1]['resident_evidence'].__setitem__('decoder_pre_image_source','gpu-snapshot'))
    def test_poisoned_validator_record(self):
        self.rejected(lambda r:r['launches'][1]['validators'][0].__setitem__('done',False),MATRIX/'replica-1-pilot/diag/frame-1/chain-run.json')


V17_MATRIX=PEER/'build/nr_full/gate1-v31t/black3'
@unittest.skipUnless((V17_MATRIX/'replica-4/receipt.json').exists(),'completed v17 fixture unavailable')
class ResidentLiveV17(unittest.TestCase):
    def rejected(self,edit):
        original=m.read
        def changed(path):
            r=original(path)
            if path==V17_MATRIX/'replica-1-pilot/receipt.json':r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=changed):
            with self.assertRaises(ValueError):m.audit(PEER,V17_MATRIX,'black3',17)
    def test_positive_v17_scope(self):
        r=m.audit(PEER,V17_MATRIX,'black3',17)
        self.assertEqual(r['body_version'],17);self.assertEqual(len(r['runs']),12)
        self.assertFalse(r['strong_live_intermediate_acceptance'])
    def test_old_harness_receipt_rejected(self):self.rejected(lambda r:r.update(schema='neuralstudio.nr.full-frame-inmemory.v10'))
    def test_wrong_body_mode_rejected(self):self.rejected(lambda r:r['frames'][0].update(body_mode='checked'))
    def test_wrong_poison_rejected(self):self.rejected(lambda r:r['frames'][0].update(poison=90))

if __name__=='__main__':unittest.main(verbosity=2)
