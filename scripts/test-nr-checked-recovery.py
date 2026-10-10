#!/usr/bin/env python3
"""Synthetic checkpoint mutations, not inference or new RTX evidence."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit-nr-checked-recovery.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class Checkpoints(unittest.TestCase):
    def setUp(self):
        path=ROOT.parent/'NeuralStudio-nr-fable/build/nr_full/baseline-prod-v1/v11-p1/process-1/diag/frame-0/chain-run.json'
        self.template=json.loads(path.read_text());self.old=copy.deepcopy(self.template)
        h='1'*64
        for x in self.old['launches']:
            if x['kind']=='ptx':x['pool_post_sha256']=h
        self.new={'launches':[], 'decoder_pre_image_source':'gpu-snapshot',
            'decoder_pre_image_sha256':self.old['decoder']['pool_pre_sha256'],
            'decoder_pre_image_expected_sha256':self.old['decoder']['pool_pre_sha256'],
            'pool_final_sha256':self.old['decoder']['pool_final_sha256'],
            'tex155_output_sha256':next(x for x in self.old['launches'] if x['launch']==155)['texture']['output_sha256']}
        for x in self.old['launches']:
            d={k+'_sha256':v for k,v in x.get('outputs',{}).items()}
            if x['kind']=='ptx':d['pool_sha256']=h
            self.new['launches'].append({'launch':x['launch'],'digests':d})
    def check(self):return m.compare_checkpoints(self.old,self.new,self.template)
    def test_complete(self):self.assertEqual(self.check(),152)
    def test_missing_launch(self):
        self.old['launches'].pop()
        with self.assertRaisesRegex(ValueError,'exact baseline launch inventory'):self.check()
    def test_reordered_launch(self):
        self.old['launches'][0:2]=self.old['launches'][1::-1]
        with self.assertRaisesRegex(ValueError,'exact baseline launch inventory'):self.check()
    def test_duplicate_checkpoint(self):
        self.new['launches'].append(copy.deepcopy(self.new['launches'][0]))
        with self.assertRaisesRegex(ValueError,'duplicate resident checkpoint'):self.check()
    def test_missing_checkpoint(self):
        self.new['launches'][0]['digests'].pop(next(iter(self.new['launches'][0]['digests'])))
        with self.assertRaisesRegex(ValueError,'complete paired checkpoint inventory'):self.check()
    def test_changed_hash(self):
        self.new['launches'][0]['digests'][next(iter(self.new['launches'][0]['digests']))]='2'*64
        with self.assertRaisesRegex(ValueError,'SHA256 values match'):self.check()
    def test_null_pool_not_equal(self):
        self.old['launches'][-1]['pool_post_sha256']=None
        self.new['launches'][-1]['digests']['pool_sha256']=None
        with self.assertRaisesRegex(ValueError,'SHA256 values match'):self.check()
    def test_preimage_not_snapshot(self):
        self.new['decoder_pre_image_source']='cpu-reconstruction'
        with self.assertRaisesRegex(ValueError,'actual pre-image equality'):self.check()
    def test_changed_final_pool(self):
        self.new['pool_final_sha256']='3'*64
        with self.assertRaisesRegex(ValueError,'final pool equality'):self.check()
    def test_changed_texture(self):
        self.new['tex155_output_sha256']='4'*64
        with self.assertRaisesRegex(ValueError,'post texture equality'):self.check()

if __name__=='__main__':unittest.main()
