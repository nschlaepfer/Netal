#!/usr/bin/env python3
"""Negative controls for the independent full-chain seal audit; no GPU/files."""
import copy
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-full-chain.py'))
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
class SealTests(unittest.TestCase):
    def setUp(self):
        self.root=Path('/tmp/nr-audit-unit-fixture')
        self.encoder={'result':'pass','receipt_root_sha256':'1'*64,'fresh_native_processes':12,'claims':{'recorded_block0_entry':False,'native_image_to_blocks0_through22':True}}
        self.chain={'result':'pass','steps':[{'label':'encoder-verify','exit_code':0}]+[{'label':f'{s}-{r}','exit_code':0} for r in audit.RUNS for s in ('middle','body')],
                    'stages':{r:{} for r in audit.RUNS},'encoder_run':{'path':str(self.root/'encoder-run'),'verification':copy.deepcopy(self.encoder)}}
    def reject(self,mutate):
        mutate(self.chain,self.encoder)
        self.assertTrue(audit.seal_issues(self.chain,self.encoder,self.root))
    def test_valid_metadata(self):self.assertEqual(audit.seal_issues(self.chain,self.encoder,self.root),[])
    def test_valid_reference_seal(self):
        enc=self.chain['encoder_run'];enc.pop('verification');enc.update(verification_root=self.encoder['receipt_root_sha256'],verification_result='pass',verification_sha256='3'*64)
        self.assertEqual(audit.seal_issues(self.chain,self.encoder,self.root),[])
    def test_valid_nested_resume(self):
        self.chain['encoder_run']['path']=str(self.root/'resumed/encoder-run')
        self.assertEqual(audit.seal_issues(self.chain,self.encoder,self.root),[])
    def test_failed_verifier(self):self.reject(lambda c,e:c['steps'][0].update(exit_code=2))
    def test_missing_verifier_step(self):self.reject(lambda c,e:c.update(steps=c['steps'][1:]))
    def test_failed_body_hidden_by_pass(self):self.reject(lambda c,e:c['steps'][2].update(exit_code=1))
    def test_missing_middle_execution(self):self.reject(lambda c,e:c['steps'].pop(1))
    def test_duplicate_stage_execution(self):self.reject(lambda c,e:c['steps'].append(copy.deepcopy(c['steps'][1])))
    def test_only_one_pass_per_poison(self):self.reject(lambda c,e:c.update(stages={'a5':{},'5a':{}}))
    def test_external_resume(self):self.reject(lambda c,e:c['encoder_run'].update(path='/tmp/old-encoder'))
    def test_missing_verification(self):self.reject(lambda c,e:c['encoder_run'].update(verification='not produced'))
    def test_changed_verification_root(self):self.reject(lambda c,e:c['encoder_run']['verification'].update(receipt_root_sha256='2'*64))
    def test_recorded_entry(self):self.reject(lambda c,e:e['claims'].update(recorded_block0_entry=True))
    def test_native_producer_claim_missing(self):self.reject(lambda c,e:e['claims'].pop('native_image_to_blocks0_through22'))
    def test_fewer_native_processes(self):self.reject(lambda c,e:e.update(fresh_native_processes=6))
    def test_failed_encoder_data(self):self.reject(lambda c,e:e.update(result='fail'))
if __name__=='__main__':unittest.main()
