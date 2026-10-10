#!/usr/bin/env python3
"""Fail-closed checks on retained native prefix; no GPU execution."""
import copy,importlib.util,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('audit',ROOT/'scripts/audit-nr-recovery-run.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PEER=ROOT.parent/'NeuralStudio-nr-fable';RUN=ROOT/'build/nr_sixteen/native-recovery-v1'
TARGET=RUN/'repeat-a/receipt.json'
class PrefixControls(unittest.TestCase):
    def reject(self,edit,reason):
        original=m.read
        def changed(p):
            r=original(p)
            if Path(p)==TARGET:r=copy.deepcopy(r);edit(r)
            return r
        with patch.object(m,'read',side_effect=changed):
            with self.assertRaisesRegex(ValueError,reason):m.audit(PEER,RUN,17,'sixteen',prefix_only=True)
    def test_missing_last_frame(self):self.reject(lambda r:r['frames'].pop(),'native frame inventory')
    def test_wrong_encoder_build(self):self.reject(lambda r:r.update(executable_sha256='0'*64),'actual runtime identity')
    def test_no_reference_bypass(self):self.reject(lambda r:r['frames'][0].update(no_reference_no_parity_meaning=True),'no bypass')
    def test_late_history_damage(self):self.reject(lambda r:r['frames'][15]['body'].update(history_sha256_before='0'*64),'own prior exact history')
    def test_fault_hook(self):self.reject(lambda r:r['frames'][0].update(test_fault='fault'),'no bypass or fault')
if __name__=='__main__':unittest.main()
