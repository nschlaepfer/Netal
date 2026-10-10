#!/usr/bin/env python3
import copy,importlib.util,json,unittest
from pathlib import Path
s=importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-connected-telemetry.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
PEER=Path('/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt')
R=json.loads((PEER/'build/nr_full/telemetry-parity-v1/on/on-1/receipt.json').read_text());F=R['frames'][0]
class Telemetry(unittest.TestCase):
    def reject(self,change):
        t=copy.deepcopy(F['encoder_telemetry']);change(t)
        with self.assertRaises(ValueError):m.validate_telemetry(t,F,R['pid'],0)
    def test_positive(self):self.assertEqual(m.validate_telemetry(F['encoder_telemetry'],F,R['pid'],0),139)
    def test_stale_generation(self):self.reject(lambda t:t.update(generation=2))
    def test_other_process(self):self.reject(lambda t:t.update(pid=R['pid']+1))
    def test_other_input(self):self.reject(lambda t:t.update(input_sha256='0'*64))
    def test_guard_relaxation(self):self.reject(lambda t:t.update(guards_unchanged=False))
    def test_changed_shader(self):self.reject(lambda t:t.update(shader_changes=True))
    def test_missing_input_blit(self):self.reject(lambda t:t['records'].pop(0))
    def test_missing_gpu_stamp(self):self.reject(lambda t:t['records'][0].update(gpu_timestamp_valid=False))
    def test_wrong_units(self):self.reject(lambda t:t['records'][0].update(gpu_ms=t['records'][0]['gpu_ms']*1000))
if __name__=='__main__':unittest.main()
