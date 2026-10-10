#!/usr/bin/env python3
"""Mutation controls for the recovered live-run auditor; no GPU execution."""
import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('audit',Path(__file__).with_name('audit-nr-recovery-run.py'))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
PEER = m.ROOT.parent/'NeuralStudio-nr-fable'
RUN = m.ROOT/'build/nr_recovery_runs/black3-v11-v1'
TARGET = RUN/'replica-1-pilot/receipt.json'


class RecoveryRun(unittest.TestCase):
    def reject(self, edit):
        original = m.read
        def altered(path):
            value = original(path)
            if Path(path) == TARGET:
                value = copy.deepcopy(value)
                edit(value)
            return value
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):
                m.audit(PEER,RUN,11)

    def test_missing_frame(self):
        self.reject(lambda r:r['frames'].pop())

    def test_reference_bypass(self):
        self.reject(lambda r:r['frames'][0].update(plumbing_only_no_parity_meaning=True))

    def test_missing_output_write(self):
        self.reject(lambda r:r['frames'][0]['body'].update(final_pixels_not_written=1))

    def test_wrong_history(self):
        self.reject(lambda r:r['frames'][1]['body'].update(history_sha256_before='0'*64))

    def test_wrong_boundary(self):
        self.reject(lambda r:r['frames'][0]['regions']['skip_c32'].update(sha256='0'*64))

    def test_missing_loaded_provenance(self):
        self.reject(lambda r:r['frames'][0]['body']['loaded'].pop(next(iter(r['frames'][0]['body']['loaded']))))

    def test_suppressed_omission(self):
        self.reject(lambda r:r['frames'][0]['body'].update(checks_skipped=[]))


class MidSequenceReset(unittest.TestCase):
    directory=m.ROOT/'build/nr_recovery_runs/varied-v1/midreset8'

    def test_complete_reset_sequence(self):
        result=m.audit(PEER,self.directory,17,'midreset8')
        self.assertEqual(result['frames'],32)
        self.assertEqual(result['actual_output_history_bytes_checked'],1592524800)

    def reject(self,edit):
        original=m.read
        def altered(path):
            value=original(path)
            if Path(path)==self.directory/'replica-1-pilot/receipt.json':
                value=copy.deepcopy(value);edit(value)
            return value
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER,self.directory,17,'midreset8')

    def test_global_index_is_not_reset_local_index(self):
        self.reject(lambda r:r['frames'][4]['encoder'].update(frame_index=4))

    def test_reset_disagreement_rejected(self):
        self.reject(lambda r:r['frames'][4]['temporal'].update(body_reset=False))

    def test_pre_reset_history_rejected(self):
        self.reject(lambda r:r['frames'][5]['body'].update(history_sha256_before=r['frames'][3]['body']['history_sha256_after']))


class EncoderGpuFill(unittest.TestCase):
    directory=m.ROOT/'build/nr_encoder_gpu_fill/block2-correctness-v1'

    def test_original_reference_identity(self):
        _,n,resets,images,_,_=m.reference_contract('encfill-original3')
        self.assertEqual((n,resets),(3,[1,0,0]))
        self.assertTrue(all(len(h)==64 for h in images))
        self.assertEqual(images[0],'81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342')

    def test_complete_candidate(self):
        r=m.audit(PEER,self.directory,17,'encfill-original3')
        self.assertEqual(r['frames'],12)
        self.assertFalse(r['performance_claim'])

    def reject(self,edit):
        original=m.read
        def altered(path):
            r=original(path)
            if Path(path)==self.directory/'process-1/receipt.json':r=copy.deepcopy(r);edit(r['frames'][0])
            return r
        with patch.object(m,'read',side_effect=altered):
            with self.assertRaises(ValueError):m.audit(PEER,self.directory,17,'encfill-original3')

    def test_missing_branch_observation(self):
        self.reject(lambda f:f['encoder_telemetry'].update(phases=[p for p in f['encoder_telemetry']['phases'] if p['label']!='block2/gpu_fill_encoding']))

    def test_unreviewed_second_block(self):
        self.reject(lambda f:f['encoder_telemetry']['phases'].append({'label':'block3/gpu_fill_encoding','wall_ms':0}))

    def test_missing_telemetry(self):
        self.reject(lambda f:f.update(encoder_telemetry_enabled=False))


if __name__ == '__main__':
    unittest.main()
