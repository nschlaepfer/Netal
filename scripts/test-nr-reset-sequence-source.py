#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
spec=importlib.util.spec_from_file_location('seq',Path(__file__).with_name('prepare-nr-reset-sequence.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
PARENT=Path(__file__).resolve().parents[1]/'build/nr_color_encoder_reference/source-v1'
class Sequence(unittest.TestCase):
    def test_one_creation_one_release_looped_evaluate(self):
        s=m.host_sequence((PARENT/'color_cases_trace.cpp').read_text())
        self.assertEqual(s.count('const NVSDK_NGX_Result create_result = thunk_create('),1)
        self.assertEqual(s.count('thunk_release(release, handle)'),1)
        begin=s.index('for(NrEvaluationIndex=0;NrEvaluationIndex<2;')
        self.assertLess(begin,s.index('const NVSDK_NGX_Result evaluate_result ='))
        self.assertLess(s.index('runtime.sequence_frames=2'),s.index('thunk_release(release, handle)'))
        self.assertIn('(NrEvaluationIndex ? "frame-1-" : "") + filename',s)
        self.assertIn('std::fill(output_host.begin(),output_host.end(),output_poison)',s)
        self.assertIn('color_after!=color_host || depth_after!=depth_host || motion_after!=motion_host',s)
    def test_reset_contract_is_explicit(self):
        s=m.host_sequence((PARENT/'color_cases_trace.cpp').read_text())
        self.assertIn("reset_text[0]!='0'&&reset_text[0]!='1'",s)
        self.assertIn("NrEvaluationIndex ? reset_text[0]-'0' : 1",s)
    def test_capture_keeps_first_metrics_unique_and_seals_second_files(self):
        s=m.capture_sequence((PARENT/'capture-color.ps1').read_text())
        self.assertIn('$counts.launches -ne 312',s);self.assertIn('$argumentDumps.Count -ne 312',s)
        self.assertIn("'frame-1-input-color-after.rgba16le'",s)
        self.assertIn('second_output_sha256 = $secondOutputHash',s)
        self.assertIn("if ($CaptureSnapshots -or $Fp16Analysis) { throw",s)
    def test_changed_parent_rejected(self):
        with self.assertRaises(ValueError):m.host_sequence('unknown')
        with self.assertRaises(ValueError):m.capture_sequence('unknown')
    def test_repeated_transform_rejected(self):
        with self.assertRaises(ValueError):m.host_sequence(m.host_sequence((PARENT/'color_cases_trace.cpp').read_text()))
if __name__=='__main__':unittest.main(verbosity=2)
