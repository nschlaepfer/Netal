#!/usr/bin/env python3
import importlib.util
from pathlib import Path
import unittest
s=importlib.util.spec_from_file_location('prep',Path(__file__).with_name('prepare-nr-eight-frame-reference.py'));m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
ROOT=Path(__file__).resolve().parents[1]
class Eight(unittest.TestCase):
    def test_host_single_handle_and_mid_reset(self):
        p=ROOT/'build/nr_three_frame_history/source-v1/color_cases_trace.cpp';source=m.host(p.read_text())
        self.assertEqual(source,(ROOT/'build/nr_eight_frame/source-v1/color_cases_trace.cpp').read_text())
        self.assertEqual(source.count('const NVSDK_NGX_Result create_result = thunk_create('),1)
        self.assertEqual(source.count('thunk_release(release, handle)'),1)
        self.assertIn('NrEvaluationIndex<8',source)
        self.assertIn("(NrEvaluationIndex%4u)==0u ? 1 : reset_text[0]-'0'",source)
        with self.assertRaises(ValueError):m.host(source)
    def test_capture_tracks_all_frames(self):
        p=ROOT/'build/nr_three_frame_history/source-v1/capture-color.ps1';source=m.capture(p.read_text())
        self.assertEqual(source,(ROOT/'build/nr_eight_frame/source-v1/capture-color.ps1').read_text())
        self.assertIn('$counts.launches -ne 1248',source);self.assertIn('$historyEvents.Count -ne 16',source)
        for i in range(1,8):self.assertIn(f"'frame-{i}-output.rgba16le'",source)
        self.assertEqual(m.SNAPSHOTS,[156*i+j for i in range(8) for j in (2,24)])
        with self.assertRaises(ValueError):m.capture(source)
if __name__=='__main__':unittest.main(verbosity=2)
