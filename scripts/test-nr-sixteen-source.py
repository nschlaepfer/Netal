#!/usr/bin/env python3
from pathlib import Path
import unittest
ROOT=Path(__file__).resolve().parents[1]/'build/nr_sixteen/source-v1'
class Sixteen(unittest.TestCase):
    def test_one_handle_no_mid_reset(self):
        s=(ROOT/'color_cases_trace.cpp').read_text();self.assertIn('NrEvaluationIndex<16',s);self.assertIn("NrEvaluationIndex==0u ? 1 : reset_text[0]-'0'",s);self.assertNotIn('NrEvaluationIndex%4u',s)
        self.assertEqual(s.count('const NVSDK_NGX_Result create_result = thunk_create('),1);self.assertEqual(s.count('thunk_release(release, handle)'),1)
    def test_all_capture_extents(self):
        s=(ROOT/'capture-color.ps1').read_text();self.assertIn('$counts.launches -ne 2496',s);self.assertIn('$historyEvents.Count -ne 32',s);self.assertIn('$counts.snapshots -ne 32',s)
        for i in range(1,16):self.assertIn(f"'frame-{i}-output.rgba16le'",s)
        self.assertIn('snapshot_launch_limit = 2364',s);self.assertIn('if($i -eq 0)',s)
    def test_observer_window(self):
        s=(ROOT/'nr_history_readback.inl').read_text();self.assertIn('frame < 16 && after < 2',s)
        p=(ROOT/'prefix_proxy.cpp').read_text()
        for i in range(16):
            for j in (2,24):self.assertIn(f'launch_id != {156*i+j}',p)
if __name__=='__main__':unittest.main(verbosity=2)
