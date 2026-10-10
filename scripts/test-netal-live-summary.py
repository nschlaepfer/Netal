#!/usr/bin/env python3
"""Ensure live receipts cannot promote initialization, sampled counters or invalid metrics."""
import importlib.util
import json
from pathlib import Path

path = Path(__file__).with_name('summarize-re9-live-sustained-sr.py')
spec = importlib.util.spec_from_file_location('live_summary', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
ready = module.summarize(b'event=sr_arm_begin suffix=_hw7\nevent=sr_arm_ready bundle=/runtime\nevent=fg_arm_ready bundle=/runtime\n')
assert ready['sr']['initialization_ready_reported'] and ready['fg']['initialization_ready_reported']
assert not ready['sr']['work_scheduled_seen'] and ready['fg']['generated_frames_reported'] is None
assert ready['sr']['initial_mode_hint'] == 'fast' and not any(ready['claims'].values())
samples = module.summarize(b'event=sr_arm_begin suffix=\nevent=sr_exact_scheduled frame=120 input=1280x720 output=1920x1080\nevent=sr_exact_completed frame=120 status=4 command_gpu_ms=10 error=none\nevent=fg_served frame=120 generated=119 fallbacks=0\n')
assert samples['sr']['initial_mode_hint'] == 'reference'
assert samples['sr']['scheduled_max_index_logged'] == 120 and samples['sr']['completion_events_logged'] == 1
assert samples['fg']['generated_frames_reported'] == 119 and not samples['claims']['displayed_fps_verified']
invalid = module.summarize(b'event=sr_exact_completed frame=bad command_gpu_ms=nan status=5 error=GPU\nevent=sr_flash frame=25 total=5\nevent=sr_disarmed reason=gpu_error\n')
assert invalid['sr']['command_gpu_ms_logged_samples']['median'] is None and invalid['invalid_numeric_fields']
assert invalid['sr']['disarm_events_logged'] == 1 and not any(invalid['claims'].values())
assert invalid['sr']['flash_total_reported'] == 5
json.dumps(invalid, allow_nan=False)
partial = module.summarize(b'event=sr_arm_ready x=1\nevent=sr_exact_scheduled frame=999')
assert partial['partial_tail_ignored'] and not partial['sr']['work_scheduled_seen']
print('4 live-summary scenarios passed: initialization, sampled counters, invalid metrics and partial writes')
