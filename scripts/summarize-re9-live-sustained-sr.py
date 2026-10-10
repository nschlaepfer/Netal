#!/usr/bin/env python3
"""Summarize observed SR/FG events without inferring visual quality or presentation."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


def field(line: str, name: str) -> str | None:
    match = re.search(rf'(?:^| ){re.escape(name)}=([^ ]*)', line)
    return match.group(1) if match else None


def summarize(data: bytes) -> dict:
    lines = data.decode('utf-8', errors='replace').splitlines()
    partial = bool(data and not data.endswith(b'\n'))
    if partial:
        lines = lines[:-1]
    groups: dict[str, list[str]] = {}
    for line in lines:
        event = field(line, 'event')
        if event:
            groups.setdefault(event, []).append(line)
    invalid = []

    def events(name):
        return groups.get(name, [])

    def number(line, key, integer=False):
        value = field(line, key)
        if value is None:
            return None
        try:
            parsed = int(value) if integer else float(value)
            if parsed < 0 or not math.isfinite(parsed):
                raise ValueError('invalid metric')
            return parsed
        except (ValueError, OverflowError):
            invalid.append({'event': field(line, 'event'), 'field': key})
            return None

    def maximum(rows, key):
        values = [v for row in rows if (v := number(row, key, True)) is not None]
        return max(values) if values else None

    def dimensions(rows):
        return {key: sorted({v for row in rows if (v := field(row, key))}) for key in ('input', 'render', 'output')}

    scheduled, completed = events('sr_exact_scheduled'), events('sr_exact_completed')
    served = events('fg_served')
    gpu = [v for row in completed if (v := number(row, 'command_gpu_ms')) is not None]
    beginning = events('sr_arm_begin')
    suffix = field(beginning[0], 'suffix') if beginning else None
    mode = ('unknown' if suffix is None else 'reference' if suffix == '' else
            'fast' if suffix.startswith('_hw') else 'custom')
    last = completed[-1] if completed else ''
    report = {
        'schema': 'netal.live_graphics_log_summary.v2',
        'log_sha256': hashlib.sha256(data).hexdigest(),
        'partial_tail_ignored': partial,
        'event_counts_logged': dict(sorted(Counter({name: len(rows) for name, rows in groups.items()}).items())),
        'sr': {
            'initialization_ready_reported': bool(events('sr_arm_ready')),
            'initial_mode_hint': mode,
            'initial_suffix': suffix,
            'mode_change_events': [row for row in lines if field(row, 'event') in ('sr_control_suffix', 'sr_control_pipeline')][-50:],
            'work_scheduled_seen': bool(scheduled),
            'scheduled_max_index_logged': maximum(scheduled, 'frame'),
            'completed_max_index_logged': maximum(completed, 'frame'),
            'completion_events_logged': len(completed),
            'completion_statuses_logged': sorted({v for row in completed if (v := field(row, 'status'))}),
            'completion_errors_logged': sorted({v for row in completed if (v := field(row, 'error'))}),
            'fallback_events_logged': len(events('sr_fallback')),
            'disarm_events_logged': len(events('sr_disarmed')),
            'flash_events_logged': len(events('sr_flash')),
            'flash_total_reported': max((value for value in (maximum(completed, 'flashes'), maximum(events('sr_flash'), 'total')) if value is not None), default=None),
            'last_initialization_wait_seconds_logged': maximum(events('sr_arm_waiting'), 'elapsed_s'),
            'ring_timeouts_reported': maximum(completed, 'ring_timeouts'),
            'dimensions_logged': dimensions(scheduled),
            'command_gpu_ms_logged_samples': {
                'count': len(gpu), 'min': min(gpu) if gpu else None,
                'median': statistics.median(gpu) if gpu else None, 'max': max(gpu) if gpu else None,
            },
            'reported_running_mean_command_gpu_ms': number(last, 'mean_command_gpu_ms'),
            'reported_running_max_command_gpu_ms': number(last, 'max_command_gpu_ms'),
        },
        'fg': {
            'initialization_ready_reported': bool(events('fg_arm_ready')),
            'model_invocation_events_logged': len(served),
            'served_max_index_logged': maximum(served, 'frame'),
            'generated_frames_reported': maximum(served, 'generated'),
            'fallback_events_logged': len(events('fg_fallback')),
            'fallback_total_reported': maximum(served + events('fg_fallback'), 'fallbacks'),
            'disarm_events_logged': len(events('fg_disarmed')),
            'dimensions_logged': dimensions(served),
        },
        'invalid_numeric_fields': invalid,
        'claims': {
            'visual_quality_validated': False, 'sustained_gameplay_validated': False,
            'displayed_fps_verified': False, 'live_rtx_pixel_parity_verified': False,
        },
        'notes': [
            'Initialization and model invocation are separate observations; generated counters do not prove presentation.',
            'Event names containing exact are legacy names and do not establish numerical parity, especially in Fast mode.',
            'Frame indices/counters are sampled log observations, not a complete frame or presentation trace.',
            'Command GPU time includes game work in the same command buffer; it is not isolated SR cost.',
            'Dispatch count and displayed FPS are not measured by this summary.',
        ],
    }
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', required=True, type=Path, help='exact per-process session log to summarize')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = summarize(args.log.read_bytes())
    report['log'] = str(args.log)
    text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + '\n'
    if args.output:
        args.output.write_text(text)
    print(text, end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
