#!/usr/bin/env python3
"""Measure foreground contours in exported synthetic occlusion frames.

Uses an independent red-minus-green contour threshold on actual/reference raw
images. Does not run a model, change images, or certify visual acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def digest(data):
    return hashlib.sha256(data).hexdigest()


def percentile(values, q=0.95):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values)-1, int(q*(len(values)-1)+0.5))]


def edges(raw, row):
    offset = row*640*4
    scores = [raw[offset+x*4]-raw[offset+x*4+1] for x in range(640)]
    foreground = [x for x, score in enumerate(scores) if score > 64]
    if not foreground or foreground[0] == 0 or foreground[-1] == 639:
        return None
    left, right = foreground[0], foreground[-1]
    return (left-0.5+(64-scores[left-1])/(scores[left]-scores[left-1]),
            right+0.5+(scores[right]-64)/(scores[right]-scores[right+1]))


def analyze(report_path):
    source = report_path.read_bytes()
    report = json.loads(source)
    if report.get('schema') != 'netal.native-fg-quality.v1' or report.get('scene') not in ('occlusion', 'occlusion-linear'):
        raise ValueError('Use a native lab occlusion scene with --export-inputs')
    if (report.get('width'), report.get('height')) != (640, 360) or not report.get('input_sequence_exported'):
        raise ValueError('Raw evidence for the 640x360 profile is required')
    frames = []
    for row in report['frames']:
        if not row['generated']:
            continue
        images = []
        for kind, key in (('native', 'output_sha256'), ('reference', 'reference_sha256')):
            data = (report_path.parent / f'frame-{row["frame_id"]}-{kind}.rgba8').read_bytes()
            if len(data) != 640*360*4 or digest(data) != row[key]:
                raise ValueError('Raw image size/hash does not match its receipt')
            images.append(data)
        left, right, missing, eligible = [], [], 0, 0
        for y in range(83, 277):
            actual, reference = edges(images[0], y), edges(images[1], y)
            # White stripes have no red contour and are excluded by reference,
            # never by model output. A missing model contour is recorded.
            if reference is None:
                continue
            eligible += 1
            if actual is None:
                missing += 1
                continue
            left.append(actual[0]-reference[0]); right.append(actual[1]-reference[1])
        left_bias = statistics.median(left) if left else None
        right_bias = statistics.median(right) if right else None
        frames.append({'frame_id': row['frame_id'], 'eligible_reference_rows': eligible,
            'missing_actual_rows': missing, 'left_bias_px': left_bias, 'right_bias_px': right_bias,
            'p95_absolute_contour_error_px': percentile([abs(x) for x in left+right]),
            'p95_contour_bending_px': percentile([abs(x-left_bias) for x in left]+[abs(x-right_bias) for x in right])})
    changes = []
    for previous, current in zip(frames, frames[1:]):
        if current['frame_id'] != previous['frame_id']+1:
            continue
        for key in ('left_bias_px', 'right_bias_px'):
            if current[key] is not None and previous[key] is not None:
                changes.append(abs(current[key]-previous[key]))
    return {'schema': 'netal.native-fg-occlusion-contours.v1', 'source_report_sha256': digest(source),
        'scene': report['scene'], 'hud_mode': report['hud_mode'], 'frames': frames,
        'missing_actual_rows_total': sum(r['missing_actual_rows'] for r in frames),
        'max_frame_p95_contour_error_px': max((r['p95_absolute_contour_error_px'] for r in frames if r['p95_absolute_contour_error_px'] is not None), default=None),
        'max_frame_p95_bending_px': max((r['p95_contour_bending_px'] for r in frames if r['p95_contour_bending_px'] is not None), default=None),
        'p95_adjacent_frame_bias_change_px': percentile(changes),
        'scope': 'Synthetic red-contrast contour diagnostics; no runtime promotion or visual-acceptance claim'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error('Use a new output file')
    result = analyze(args.report)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2, sort_keys=True); output.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'frames'}, sort_keys=True))


if __name__ == '__main__':
    main()
