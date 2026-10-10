#!/usr/bin/env python3
"""Run Netal's own native FG model on controlled offline 3D inputs.

Requires an explicit local runtime bundle. Never downloads assets, captures a
screen, launches a game, or changes the installed app. Exit2 records quality
failures; exit1 is an execution/integrity failure.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCENES = {'courtyard', 'orbit', 'fast-turn', 'scene-cut', 'scene-cut-odd', 'translation-x', 'translation-y', 'hud-stress', 'occlusion', 'occlusion-linear'}


class Arguments(argparse.ArgumentParser):
    def error(self, message):
        self.print_usage(sys.stderr)
        self.exit(64, f'{self.prog}: {message}\n')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_bundle(bundle):
    spec = importlib.util.spec_from_file_location('fg_assembly', ROOT / 'scripts/assemble-netal-fg-runtime.py')
    assembly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(assembly)
    manifest = assembly.load_json(bundle / 'manifest.json')
    if manifest.get('schema') != 'netal.fg.runtime-bundle.v1' or manifest.get('profile') != '640x360':
        raise ValueError('Unsupported native FG bundle/profile')
    files = manifest.get('files')
    if not isinstance(files, dict) or not all(isinstance(p, str) and assembly.safe_relative(p) for p in files):
        raise ValueError('Unsafe or absent native FG inventory')
    if assembly.file_set(bundle) != set(files) | {'manifest.json'}:
        raise ValueError('Native FG inventory is incomplete or has extra files')
    for name, identity in files.items():
        assembly.checked_file(bundle / name, identity)
    contract = assembly.load_json(ROOT / 'src/fg_runtime/runtime-inputs.json')
    if assembly.file_set(bundle / 'model') != set(contract['model']):
        raise ValueError('Model inventory must contain exactly the pinned buffers')
    for name, identity in contract['model'].items():
        assembly.checked_file(bundle / 'model' / name, identity)
    for name, identity in contract['math'].items():
        assembly.checked_file(bundle / name, identity)
    if digest(bundle / 'include/fg_runtime.h') != digest(ROOT / 'src/fg_runtime/fg_runtime.h'):
        raise ValueError('Runtime public header differs; rebuild or explicitly reconcile the ABI first')
    return {'manifest_sha256': digest(bundle / 'manifest.json'), 'inventory_files_verified': len(files),
            'pinned_model_buffers_verified': len(contract['model']), 'pinned_math_tables_verified': len(contract['math']),
            'archive_sha256': digest(bundle / 'lib/libnetal_fg_runtime-arm64.a'),
            'model_contract_sha256': digest(ROOT / 'src/fg_runtime/runtime-inputs.json')}


def main():
    parser = Arguments(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=16)
    parser.add_argument('--scenes', default='courtyard,orbit,fast-turn,scene-cut,scene-cut-odd')
    parser.add_argument('--export-inputs', action='store_true', help='Save exact submitted color/motion/depth in inputs.nfg for independent replay')
    parser.add_argument('--hud-mode', choices=('embedded', 'separate'), default='embedded',
                        help='Keep HUD in model inputs, or interpolate the scene then composite the explicit current UI layer')
    parser.add_argument('--motion-experiment', default='none', choices=('none', 'zero', 'extent-scaled', 'extent-negated', 'extent-flip-y'),
                        help='Offline motion-input diagnostic only; leaves the native runtime and expected images unchanged')
    args = parser.parse_args()
    scenes = args.scenes.split(',')
    if not (2 <= args.frames <= 64) or not scenes or len(set(scenes)) != len(scenes) or not set(scenes) <= SCENES:
        parser.error('Choose 2…64 frames and a unique subset of the documented scenes')
    if any(s.startswith('scene-cut') for s in scenes) and args.frames < 15:
        parser.error('Scene-cut fixtures require at least15 frames')
    bundle, output = args.bundle.resolve(), args.output.resolve()
    if args.bundle.is_symlink():
        parser.error('Use the actual runtime bundle directory, not a symlink')
    if output.exists() or args.output.is_symlink() or output.is_relative_to(bundle):
        parser.error('Use a new output directory outside the runtime bundle')
    verified = verify_bundle(bundle)
    output.mkdir(parents=True)
    src = ROOT / 'src/fg_runtime/quality_lab'
    runner = Path(__file__).resolve()
    sequence_header = ROOT / 'src/fg_host/offline_sequence.h'
    hud_shader = ROOT / 'src/fg_runtime/hud_compositor.metal'
    source_hashes = {path: digest(path) for path in (src / 'main.mm', src / 'scene.metal', ROOT / 'src/fg_runtime/fg_runtime.h', sequence_header, hud_shader, runner)}
    binary = output / 'native-fg-quality'
    compile_command = ['xcrun', 'clang++', '-std=c++20', '-fobjc-arc', '-ffp-contract=off', '-O2',
        '-target', 'arm64-apple-macos26.0', '-Wall', '-Wextra', '-Werror', '-Wno-deprecated-declarations',
        str(src / 'main.mm'), str(bundle / 'lib/libnetal_fg_runtime-arm64.a'),
        '-framework', 'Foundation', '-framework', 'Metal', '-framework', 'CoreGraphics', '-framework', 'ImageIO',
        '-o', str(binary)]
    with (output / 'build.log').open('w') as log:
        subprocess.run(compile_command, stdout=log, stderr=subprocess.STDOUT, check=True)
    # Native reference arithmetic, not the game adapter's accelerated overrides.
    env = {k: v for k, v in os.environ.items() if not k.startswith('NSFG_')}
    runs = []
    execution_failed = False
    for scene in scenes:
        if any(digest(path) != expected for path, expected in source_hashes.items()):
            raise ValueError('Source changed during this run; preserve the partial output and rerun')
        with (output / (scene + '.stdout')).open('w') as stdout, (output / (scene + '.stderr')).open('w') as stderr:
            run = subprocess.run([str(binary), str(bundle), str(src / 'scene.metal'), str(output / scene), scene, str(args.frames), args.motion_experiment, 'yes' if args.export_inputs else 'no', args.hud_mode],
                                 env=env, stdout=stdout, stderr=stderr)
        report_path = output / scene / 'report.json'
        if run.returncode not in (0, 2) or not report_path.exists():
            execution_failed = True
            runs.append({'scene': scene, 'status': 'execution-failed', 'exit_code': run.returncode})
        else:
            report = json.loads(report_path.read_text())
            if bool(report['input_sequence_exported']) != args.export_inputs:
                raise ValueError('Input sequence export status differs from request')
            if args.export_inputs and digest(output / scene / 'inputs.nfg') != report['input_sequence_sha256']:
                raise ValueError('Exported input sequence hash mismatch')
            runs.append({'scene': scene, 'status': 'executed', 'quality_failures': report['quality_failures'],
                         'generated_frames': report['generated_frames'], 'receipt_sha256': digest(report_path)})
        print(json.dumps(runs[-1]), flush=True)
    if any(digest(path) != expected for path, expected in source_hashes.items()):
        raise ValueError('Source changed during this run; partial output cannot be certified')
    if verify_bundle(bundle) != verified:
        raise ValueError('Runtime bundle changed during this run; partial output cannot be certified')
    receipt = {'schema': 'netal.native-fg-quality-suite.v1', 'bundle_validation': verified,
        'host_sha256': source_hashes[src / 'main.mm'], 'fixture_sha256': source_hashes[src / 'scene.metal'],
        'runner_sha256': source_hashes[runner], 'bundle_unchanged_after_run': True,
        'sequence_header_sha256': source_hashes[sequence_header], 'input_sequences_exported': args.export_inputs,
        'hud_compositor_sha256': source_hashes[hud_shader], 'hud_mode': args.hud_mode,
        'binary_sha256': digest(binary), 'reference_environment_sanitized': True,
        'frames_per_scene': args.frames, 'runs': runs, 'game_used': False, 'screen_capture_used': False,
        'motion_experiment': args.motion_experiment, 'motion_input_perturbed': args.motion_experiment != 'none',
        'apple_fg_used': False, 'block_flow_fallback_used': False,
        'new_fixture_rtx_parity_proven': False,
        'status': 'execution-failed' if execution_failed else 'native-model-executed'}
    (output / 'suite.json').write_text(json.dumps(receipt, indent=2, sort_keys=True) + '\n')
    if execution_failed:
        return 1
    return 2 if any(r['quality_failures'] for r in runs) else 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        print(f'native-quality-error: {error}', file=sys.stderr)
        sys.exit(1)
