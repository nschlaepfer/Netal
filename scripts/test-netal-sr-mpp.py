#!/usr/bin/env python3
"""Weight-free MPP profile/ABI coverage. Optional --gpu runs synthetic Metal math.

The generated model shapes are real; the values/layout fixtures are synthetic.
This test does not validate model image quality or iPhone speed.
"""
import argparse
import ast
import importlib.util
import json
import os
import re
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tools/pwin_pipeline'))
from mpp_backend import convert_pipeline, dispatch, validate_params


def fixture(base, generators=ROOT / 'tools/pwin_pipeline'):
    layouts = base / 'layouts'
    layouts.mkdir(parents=True)
    names = set()
    for path in generators.glob('make-*.py'):
        tree = ast.parse(path.read_text())
        names.update(node.value for node in ast.walk(tree) if isinstance(node, ast.Constant) and isinstance(node.value, str))
    for layer in ['enc0','enc1','enc2','enc3','enc4','dec5','dec4','dec3','dec2','dec1','dec0']:
        layout = {name: {'offset': 0} for name in names}
        layout.update({f'Bpos{h}': {'offset': h * 8192} for h in range(8)})
        (layouts / f'layout-{layer}.json').write_text(json.dumps(layout))
    bundle = base / 'bundle'
    for profile in (bundle, bundle / 'profiles/1920x1080'):
        for stage in ['enc0'] + [f'pwin-{i:02}' for i in range(1, 12)]:
            directory = profile / 'pwin' / stage
            directory.mkdir(parents=True)
            data = bytearray(168 + 16 * 101)
            struct.pack_into('<I', data, 164, 101)
            for parameter in range(101):
                struct.pack_into('<IIQ', data, 168 + parameter * 16, parameter, 6, 0)
            (directory / 'replay-config.bin').write_bytes(data)
            args = bytearray(64)
            struct.pack_into('<II', args, 0, 16, 16)
            padding = {'pwin-04': (4,4), 'pwin-05': (4,0), 'pwin-06': (0,2), 'pwin-07': (4,0), 'pwin-08': (4,4)}.get(stage, (4,4))
            struct.pack_into('<II', args, 56, *padding)
            (directory / 'arguments-template.bin').write_bytes(args)
        shutil.copyfile(profile / 'pwin/enc0/arguments-template.bin', profile / 'pwin/enc0-arguments-template.bin')
    return bundle, layouts


def generate(bundle, layouts, backend='reference'):
    env = dict(os.environ)
    for key in list(env):
        if key.startswith('PWIN_'):
            del env[key]
    result = subprocess.run([sys.executable, str(ROOT / 'tools/pwin_pipeline/build-profile-pipelines.py'),
                             str(bundle), '--layout-dir', str(layouts), '--gemm-backend', backend],
                            env=env, capture_output=True, text=True)
    if result.returncode:
        raise AssertionError(result.stderr + result.stdout)
    return {str(path.relative_to(bundle)): json.loads(path.read_text()) for path in bundle.rglob('pipeline/*.json')}


class Tests(unittest.TestCase):
    def test_reject_unsupported(self):
        params = [65,32,48,32,40,48,0,0,0,8,0,0,40,0,0,0]
        validate_params(params)
        for index, value in [(1,656),(2,784),(4,39),(9,0),(6,4),(8,7),(8,38),(15,1)]:
            broken = params.copy()
            broken[index] = value
            with self.subTest(index=index,value=value), self.assertRaises(ValueError):
                validate_params(broken)
        self.assertEqual(dispatch(params, [])['grid'], [2,2,1])
        with self.assertRaises(ValueError):
            convert_pipeline({'dispatches':[{'function':'pwin_gemm2_tokens'}]})

    def test_every_profile_layer_phase(self):
        with tempfile.TemporaryDirectory(prefix='netal-mpp-fixture-') as temporary:
            bundle, layouts = fixture(Path(temporary))
            reference = generate(bundle, layouts)
            candidate = generate(bundle, layouts, 'mpp')
            self.assertEqual(set(reference), set(candidate))
            shapes, gemms = set(), 0
            for name, q in reference.items():
                expected = convert_pipeline(q)
                self.assertEqual(expected, candidate[name], name)
                for d in candidate[name]['dispatches']:
                    if d['function'] == 'pwin_gemm_mpp':
                        params = d['bytes'][0]['u32']
                        shapes.add(tuple(params[1:3]))
                        gemms += 1
            self.assertEqual(len(reference), 92)  # 11 layers + 5*7 shift variants, ROOT + profile
            self.assertIn((640,160), shapes)
            self.assertIn((160,768), shapes)
            self.assertIn((32,48), shapes)
            functions = {d['function'] for q in candidate.values() for d in q['dispatches']}
            spec = importlib.util.spec_from_file_location('mpp_builder', ROOT / 'scripts/build-netal-sr-mpp-kernels.py')
            builder = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(builder)
            compiled_entries = set()
            for source in builder.SOURCES:
                text = (ROOT / 'tools/pwin_pipeline' / source).read_text()
                compiled_entries.update(re.findall(r'kernel\s+void\s+(\w+)\s*\(', text))
            self.assertFalse(functions - compiled_entries, functions - compiled_entries)
            print('profile_coverage=' + json.dumps({'pipelines':len(reference),'gemm_dispatches':gemms,'shapes':sorted(shapes)}))


def gpu_cases(candidate):
    schedules = {}
    for pipeline in candidate.values():
        for d in pipeline['dispatches']:
            if d['function'] == 'pwin_gemm_mpp':
                params = d['bytes'][0]['u32'].copy()
                params[0] = 65  # tail token tile, never padded to the 64-token MPP grid
                schedules[tuple(params[1:])] = params
    cases = [{'name': f'profile-{i}', 'params': p} for i, p in enumerate(schedules.values())]
    for mode in [0,1,2,3,4,5,6,70]:
        cases.append({'name': f'split-{mode}', 'params': [65,96,48,112,48,48,0,0,mode,8,4,64,0,0,0,0]})
    cases += [
        {'name':'store-cols40-tail','params':[65,32,48,48,40,48,0,0,2,8,0,64,40,0,0,0]},
        {'name':'row-filter','params':[65,16,16,64,32,16,2,1,0,8,4,256,0,0,1,0]},
        {'name':'residual-stride','params':[65,64,64,80,96,64,3,0,3,8,4,80,0,96,0,0]},
        {'name':'relu-bias-broadcast','params':[65,32,48,32,48,0,1,0,2,8,4,64,0,0,0,0]},
    ]
    for item in cases:
        validate_params(item['params'])
    return cases


def run_gpu(library, receipt):
    with tempfile.TemporaryDirectory(prefix='netal-mpp-gpu-') as temporary:
        base = Path(temporary)
        bundle, layouts = fixture(base)
        cases = base / 'cases.json'
        cases.write_text(json.dumps(gpu_cases(generate(bundle, layouts, 'mpp'))))
        executable = base / 'test-mpp'
        subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-O2','-framework','Foundation',
                        '-framework','Metal',str(ROOT / 'tools/pwin_pipeline/test-mpp.mm'),'-o',str(executable)], check=True)
        result = subprocess.run([str(executable),str(library),str(cases)], capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stderr + result.stdout)
        data = json.loads(result.stdout)
        if receipt:
            if receipt.exists():
                raise SystemExit('Receipt exists; choose a new file')
            receipt.write_text(result.stdout)
        print('gpu_synthetic=' + json.dumps({'device':data['device'],'cases':len(data['cases']),
              'max_abs_scalar_fp32':max(x['max_abs_scalar_fp32'] for x in data['cases']),
              'max_abs_gemm3':max(x['max_abs_gemm3'] for x in data['cases']),
              'canary_violations':sum(x['canary_violations'] for x in data['cases']),
              'reference_exact':False,'iphone_test':False,'pass':data['pass']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true', help='run on the local Mac Metal device')
    parser.add_argument('--library', type=Path, help='candidate Mac pwin_mpp_all.metallib')
    parser.add_argument('--receipt', type=Path, help='new private JSON output file for GPU measurements')
    args = parser.parse_args()
    if args.gpu and not args.library:
        parser.error('--gpu requires --library')
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(Tests)
    if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():
        raise SystemExit(1)
    if args.gpu:
        run_gpu(args.library.absolute(), args.receipt)
