#!/usr/bin/env python3
"""Full-K relaxed GEMM tests. Existing strict MPP/Reference tests are unchanged.

Native FP16-input math is checked against an independent full-FP32 oracle.
Differences from original GEMM3 are reported separately as approximation error.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('fast_half_shapes',ROOT/'scripts/test-netal-sr-mpp.py')
coverage=importlib.util.module_from_spec(spec);spec.loader.exec_module(coverage)


def cases(pipelines):
    result=coverage.gpu_cases(pipelines)
    for tokens in [1,15,16,17,31,32,63,64,127,128,129]:
        result.append({'name':f'full-k-tail-{tokens}',
                       'params':[tokens,96,48,112,48,48,0,0,4,8,4,64,0,0,0,0]})
    return result


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary=tempfile.TemporaryDirectory(prefix='netal-fast-half-shapes-')
        bundle,layouts=coverage.fixture(Path(cls.temporary.name))
        cls.pipelines=coverage.generate(bundle,layouts,'mpp')

    @classmethod
    def tearDownClass(cls):cls.temporary.cleanup()

    def test_full_model_shapes_and_abi(self):
        data=cases(self.pipelines)
        self.assertEqual(len(data),67)
        self.assertEqual(len({tuple(x['params'][1:3])for x in data}),27)
        self.assertTrue({(640,160),(160,768),(32,48)}<={tuple(x['params'][1:3])for x in data})
        for item in data:
            coverage.validate_params(item['params'])


def gpu(library,receipt):
    with tempfile.TemporaryDirectory(prefix='netal-fast-half-gpu-')as temporary:
        base=Path(temporary);fixture=base/'cases.json';fixture.write_text(json.dumps(cases(Tests.pipelines)))
        exe=base/'test-fast-half'
        subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-O2','-framework','Foundation','-framework','Metal',
                        str(ROOT/'tools/pwin_pipeline/test-fast-half.mm'),'-o',str(exe)],check=True)
        result=subprocess.run([str(exe),str(library),str(fixture)],capture_output=True,text=True)
        if receipt:
            if receipt.exists():raise SystemExit('Receipt exists; choose a fresh output file')
            receipt.write_text(result.stdout)
        if result.returncode:raise SystemExit(result.stderr+result.stdout)
        r=json.loads(result.stdout)
        print('fast_half_synthetic='+json.dumps({'device':r['device'],'cases':len(r['cases']),'pass':r['pass'],
            'max_abs_full_fp32_oracle':max(x['max_abs_full_fp32_oracle']for x in r['cases']),
            'max_abs_original_gemm3':max(x['max_abs_original_gemm3']for x in r['cases']),
            'worst_rms_original_gemm3':max(x['rms_original_gemm3']for x in r['cases']),
            'canary_violations':sum(x['canary_violations']for x in r['cases']),
            'nonfinite_elements':sum(x['nonfinite_elements']for x in r['cases']),
            'model_inference_validated':False,'original_gemm3_error_is_reported_not_admitted':True}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--library',type=Path);parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    if args.gpu and not args.library:parser.error('--gpu requires a fast-half candidate library')
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful():raise SystemExit(1)
    if args.gpu:gpu(args.library.absolute(),args.receipt)
