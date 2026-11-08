#!/usr/bin/env python3
"""Weight-free admission tests for the opt-in mpp-half fused MLP candidate.

--gpu compares the local Mac candidate with the retained original MLP and an
independent scalar oracle. It cannot establish model quality or A20 performance.
"""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def validate_case(case):
    p, c = case['params'], case['constants']
    if len(p) != 8 or len(c) != 7 or any(type(x) is not int or not 0 <= x <= 0xffffffff for x in p + c):
        raise ValueError('MLP requires 8 u32 parameters and seven uint constants')
    k, n, xs, cs, ys, hd, ws = c
    if (k, hd, n) not in ((32, 128, 32), (64, 256, 64)):
        raise ValueError('Unsupported MLP shape')
    if p[0] == 0 or p[2] == 0 or p[1] not in (0, 2, 5, 6) or p[4] not in (0, 1):
        raise ValueError('Unsupported MLP parameters')
    if p[5] >= k // 16 or p[6:] != [0, 0] or xs < k or cs < n or ys < n or (ws or n) < n:
        raise ValueError('Unsupported MLP strides, rounds or reserved words')
    if any(p[0] * stride > 0xffffffff for stride in (xs, cs, ys)):
        raise ValueError('MLP tensor addressing exceeds uint32')
    return case


def cases():
    out = []
    for layer, k, hd, n, split in [('enc0',32,128,32,2), ('dec0',32,128,32,2),
                                   ('enc1',64,256,64,5), ('enc2',64,256,64,5),
                                   ('dec1',64,256,64,5), ('dec2',64,256,64,5)]:
        out.append({'name': 'actual-' + layer, 'params': [65,split,8,4,1,0,0,0],
                    'constants': [k,n,k,n,n,hd,n]})
    for k, hd, n, split in [(32,128,32,2),(64,256,64,5)]:
        for tokens in [1,15,16,17,31,32,63,64,65,127,128,129]:
            out.append({'name': f'tail-{k}-{tokens}', 'params':[tokens,split,8,0,1,0,0,0],
                        'constants':[k,n,k+16,n+32,n+16,hd,n+16]})
        for mode in [0,2,5,6]:
            out.append({'name':f'split-{k}-{mode}', 'params':[65,mode,8,0,1,0,0,0],
                        'constants':[k,n,k,n,n,hd,0]})
        out.append({'name':f'no-scale-{k}', 'params':[65,split,3,6,0,0,0,0],
                    'constants':[k,n,k,n,n,hd,n]})
        out.append({'name':f'stress-{k}', 'params':[65,split,8,4,1,0,0,0],
                    'constants':[k,n,k,n,n,hd,n], 'stress':3.0})
    out.append({'name':'explicit-rounds', 'params':[65,5,8,4,1,1,0,0],
                'constants':[64,64,80,96,80,256,80]})
    return [validate_case(case) for case in out]


class Tests(unittest.TestCase):
    def test_actual_shapes_and_tails(self):
        data = cases()
        self.assertEqual(len([x for x in data if x['name'].startswith('actual-')]), 6)
        self.assertEqual({(c['constants'][0],c['constants'][5],c['constants'][1]) for c in data},
                         {(32,128,32),(64,256,64)})
        self.assertTrue({1,15,16,17,31,32,63,64,65,127,128,129} <= {c['params'][0] for c in data})

    def test_unsupported_shape_and_abi(self):
        good = cases()[0]
        for field, index, value in [('constants',0,48),('constants',2,16),('params',0,0),
                                     ('params',1,1),('params',2,0),('params',4,2),
                                     ('params',5,2),('params',6,1)]:
            item = {'params':good['params'].copy(),'constants':good['constants'].copy()}
            item[field][index] = value
            with self.subTest(field=field,index=index), self.assertRaises(ValueError):
                validate_case(item)


def gpu(candidate, original, receipt):
    with tempfile.TemporaryDirectory(prefix='netal-mpp-mlp-') as temporary:
        base = Path(temporary)
        fixture = base / 'cases.json'
        fixture.write_text(json.dumps(cases()))
        executable = base / 'test-mpp-mlp'
        subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-O2','-framework','Foundation',
                        '-framework','Metal',str(ROOT / 'tools/pwin_pipeline/test-mpp-mlp.mm'),
                        '-o',str(executable)], check=True)
        result = subprocess.run([str(executable),str(candidate),str(original),str(fixture)], capture_output=True, text=True)
        if receipt:
            if receipt.exists():
                raise SystemExit('Receipt exists; select a new output file')
            receipt.write_text(result.stdout)
        if result.returncode:
            raise SystemExit(result.stderr + result.stdout)
        r = json.loads(result.stdout)
        print('mlp_synthetic=' + json.dumps({'device':r['device'],'cases':len(r['cases']),
              'max_abs_original':max(x['max_abs_original_mlp2'] for x in r['cases']),
              'max_abs_scalar':max(x['max_abs_scalar_fp32'] for x in r['cases']),
              'canary_violations':sum(x['canary_violations'] for x in r['cases']),
              'nonfinite_elements':sum(x['nonfinite_elements'] for x in r['cases']),
              'fp8':False,'model_inference_validated':False,'pass':r['pass']}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--candidate', type=Path)
    parser.add_argument('--original', type=Path)
    parser.add_argument('--receipt', type=Path)
    args = parser.parse_args()
    if args.gpu and (not args.candidate or not args.original):
        parser.error('--gpu requires --candidate and --original metallibs')
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful():
        raise SystemExit(1)
    if args.gpu:
        gpu(args.candidate.absolute(), args.original.absolute(), args.receipt)
