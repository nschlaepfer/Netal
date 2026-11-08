#!/usr/bin/env python3
"""Weight-free FP8 graph/ABI tests and optional local-Mac real-quant GPU oracle.

The original FP16 comparison is an approximation metric, not an admission pass.
Native FP8 is checked against independently decoded, actually quantized inputs.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


coverage=module('fp8_coverage_fixture',ROOT/'scripts/test-netal-sr-mpp.py')
converter=module('fp8_pipeline_converter',ROOT/'scripts/build-netal-sr-fp8-pipelines.py')


def fake_manifest(pipelines):
    """Procedural metadata, with no model or vendor bytes, for graph tests only."""
    tensors={};ends={}
    for name,q in pipelines.items():
        layer=converter.ENTRY.fullmatch(Path(name).name).group(1)
        for d in q['dispatches']:
            if d['function']!='pwin_gemm_mpp':continue
            p=d['bytes'][0]['u32'];k,n=p[1:3];offset=converter.binding(d,1)['offset']
            key=(layer,offset,k,n,p[11]or n)
            if key in tensors:continue
            identity=f'{layer}.K{k}N{n}'
            if layer=='enc0'and(k,n)==(16,32):identity='enc0.W1'
            if layer=='dec0'and(k,n)==(32,48):identity='dec0.W7'
            np,kp,stride=converter.align(n,32),converter.align(k,32),converter.align(k,128)
            wo,so=ends.get(layer,(0,0));ends[layer]=(wo+np*stride,converter.align(so+np*4,128))
            tensors[key]={'id':identity,'layer':layer,'name':identity.split('.')[1],'shape':[k,n],
                'source':{'path':f'pipeline/{layer}-weights.bin','offset':offset,'dtype':'float16-le',
                          'row_stride_elements':p[11]or n,'logical_sha256':'fixture-only'},
                'precision':'float16-le'if identity in converter.KEEP_HALF else'e4m3fn',
                'payload':{'file':f'pipeline/fp8/{layer}-weights.e4m3fn.bin','offset':wo,'bytes':np*stride,
                           'shape':[np,kp],'order':'N,K','row_stride_bytes':stride,'padded_k':kp,'padded_n':np},
                'scales':{'file':f'pipeline/fp8/{layer}-scales.f32le.bin','offset':so,'bytes':np*4,
                          'count':np,'dtype':'float32-le','axis':'output_column',
                          'direction':'decoded_value * dequant_scale'}}
    return{'tensors':list(tensors.values()),'pack_root_sha256':'0'*64}


class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory(prefix='netal-fp8-test-')
        bundle,layouts=coverage.fixture(Path(cls.temp.name))
        cls.pipelines=coverage.generate(bundle,layouts,'mpp')
        cls.manifest=fake_manifest(cls.pipelines)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_full_graph_coverage_and_scratch(self):
        totals={'fp8':0,'fallback':0,'mlp':0};frame={}
        for name,q in self.pipelines.items():
            layer,phase=converter.ENTRY.fullmatch(Path(name).name).groups()
            new=converter.convert(layer,q,self.manifest)
            s=new['fp8_scratch'];self.assertGreaterEqual(s['activation_offset'],q['scratch_bytes'])
            self.assertEqual(s['activation_offset']%256,0)
            self.assertGreaterEqual(s['scale_offset'],s['activation_offset']+s['activation_bytes'])
            self.assertLessEqual(s['scale_offset']+s['scale_bytes'],new['scratch_bytes'])
            for index,d in enumerate(new['dispatches']):
                if d['function']=='pwin_gemm_fp8_e4m3':
                    totals['fp8']+=1
                    self.assertEqual(new['dispatches'][index-1]['function'],'pwin_quantize_rows_e4m3')
                    self.assertEqual(converter.binding(d,0)['offset'],s['activation_offset'])
                    self.assertEqual(converter.binding(d,7)['offset'],s['scale_offset'])
                    self.assertEqual(d['bytes'][0]['u32'][3]%128,0)
                elif d['function']=='pwin_gemm_mpp':totals['fallback']+=1
                elif d['function']=='pwin_mlp2_tokens':
                    totals['mlp']+=1;p=d['bytes'][0]['u32'];self.assertEqual(d['grid'],[(p[0]+63)//64,1,1]);self.assertEqual(d['block'],[128,1,1])
            self.assertTrue(set(new['fp8_gemm_coverage']['fp16_fallbacks'])<=converter.KEEP_HALF)
            if not phase:frame.setdefault(Path(name).parent.as_posix(),0);frame[Path(name).parent.as_posix()]+=len(new['dispatches'])
        self.assertEqual(totals,{'fp8':416,'fallback':4,'mlp':12})
        self.assertEqual(set(frame.values()),{134})

    def test_reject_bad_payload_metadata(self):
        t=next(t for t in self.manifest['tensors']if t['precision']=='e4m3fn')
        for field,key,value in [('payload','order','K,N'),('payload','offset',32),('payload','row_stride_bytes',64),
                                ('payload','padded_k',16),('scales','axis','K'),('scales','dtype','float16-le')]:
            bad=copy.deepcopy(t);bad[field][key]=value
            with self.subTest(field=field,key=key),self.assertRaises(ValueError):converter.check_tensor(bad)

    def test_reject_os26_before_building(self):
        output=Path(self.temp.name)/'unsupported-os26'
        result=subprocess.run([sys.executable,str(ROOT/'scripts/build-netal-sr-mpp-kernels.py'),
                               '--variant','fp8','--sdk','iphoneos','--deployment-target','26.0',
                               '--output-dir',str(output)],capture_output=True,text=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('27.0',result.stderr)
        self.assertFalse(output.exists())


def gpu_cases():
    data=coverage.gpu_cases(Tests.pipelines)
    for tokens in [1,16,31,32,63,64,127,128,129]:
        data.append({'name':f'fp8-tail-{tokens}','params':[tokens,96,48,112,48,48,0,0,4,8,4,64,0,0,0,0]})
    data.append({'name':'codec-rne-boundaries','params':[65,32,32,32,32,32,0,0,0,8,0,0,0,0,0,0],'codec':True})
    return data


def gpu(library,receipt):
    # unittest class fixtures have been torn down, but the in-memory graph is retained.
    with tempfile.TemporaryDirectory(prefix='netal-fp8-gpu-')as temporary:
        base=Path(temporary);fixture=base/'cases.json';fixture.write_text(json.dumps(gpu_cases()));exe=base/'test-fp8'
        subprocess.run(['xcrun','clang++','-std=c++17','-fobjc-arc','-O2','-framework','Foundation','-framework','Metal',
                        str(ROOT/'tools/pwin_pipeline/test-fp8.mm'),'-o',str(exe)],check=True)
        result=subprocess.run([str(exe),str(library),str(fixture)],capture_output=True,text=True)
        if receipt:
            if receipt.exists():raise SystemExit('Receipt exists; choose a fresh output file')
            receipt.write_text(result.stdout)
        if result.returncode:raise SystemExit(result.stderr+result.stdout)
        r=json.loads(result.stdout);math_cases=[x for x in r['cases']if'max_abs_real_quant_oracle'in x]
        print('fp8_synthetic='+json.dumps({'device':r['device'],'cases':len(r['cases']),'pass':r['pass'],
              'max_abs_real_quant_oracle':max(x['max_abs_real_quant_oracle']for x in math_cases),
              'max_abs_original_fp16':max(x['max_abs_original_fp16']for x in math_cases),
              'max_rms_original_fp16':max(x['rms_original_fp16']for x in math_cases),
              'quantized_byte_mismatches':sum(x['quantized_byte_mismatches']for x in r['cases']),
              'scale_mismatches':sum(x['scale_mismatches']for x in r['cases']),
              'canary_violations':sum(x['canary_violations']for x in r['cases']),
              'model_quality_admitted':False,'native_instruction_path_verified':False}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--gpu',action='store_true')
    parser.add_argument('--library',type=Path);parser.add_argument('--receipt',type=Path);args=parser.parse_args()
    if args.gpu and not args.library:parser.error('--gpu requires an OS27 FP8 library')
    if not unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests)).wasSuccessful():raise SystemExit(1)
    if args.gpu:gpu(args.library.absolute(),args.receipt)
