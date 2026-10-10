#!/usr/bin/env python3
"""Seal bounded ANE observations without promoting success from API execution."""
import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    runs=[];pids=set();outputs=set();weights=set();inputs=set();times=[]
    for label in ('a5-a','a5-b','5a-a','5a-b'):
        path=a.attempt/('fp16-'+label);r=json.loads((path/'run.json').read_text())
        assert r['result']=='observed' and r['compiled'] and r['loaded'] and r['evaluated']
        assert r['pid'] not in pids and r['poison']==int(label[:2],16);pids.add(r['pid'])
        x=np.fromfile(path/'input.f32',dtype='<f4').reshape(64,64);w=np.fromfile(path/'weights.f32',dtype='<f4').reshape(64,64);y=np.fromfile(path/'output.f32',dtype='<f4').reshape(64,64)
        expected=(w.astype(np.float64)@x.astype(np.float64)).astype(np.float32)
        retained=np.fromfile(path/'expected-comparison-only.f32',dtype='<f4').reshape(64,64);assert np.array_equal(expected,retained)
        mismatches=int(np.count_nonzero(y.view(np.uint32)!=expected.view(np.uint32)))
        assert mismatches==r['required_fp32_mismatches']==160 and expected[0,0]==2049 and y[0,0]==2050
        assert r['repeat_equal'] and r['input_immutable'] and not r['ane_execution_proven']
        assert r['hardware_time_after']==0 and not r['stats_buffer_changed']
        assert not any((path/'request-perf-stats.bin').read_bytes())
        outputs.add(sha(path/'output.f32'));weights.add(sha(path/'weights.f32'));inputs.add(sha(path/'input.f32'))
        steady=[t for t in r['timings'] if not t['warmup']];assert len(steady)==6
        for t in steady:
            assert t['hardware_time']==0 and t['roundtrip_ms']>=t['evaluate_wait_ms']>=0
            times.append(t['roundtrip_ms'])
        runs.append({'run':label,'pid':r['pid'],'receipt_sha256':sha(path/'run.json'),'output_sha256':sha(path/'output.f32'),'exact_fp32_mismatches':mismatches,
                     'median_host_roundtrip_ms':median(t['roundtrip_ms'] for t in steady),'compile_call_ms':r['compile_ms'],'load_ms':r['load_ms']})
    assert len(outputs)==len(weights)==len(inputs)==1
    fp32=json.loads((a.attempt/'fp32-a5-a/run.json').read_text());assert fp32['result']=='failed' and not fp32['compiled']
    result={'schema':'neuralstudio.nr.ane-feasibility-seal.v1','experiment':'ANE-FEAS-001','result':'not_promoted','placement':'inconclusive',
      'ane_execution_proven':False,'exact_required_fp32_outputs':False,'output_poison_invariance':True,'fresh_processes':4,'runs':runs,
      'observed_host_roundtrip_ms':{'median':median(times),'min':min(times),'max':max(times),'steady_samples':len(times)},
      'input_sha256':next(iter(inputs)),'weights_sha256':next(iter(weights)),'output_sha256':next(iter(outputs)),
      'source_sha256':sha(a.attempt/'conv_probe.mm'),'probe_sha256':sha(a.attempt/'probe'),
      'fp32_compile_failure_receipt_sha256':sha(a.attempt/'fp32-a5-a/run.json'),
      'internal_accumulation_precision_proven':False,'gpu_ane_transfer_and_repair_cost_measured':False,'original_model_group_tested':False,
      'scope':'Synthetic64x64 operation through private ANE API, FP32 I/O and explicit FP16 conv. Request statistics remained zero; placement not established. FP32-blob MIL failed compilation, not proof of universal FP32 hardware incapability. Host timings are observations, not ANE hardware latency or a speedup. No model port/baseline change.'}
    result['receipt_root_sha256']=canonical(result);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
