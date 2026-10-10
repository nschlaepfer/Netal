#!/usr/bin/env python3
"""Verify arithmetic-domain capture and offline composition against the PTX tap.

Only comparison/analysis: this script never executes or supplies a native model.
The captured radius column is excluded from runtime artifacts by contract.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
N=1<<24
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('results',type=Path);p.add_argument('--features',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve verification';errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    inputs=read(a.source/'inputs.json');job=read(a.results/'receipt.json');body=dict(inputs);claimed=body.pop('receipt_root_sha256')
    check(hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==claimed,'input root')
    check(job['inputs_root_sha256']==claimed,'remote source root')
    for name,h in inputs['files'].items():check(sha(a.source/name)==h,'source '+name)
    check(sha(a.results/'sass.txt')==job['sass_sha256'],'SASS identity')
    for item in job['artifacts']:
        path=a.results/'output'/item['name'];check(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'transfer '+item['name'])
    pids=[];hashes=[];runs=[]
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            prefix=a.results/'output'/f'{poison}-{ordinal}';r=read(prefix.with_suffix('.json'))
            check(r['result']=='pass' and r['coverage_errors']==0 and r['canary_errors']==0,'gates '+prefix.name)
            check(r['poison']==int(poison,16) and r['records']==N and r['model_inputs'] is False and r['expected_gpu_input'] is False,'configuration '+prefix.name)
            c=np.memmap(prefix.with_suffix('.coverage.bin'),mode='r',dtype='<u4');check(c.size==N*6 and np.all(c==1),'coverage '+prefix.name)
            pids.append(r['pid']);hashes.append(sha(prefix.with_suffix('.bin')));runs.append({'run':prefix.name,'pid':r['pid'],'output_sha256':hashes[-1]})
    check(len(set(pids))==4,'fresh processes');check(len(set(hashes))==1,'repeat/poison equality')
    table=np.memmap(a.results/'output/a5-a.bin',mode='r',dtype='<u4').reshape(N,6)
    check(np.array_equal(table[:N//2,3:5],table[N//2:,3:5]),'SQRT repeated normalized domains')
    # Entire legal positive 24-bit grid: SQRT exponent parity reconstruction.
    def radius(indices):
        log=table[indices,0].view(np.float32)
        arg=(log*np.float32(float.fromhex('0x1.62e430p-1')))*np.float32(-2)
        bits=arg.view(np.uint32);e=((bits>>23)&255).astype(np.int32)-127
        m=bits&0x7fffff
        answer=table[m,3+(e&1)].astype(np.int64)+(e//2).astype(np.int64)*0x800000
        answer=np.where((bits&0x7fffffff)==0,bits,answer).astype(np.uint32)
        return answer
    sqrt_mismatch=0;sqrt_examples=[]
    for start in range(0,N,1<<20):
        indices=np.arange(start,min(start+(1<<20),N),dtype=np.uint32);actual=radius(indices);expected=table[indices,5]
        bad=np.flatnonzero(actual!=expected);sqrt_mismatch+=bad.size
        sqrt_examples += [{'unit_index':int(indices[j])+1,'actual':f'{actual[j]:08x}','expected':f'{expected[j]:08x}'} for j in bad[:max(0,16-len(sqrt_examples))]]
    check(sqrt_mismatch==0,'SQRT range reduction')
    check(sha(a.features)=='49abcec0ab2e1068a03339564aa99c5358c0720eb06fe4930ccc803aa2b0a06f','feature identity')
    expected=np.memmap(a.features,mode='r',dtype='<u2').reshape(34560,1024)
    pixel=np.arange(1920*1152,dtype=np.uint32);x=pixel%1920;y=pixel//1920
    state=(x*np.uint32((-1918454973)&0xffffffff))^(y*np.uint32((-669632447)&0xffffffff))^np.uint32(608135816)
    def word(s):return ((s>>((s>>28)+4))^s)*np.uint32(277803737)
    w=word(state);seed=(w>>22)^w
    def index(m,b):
        w=word(seed*np.uint32(m&0xffffffff)+np.uint32(b&0xffffffff));return (w>>30)^(w>>8)
    i0=index(747796405,-1403630843);i1=index(-93469191,1192405134);i2=index(-895109107,568162667);i3=index(-2094846927,878960812)
    r0=radius(i0).view(np.float32);r2=radius(i2).view(np.float32)
    values=[r0*table[i1,2].view(np.float32),r0*table[i1,1].view(np.float32),r2*table[i3,2].view(np.float32)]
    tiles=(y//8)*240+x//8;slots=(y%8)*8+x%8;noise=[]
    for k,value in enumerate(values):
        actual=value.astype(np.float16).view(np.uint16);ref=expected[tiles,slots*8+k];bad=np.flatnonzero(actual!=ref)
        noise.append({'channel':k,'mismatching_half_values':int(bad.size),'examples':[{'x':int(x[j]),'y':int(y[j]),'actual':f'{actual[j]:04x}','expected':f'{ref[j]:04x}'} for j in bad[:8]]})
    check(all(x['mismatching_half_values']==0 for x in noise),'offline primitive composition vs original PTX feature tap')
    receipt={'schema':'neuralstudio.nr.noise-math-verification.v1','inputs_root_sha256':claimed,'job_receipt_sha256':sha(a.results/'receipt.json'),
             'runs':runs,'sqrt_range_mismatches':int(sqrt_mismatch),'sqrt_range_counterexamples':sqrt_examples,'offline_feature_composition':noise,
             'errors':errors,'result':'fail' if errors else 'pass','native_mac_parity':False,'runtime_radius_lookup_allowed':False,
             'scope':'Math-domain oracle repeat and SQRT decomposition audit; offline feature composition only. No Mac primitive correction/runtime has been accepted.'}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
