#!/usr/bin/env python3
"""Validate replay and noninterfering diagnostic taps, without accepting a port."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(b):return hashlib.sha256(b).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('results',type=Path)
    p.add_argument('recorded_pool',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve verification';errors=[]
    def check(ok,message):
        if not ok:errors.append(message)
    inputs=read(a.source/'inputs.json');job=read(a.results/'receipt.json');root=dict(inputs);claimed=root.pop('receipt_root_sha256')
    check(digest(json.dumps(root,sort_keys=True,separators=(',',':')).encode())==claimed,'input root')
    check(job['inputs_root_sha256']==claimed,'remote input identity')
    for name,expected in inputs['files'].items():check(sha(a.source/name)==expected,'source '+name)
    for item in job['artifacts']:
        path=a.results/'output'/item['name'];check(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'transfer '+item['name'])
    pool_sha=sha(a.recorded_pool);check(pool_sha=='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db','recorded pool identity')
    pool=np.memmap(a.recorded_pool,dtype=np.uint8,mode='r');references={'skip':(388096,70778880),'encoder':(71166976,17694720)}
    expected={k:digest(pool[off:off+n]) for k,(off,n) in references.items()};runs=[];pids=[];features=[]
    for kind in ('original','feature-tap'):
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                prefix=a.results/'output'/f'{kind}-{poison}-{ordinal}';r=read(prefix.with_suffix('.json'));pids.append(r['pid'])
                check(r['result']=='pass' and r['tap_coverage_errors']==0 and r['canary_errors']==0 and r['immutable_errors']==0,'execution gates '+prefix.name)
                check(r['poison']==int(poison,16) and r['tap_enabled']==(kind=='feature-tap') and r['expected_gpu_input'] is False,'configuration '+prefix.name)
                mismatches={};hashes={}
                for k,(off,n) in references.items():
                    path=prefix.with_suffix('.'+k+'.e4m3');hashes[k]=sha(path)
                    actual=np.memmap(path,dtype=np.uint8,mode='r');check(actual.size==n,'output extent '+prefix.name+k)
                    mismatches[k]=int(np.count_nonzero(actual!=pool[off:off+n]));check(mismatches[k]==0,'RTX replay output '+prefix.name+k)
                if kind=='feature-tap':
                    path=prefix.with_suffix('.features-shared.f16');features.append(sha(path));check(path.stat().st_size==70778880,'tap extent')
                    coverage=np.memmap(prefix.with_suffix('.coverage.bin'),dtype='<u4',mode='r')
                    check(coverage.size==70778880//4 and np.all(coverage==1),'actual tap coverage '+prefix.name)
                runs.append({'run':prefix.name,'pid':r['pid'],'mismatching_bytes':mismatches,'output_sha256':hashes})
    check(len(set(pids))==8,'fresh processes');check(len(set(features))==1,'feature repeat and poison invariance')
    r={'schema':'neuralstudio.nr.pre-oracle-verification.v1','inputs_root_sha256':claimed,'job_receipt_sha256':sha(a.results/'receipt.json'),
       'recorded_pool_sha256':pool_sha,'expected_output_sha256':expected,'feature_sha256':features[0],
       'runs':runs,'errors':errors,'result':'fail' if errors else 'pass',
       'scope':'Selected PTX replay and diagnostic tap only. No native Mac Block0 parity. Texture mode is consistent with this pinned image, not universally recovered.',
       'full_model_inference':False}
    r['receipt_root_sha256']=digest(json.dumps(r,sort_keys=True,separators=(',',':')).encode());a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2));return bool(errors)
if __name__=='__main__':raise SystemExit(main())
