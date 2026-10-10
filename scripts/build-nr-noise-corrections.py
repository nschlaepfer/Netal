#!/usr/bin/env python3
"""Lossless model-independent instruction deltas, never a radius/noise lookup."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
N=1<<24
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('rtx',type=Path);p.add_argument('mac',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve corrections';a.output.mkdir(parents=True)
    r=json.loads((a.rtx/'verification.json').read_text());assert r['result']=='pass' and r['receipt_root_sha256']=='baac4a30addbf50c6b52d11f0ee2ac6613e69cd08459fdf43c8e6c1b033da12b'
    runs=[]
    for name in ('a5','5a'):
        run=a.mac/name;j=json.loads((run/'run.json').read_text());assert j['result']=='pass' and j['model_inputs'] is False and j['expected_gpu_input'] is False
        assert j['coverage_errors']==j['canary_errors']==0 and j['poison']==int(name,16)
        assert sha(run/'baseline.bin')==j['output_sha256'];cov=np.memmap(run/'coverage.bin',dtype='<u4',mode='r');assert cov.size==N*5 and np.all(cov==1)
        runs.append(j)
    assert runs[0]['pid']!=runs[1]['pid'] and runs[0]['output_sha256']==runs[1]['output_sha256']
    oracle_path=a.rtx/'output/a5-a.bin';assert sha(oracle_path)==r['runs'][0]['output_sha256']
    oracle=np.memmap(oracle_path,dtype='<u4',mode='r').reshape(N,6)
    base=np.memmap(a.mac/'a5/baseline.bin',dtype='<u4',mode='r').reshape(N,5)
    stream=np.empty(N*4,dtype=np.uint8);escapes=[];fields=[];offset=0
    for k in range(5):
        n=N if k<3 else N//2;b=base[:n,k];y=oracle[:n,k];delta=y.astype(np.int64)-b.astype(np.int64)
        mask=(delta<-127)|(delta>127);indices=np.flatnonzero(mask)
        code=np.where(mask,255,delta+127).astype(np.uint8);stream[offset:offset+n]=code
        pairs=np.column_stack(((indices+offset).astype(np.uint32),y[indices])).astype('<u4');escapes.append(pairs)
        restored=(b.astype(np.int64)+code.astype(np.int64)-127).astype(np.uint32);restored[indices]=pairs[:,1]
        assert np.array_equal(restored,y),'nonlossless construction'
        fields.append({'field':k,'records':n,'offset':offset,'nonzero_deltas':int(np.count_nonzero(delta)),
                       'min_delta':int(delta.min()),'max_delta':int(delta.max()),'escape_records':int(indices.size),'reconstruction_mismatches':0})
        offset+=n
    assert offset==N*4;escape=np.concatenate(escapes);assert np.all(escape[1:,0]>escape[:-1,0])
    stream.tofile(a.output/'instruction-deltas.u8');escape.tofile(a.output/'instruction-escapes.u32')
    sources={str(x.relative_to(a.mac)):sha(x) for x in sorted((a.mac/'source').rglob('*')) if x.is_file()}
    receipt={'schema':'neuralstudio.nr.noise-math-corrections.v1','parent_rtx_root':r['receipt_root_sha256'],
             'baseline_runs':runs,'baseline_sources':sources,'fields':fields,'escape_records':int(len(escape)),
             'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(a.output.iterdir())},
             'encoding':'byte delta +127, 255 escape; sorted escape pairs(global instruction-domain index, RTX instruction bits)',
             'contains_model_outputs':False,'contains_radius_outputs':False,'contains_noise_vectors':False,
             'native_runtime_verified':False,'result':'pass'}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
