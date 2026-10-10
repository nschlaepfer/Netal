#!/usr/bin/env python3
"""Bound full endpoint differences by independently observed feature windows.

Pure offline comparison. Does not execute or modify a model, or feed oracle
features into the native core. Coincidence within a window is not causal proof.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('run',type=Path);p.add_argument('rtx_features',type=Path);p.add_argument('pool',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(),'preserve analysis'
    assert sha(a.rtx_features)=='49abcec0ab2e1068a03339564aa99c5358c0720eb06fe4930ccc803aa2b0a06f'
    assert sha(a.pool)=='1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db'
    actual=np.memmap(a.run/'features-shared.f16',dtype='<u2',mode='r').reshape(-1,1024)
    ref=np.memmap(a.rtx_features,dtype='<u2',mode='r').reshape(-1,1024)
    feature_count=np.count_nonzero(actual!=ref,axis=1);bad=feature_count!=0
    pool=np.memmap(a.pool,dtype=np.uint8,mode='r');result={}
    for name,offset,size in [('skip',388096,70778880),('encoder',71166976,17694720)]:
        y=np.memmap(a.run/(name+'.e4m3'),dtype=np.uint8,mode='r');assert y.size==size
        indices=np.flatnonzero(y!=pool[offset:offset+size])
        if name=='skip':
            cell=indices//512;windows=(cell//480//2)*240+(cell%480//2)
        else:
            pixel=(indices%(960*576*16))//16;windows=((pixel//960)//4)*240+(pixel%960)//4
        outside=~bad[windows]
        result[name]={'mismatching_bytes':int(indices.size),'mismatching_windows':int(np.unique(windows).size),
                      'mismatching_bytes_outside_feature_failure_windows':int(np.count_nonzero(outside)),
                      'outside_counterexamples':[{'offset':int(i),'window':int(w),'actual':int(y[i]),'expected':int(pool[offset+i])} for i,w in zip(indices[outside][:32],windows[outside][:32])],
                      'output_sha256':sha(a.run/(name+'.e4m3'))}
    r={'schema':'neuralstudio.nr.pre-feature-spatial-bound.v1','run_receipt_sha256':sha(a.run/'run.json'),
       'feature_sha256':sha(a.run/'features-shared.f16'),'rtx_feature_sha256':sha(a.rtx_features),
       'windows':int(bad.size),'feature_mismatching_windows':int(np.count_nonzero(bad)),
       'feature_exact_windows':int(np.count_nonzero(~bad)),'outputs':result,
       'interpretation':'Differences outside feature-failure windows require another explanation. Differences only inside are consistent with noise propagation, not proof that the core has no other error.',
       'model_parity':False}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
