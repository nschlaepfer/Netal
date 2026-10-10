#!/usr/bin/env python3
"""Seal the false-positive history continuity check across eight native runs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r): return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p): return json.loads(Path(p).read_text())
def require(v,label):
    if not v: raise ValueError(label)
def classify(history,output):
    require(history.shape==output.shape,'same bit extent')
    mismatch=history!=output
    return int(mismatch.sum()), bool(np.all(history[mismatch]==0) and np.all(output[mismatch]==0x8000))

def audit(peer):
    repo=Path(__file__).resolve().parents[1]; rtx_path=repo/'build/nr_reset_history/matrix-audit-v1.json'; rtx=read(rtx_path); body=dict(rtx); seal=body.pop('receipt_root_sha256')
    require(canonical(body)==seal=='b2f44873eebc444a53784b2a2eaf2d4b5f221174436cb1c14fe2f450c1bf2ddf','RTX reference seal')
    expected=rtx['outputs'][1]; rows=[]; pids=set()
    for version in (2,3):
        for pair in ('a5-a5','5a-5a','a5-5a','5a-a5'):
            folder=peer/f'build/nr_full/temporal-seq-v{version}'/pair; rp=folder/'receipt.json'; r=read(rp); f=r['frames'][1]
            require(r['pid']>0 and r['pid'] not in pids,'eight distinct processes'); pids.add(r['pid'])
            require(r['result']==f['result']=='pass' and f['body']['code']==0 and not f['body']['check_failures'],'false positive was claimed')
            actual=folder/'diag/frame-1/tex_out_156.bin'; output=folder/'final-1.rgba16le'; ref=Path(f['reference']['file'])
            require(actual.stat().st_size==output.stat().st_size==ref.stat().st_size==16588800,'frame extent')
            require(sha(output)==sha(ref)==expected==f['surface_sha256_caller_readback']==f['body']['history_sha256_after'],'caller/reference/claimed state equality')
            h=sha(actual); require(h=='9fa7e06cf26e713a6611322b83d1e9079df0e01f8bde72cda5a4b08edad6dbcf' and h!=expected,'actual history divergence')
            a=np.fromfile(actual,dtype='<u2'); b=np.fromfile(output,dtype='<u2'); count,zero_sign_only=classify(a,b)
            require(count==13485 and zero_sign_only,'all and only negative-zero signs lost')
            rows.append({'matrix_version':version,'pair':pair,'pid':r['pid'],'receipt_sha256':sha(rp),'actual_history_sha256':h,'caller_output_sha256':expected,'claimed_history_sha256':f['body']['history_sha256_after'],'different_half_values':count,'only_negative_zero_to_positive_zero':zero_sign_only})
    return {'schema':'neuralstudio.nr.history-copy-counterexample.v1','result':'reproduced_false_positive_history_check','runs':rows,'RTX_reference_root':seal,'display_output_parity':True,'temporal_history_parity':False,'full_goal_complete':False,'cause':'After reading/dumping actual history, runtime v6/v7 swaps caller output into o; the subsequent hsha hashes o, now caller output. Thus osha and hsha both hash caller bytes.','required_fix':'Keep independent history and caller bytes/hashes; validate before advancing state; fix copy156 signed-zero semantics against RTX, then repeat live state tests.','limits':['Original image, two calls, zero motion, pinned v9t/v10t only.','Does not negate exact frame0/frame1 caller outputs; disproves actual frame1 history equality and receipt validity for that state.'],'verifier_sha256':sha(Path(__file__))}

def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior evidence')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
