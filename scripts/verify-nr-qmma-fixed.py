#!/usr/bin/env python3
"""Verify the handwritten Metal primitive on 65 independently captured MMAs."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists();repo=Path(__file__).resolve().parents[1]
    parents={'discriminator':(64,'d69edbc0a427bfb4bd7247317d5c13e239cee4de67c5e456ca8b4518169c0f7d','da08b7712c437c67ff13d5366607e18abaac3fa4d9e220ae42cbfbc26a12d391','build/nr_qmma_operands/discriminators-rtx-v1/verification.json','2ab6fe3da7c05a8e76ef8dc73852cbbec32e1e57e02a31bfe8d6948d5ac371e8'),
             'launch151':(1,'d65f92f913f6ebdb88c14494a500ddefc7287f8e73f3f7b258f1ea38e4f30703','91842413ae4708274893f24203dbb3d1dfcd5e46113b46bb2a260a17ac1cb846','build/nr_qmma_operands/launch151-rtx-v1/verification.json','2cbe9f6c668e06e557181c618b92c5706733a7b623282adc324adf13fa605b77')}
    runs=[];pids=set()
    for kind,(cases,input_sha,expected,parent,root) in parents.items():
        r=json.loads((repo/parent).read_text());body=dict(r);got=body.pop('receipt_root_sha256');assert canonical(body)==got==root and r['result']=='pass'
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                path=a.attempt/f'{kind}-{poison}-{ordinal}';r=json.loads((path/'run.json').read_text())
                assert r['result']=='pass' and r['cases']==cases and r['poison']==int(poison,16) and r['pid'] not in pids;pids.add(r['pid'])
                assert r['input_sha256']==input_sha and r['expected_sha256']==r['output_sha256']==sha(path/'output.bin')==expected
                assert r['mismatching_words']==r['coverage_errors']==r['canary_errors']==0 and r['input_immutable'] is True and r['expected_gpu_input'] is False
                counts=np.fromfile(path/'coverage.bin',dtype='<u4');assert counts.size==cases*64 and np.all(counts==1)
                runs.append({'run':path.name,**r,'receipt_sha256':sha(path/'run.json')})
    files={str(p.relative_to(a.attempt)):sha(p) for p in sorted((a.attempt/'source').rglob('*')) if p.is_file()};files.update({n:sha(a.attempt/n) for n in ('probe','probe.metallib')})
    r={'schema':'neuralstudio.nr.handwritten-fixed-qmma-verification.v1','runs':runs,'files':files,'output_bytes':4*65*256,
       'result':'pass','scope':'65 independent finite operand sets, four fresh dual-poison runs each. Not exhaustive arithmetic-domain, model or performance proof.'}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
