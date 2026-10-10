#!/usr/bin/env python3
"""Pin all 2048 real norm-FMA triples and a finite-half fixed-addend square sweep."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r): return hashlib.sha256(json.dumps(r, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    qpath=repo/'build/nr_sixteen/window-replay-v1/a5-a/first-window-qkv.f16'
    audit=json.loads((repo/'build/nr_sixteen/window-replay-v1/audit.json').read_text())
    assert sha(qpath)==audit['runs'][0]['artifacts'][qpath.name]
    q=np.fromfile(qpath,'<f2').reshape(64,96);triples=[];witness=None
    for row in range(64):
        for component in range(2):
            v=q[row,component*32:component*32+32]
            for k in range(8):
                for i,j in ((8+k,24+k),(k,16+k)):
                    vbits=int(v[i].view('uint16'));c=int(np.float16(float(v[j])*float(v[j])).view('uint16'))
                    if row==39 and component==1 and k==2 and i==10:witness=len(triples)
                    triples.append((vbits,vbits,c))
    assert len(triples)==2048 and triples[witness]==(0xc4c1,0xc4c1,0x17f1)
    triples += [(b,b,0x17f1) for b in range(65536) if b&0x7c00!=0x7c00]
    assert len(triples)==65536
    assert not a.output.exists();a.output.mkdir(parents=True)
    np.asarray(triples,dtype='<u2').tofile(a.output/'operands.bin')
    shutil.copyfile(repo/'src/nr_probes/nr_half_fma_oracle.cu',a.output/'half_fma_oracle.cu')
    run=(repo/'scripts/run-nr-qmma-operands.ps1').read_text()
    run=run.replace('nr-qmma-operands-','nr-half-fma-').replace('qmma_operands.cu','half_fma_oracle.cu').replace('neuralstudio.nr.qmma-operands-job.v1','neuralstudio.nr.half-fma-job.v1')
    (a.output/'run.ps1').write_text(run)
    r={'schema':'neuralstudio.nr.half-fma-fixtures.v1','source_qkv_sha256':sha(qpath),'witness_index':witness,
       'real_norm_triples':2048,'fixed_addend_finite_half_square_sweep':63488,'count':65536,
       'scope':'Primitive diagnostic only; real-window operand replay plus every finite half a with fixed c=0x17f1. Not arbitrary half FMA coverage.',
       'expected_bytes_staged':False,'files':{p.name:sha(p) for p in a.output.iterdir()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
