#!/usr/bin/env python3
"""Stage a bounded register probe; expected answers never enter the CUDA inputs."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('output',type=Path)
    a=p.parse_args();root=Path(__file__).resolve().parents[1]
    spec=json.loads(a.spec.read_text());assert spec['schema'] in ('neuralstudio.nr.mma-operand-probe.v1','neuralstudio.nr.mma-operand-probe.v2')
    cases=spec.get('cases',[spec]);assert 1<=len(cases)<=1024
    words=[]
    for case in cases:
        assert [x['lane'] for x in case['lanes']]==list(range(32))
        for lane in case['lanes']:
            assert [len(lane[k]) for k in ('a','b','c')]==[4,2,2]
            words += [int(x,16) for k in ('a','b','c') for x in lane[k]]
    a.output.mkdir(parents=True,exist_ok=False)
    (a.output/'operands.bin').write_bytes(struct.pack(f'<{len(words)}I',*words))
    shutil.copyfile(root/'src/nr_probes/qmma_operands.cu',a.output/'qmma_operands.cu')
    shutil.copyfile(root/'scripts/run-nr-qmma-operands.ps1',a.output/'run.ps1')
    receipt={'schema':'neuralstudio.nr.qmma-operands-inputs.v1','spec_sha256':sha(a.spec),
             'repository_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),
             'launch':spec.get('launch'),'cta':spec.get('cta'),'pc':spec.get('pc'),'cases':len(cases),
             'files':{x.name:sha(x) for x in sorted(a.output.iterdir())},'expected_answers_staged':False}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'inputs.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))

if __name__=='__main__':main()
