#!/usr/bin/env python3
"""Local CPU test of original-image case generation; no model/GPU execution."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    repo=Path(__file__).resolve().parents[1];source=a.output/'source';source.mkdir()
    for name in ('input_cases.h','input_cases_probe.cpp'):shutil.copyfile(repo/'src/nr_pre'/name,source/name)
    command=['xcrun','clang++','-std=c++20','-O2','-Wall','-Wextra','-Werror',str(source/'input_cases_probe.cpp'),'-o',str(a.output/'probe')]
    with (a.output/'compile.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (a.output/'run.log').open('x') as log:subprocess.run([a.output/'probe',a.output/'cases'],stdout=log,stderr=subprocess.STDOUT,check=True)
    images=[]
    for i in range(4):
        file=a.output/'cases'/f'case{i}.rgba16le';x=np.fromfile(file,dtype='<u2').reshape(1080,1920,4)
        assert file.stat().st_size==16588800 and np.all(x[:,:,3]==0x3c00)
        assert np.all((x&0x7c00)!=0x7c00) and np.all(x[:,:,:3]<=0x3c00)
        if i==0:assert sha(file)=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342'
        if i==1:assert not np.any(x[:,:,:3])
        images.append({'case':i,'sha256':sha(file),'rgb_unique_half_values':int(np.unique(x[:,:,:3]).size)})
    assert len({x['sha256'] for x in images})==4
    repeat=subprocess.run([a.output/'probe',a.output/'cases'],capture_output=True,text=True)
    assert repeat.returncode!=0 and 'preserve output' in repeat.stderr
    r={'schema':'neuralstudio.nr.color-case-cpu-test.v1','result':'pass','cases':images,'invalid_cases_rejected':3,'existing_output_rejected':True,
       'source_sha256':{x.name:sha(x) for x in source.iterdir()},'probe_sha256':sha(a.output/'probe'),'compile_command':command,
       'rtx_executed':False,'mac_model_executed':False,'scope':'CPU input generation only; original control image hash matches, finite colors, distinct cases, fail-closed arguments/output preservation.'}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
