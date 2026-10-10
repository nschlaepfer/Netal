#!/usr/bin/env python3
"""Extend only admission to one verified spatial input; frozen shaders unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(Path(p).read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=repo/'build/nr_encoder_temporal/motion-v1';pb=sealed(parent/'build.json');ref=sealed(repo/'build/nr_spatial_motion/matrix-audit-v1.json')
    assert pb['receipt_root_sha256']=='1b42bb34e487a0817c7ee7491565e88f3780b0afad724f906c452377759a6cd3' and ref['receipt_root_sha256']=='c055846cdf9534bc91782fedf6a8b05b3c4aab2163da20e4dd88d5efba0e445e'
    for n,h in pb['source_files'].items():assert sha(parent/'source'/n)==h
    for n,h in pb['assets'].items():assert sha(parent/'assets'/n)==h
    assert not a.output.exists();a.output.mkdir(parents=True);source=a.output/'source';assets=a.output/'assets';shutil.copytree(parent/'source',source);shutil.copytree(parent/'assets',assets)
    inc=source/'nr_encoder_temporal_runtime.inc';s=inc.read_text();old='motionHash!="44150d0bee7989b32461d371c352a325ef7f615e43fe4bbff8ecc5bbc629adce"';assert s.count(old)==1
    s=s.replace(old,'('+old+' && motionHash!="'+ref['motion_sha256']+'")');inc.write_text(s)
    header=source/'nr_encoder_temporal_runtime.h';s=header.read_text().replace('the pinned uniform(0.25,-0.5) input for validation','the pinned uniform(0.25,-0.5) and signed spatial1/16-grid inputs for validation');header.write_text(s)
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(source/'nr_encoder_runtime.mm'),'-o',str(a.output/'nr_encoder_runtime.o')]
    with (a.output/'compile.log').open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    r={'schema':'neuralstudio.nr.spatial-encoder-build.v1','result':'compiled_experimental' if run.returncode==0 else 'compile_failed','parent_root':pb['receipt_root_sha256'],'motion_reference_root':ref['receipt_root_sha256'],'motion_sha256':ref['motion_sha256'],'assets':pb['assets'],'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'command':cmd,'exit_code':run.returncode,'shaders_changed':False,'production_changed':False,'runtime_parity':False}
    if run.returncode==0:r['object_sha256']=sha(a.output/'nr_encoder_runtime.o')
    r['receipt_root_sha256']=canonical(r);(a.output/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'result':r['result'],'root':r['receipt_root_sha256']}))
    if run.returncode:raise SystemExit(3)
if __name__=='__main__':main()
