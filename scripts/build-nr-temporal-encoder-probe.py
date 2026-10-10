#!/usr/bin/env python3
"""Link the experimental encoder with a frozen accepted body for live history."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--encoder-build',type=Path,required=True);p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--probe-source',type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    r=json.loads((a.encoder_build/'build.json').read_text());b=dict(r);root=b.pop('receipt_root_sha256');assert canonical(b)==root and r['result']=='compiled_experimental' and sha(a.encoder_build/'nr_encoder_runtime.o')==r['object_sha256']
    source_parent=a.peer_root/'build/nr_full/v4/source/src/nr_body';assert sha(source_parent/'nr_body_runtime_v4.mm')=='ab53f3c85c6cec84f8a4129f4ed4ac40d7be51df23aae3413ea4d3730a1d996f'
    assert not a.output.exists();a.output.mkdir(parents=True);source=a.output/'source';source.mkdir()
    for name in ('nr_body_runtime.h','nr_body_runtime_v4.mm'):shutil.copyfile(source_parent/name,source/name)
    for name in ('nr_encoder_runtime.h','nr_encoder_temporal_runtime.h','nr_temporal.h'):shutil.copyfile(a.encoder_build/'source'/name,source/name)
    shutil.copyfile(a.probe_source or repo/'src/nr_encoder_runtime/nr_encoder_temporal_probe.mm',source/'probe.mm');shutil.copyfile(a.encoder_build/'nr_encoder_runtime.o',a.output/'nr_encoder_runtime.o')
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-I',str(source),str(source/'probe.mm'),str(source/'nr_body_runtime_v4.mm'),str(a.output/'nr_encoder_runtime.o'),'-framework','Foundation','-framework','Metal','-o',str(a.output/'probe')]
    with (a.output/'compile.log').open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    record={'schema':'neuralstudio.nr.temporal-encoder-probe-build.v1','encoder_root':root,'encoder_build':str(a.encoder_build.resolve()),'command':cmd,'exit_code':run.returncode,'files':{str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()},'live_history_origin':'native encoder+accepted body v4 -> GPU output -> separate GPU history copy','archived_history_inputs':False,'native_validation':False}
    record['receipt_root_sha256']=canonical(record);(a.output/'build.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps({'exit_code':run.returncode,'root':record['receipt_root_sha256']}))
    if run.returncode:raise SystemExit(3)
if __name__=='__main__':main()
