#!/usr/bin/env python3
"""Derive a motion-specific encoder experiment from the immutable accepted build.

Motion hash is admission only, never a correction or output lookup.
"""
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
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('source anchor changed '+a[:80])
    return s.replace(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--features-build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=repo/'build/nr_encoder_temporal/build-v2-sass';pb=sealed(parent/'build.json');fb=sealed(a.features_build/'build.json');ref=sealed(repo/'build/nr_motion_reference/matrix-audit-v1.json')
    assert pb['receipt_root_sha256']=='b2628242a1e425ec460cd2cb5e445da64283d736a3c5b1789bcecfa127c84c9f'
    assert ref['receipt_root_sha256']=='da60f9d5cfc806e5b3dbba313f4a6adbe1e2478af829e63065966b68bd87150b'
    assert fb['result']=='compiled_candidate' and 'rtx-fitted.metallib' in fb['files']
    for n,h in pb['source_files'].items():assert sha(parent/'source'/n)==h
    for n,h in pb['assets'].items():assert sha(parent/'assets'/n)==h
    for n,h in fb['files'].items():assert sha(a.features_build/n)==h
    assert not a.output.exists();a.output.mkdir(parents=True);source=a.output/'source';assets=a.output/'assets';shutil.copytree(parent/'source',source);shutil.copytree(parent/'assets',assets)
    inc=source/'nr_encoder_temporal_runtime.inc';s=inc.read_text()
    s=once(s,'"temporal-q8-round.metallib"};','"temporal-q8-round.metallib","temporal-rtx-fitted.metallib"};')
    s=once(s,'in.sampler_variant>2','in.sampler_variant>3')
    s=once(s,'    for(auto b:motion)if(b)throw std::runtime_error("nonzero motion not admitted by this experimental reference");', '''    bool nonzeroMotion=false;for(auto b:motion)nonzeroMotion|=(b!=0);
    // Admission restriction for this experiment, not an arithmetic correction.
    if(nonzeroMotion && (in.sampler_variant!=3 || motionHash!="'''+ref['motion_sha256']+'''"))
      throw std::runtime_error("nonzero motion outside the verified experimental input class");''')
    inc.write_text(s)
    header=source/'nr_encoder_temporal_runtime.h';s=header.read_text().replace('2 Q8 nearest; unaccepted candidates','2 Q8 nearest, 3 fitted RTX (new unpromoted motion candidate)').replace('full-size shared textures and zero motion.','full-size shared textures; modes0..2 remain zero-motion-only. Mode3 additionally admits the pinned uniform(0.25,-0.5) input for validation.');header.write_text(s)
    dest=assets/'temporal-rtx-fitted.metallib';shutil.copyfile(a.features_build/'rtx-fitted.metallib',dest)
    hashes=dict(pb['assets']);hashes[dest.name]=sha(dest)
    asset_header=source/'generated/asset_hashes.h';s=asset_header.read_text();s=once(s,'};','  {"'+dest.name+'","'+hashes[dest.name]+'"},\n};');asset_header.write_text(s)
    command=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(source/'nr_encoder_runtime.mm'),'-o',str(a.output/'nr_encoder_runtime.o')]
    with (a.output/'compile.log').open('x') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    r={'schema':'neuralstudio.nr.motion-encoder-build.v1','result':'compiled_experimental' if run.returncode==0 else 'compile_failed','parent_root':pb['receipt_root_sha256'],'features_root':fb['receipt_root_sha256'],'motion_reference_root':ref['receipt_root_sha256'],'motion_sha256':ref['motion_sha256'],'assets':hashes,'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'command':command,'exit_code':run.returncode,'production_changed':False,'runtime_parity':False}
    if run.returncode==0:r['object_sha256']=sha(a.output/'nr_encoder_runtime.o')
    r['receipt_root_sha256']=canonical(r);(a.output/'build.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'result':r['result'],'root':r['receipt_root_sha256']}))
    if run.returncode:raise SystemExit(3)
if __name__=='__main__':main()
