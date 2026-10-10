#!/usr/bin/env python3
"""Derive an experimental entry point; never rewrite the accepted encoder build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def once(s,a,b):
    assert s.count(a)==1,a[:80];return s.replace(a,b)
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--features-build',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    parent=repo/'build/nr_encoder_runtime/scratch-candidate-v1';pb=sealed(parent/'build.json');features=a.features_build or repo/'build/nr_pre_temporal/build-v2';fb=sealed(features/'build.json')
    assert pb['receipt_root_sha256']=='5a875d6cc7150a354061792427837f61bd57f9d325bfe4739ff7acfa9e949781' and fb['result']=='compiled_candidate'
    for n,h in pb['source_files'].items():assert sha(parent/'source'/n)==h
    for n,h in pb['assets'].items():assert sha(parent/'assets'/n)==h
    for n,h in fb['files'].items():assert sha(features/n)==h
    assert not a.output.exists();a.output.mkdir(parents=True);source=a.output/'source';assets=a.output/'assets';shutil.copytree(parent/'source',source);shutil.copytree(parent/'assets',assets)
    for n in ('nr_encoder_temporal_runtime.h','nr_encoder_temporal_runtime.inc'):shutil.copyfile(repo/'src/nr_encoder_runtime'/n,source/n)
    temporal=a.peer_root/'src/nr_temporal/nr_temporal.h';temporal_sha=sha(temporal);shutil.copyfile(temporal,source/'nr_temporal.h')
    host=source/'nr_encoder_runtime.mm';s=host.read_text();s=once(s,'#include "nr_encoder_runtime.h"','#include "nr_encoder_runtime.h"\n#include "nr_encoder_temporal_runtime.h"')
    s=once(s,'struct PreOutputs {','static thread_local const NrEncoderTemporalFrame* NrTemporalInput=nullptr;\nstatic Buffer ApplyTemporalFeatures(NrEncoderState&,Buffer&,uint8_t);\nstruct PreOutputs {')
    guard='if(!in.reset||in.frame_index!=0)throw std::runtime_error("unverified temporal/frame contract");'
    s=once(s,guard,'if(NrTemporalInput){if(&NrTemporalInput->frame!=&in||in.reset||in.frame_index==0)throw std::runtime_error("temporal entry mismatch");}\n    else '+guard)
    s+='\n#include "nr_encoder_temporal_runtime.inc"\n';host.write_text(s)
    pre=source/'generated/pre.inc';s=pre.read_text();s=once(s,'PreShape shape{1920,1080,1152,0,1};','PreShape shape{1920,1080,1152,NrTemporalInput?NrTemporalInput->frame.frame_index:0u,1};')
    s=once(s,'  auto featureHash=Sha(Data(features),features.bytes);','  if(NrTemporalInput){Write(out/"features-without-history.f16",features);features=ApplyTemporalFeatures(state,features,poison);}\n  auto featureHash=Sha(Data(features),features.bytes);');pre.write_text(s)
    hashes=dict(pb['assets'])
    for name in ('hardware','q8-truncate','q8-round'):
        dest=assets/('temporal-'+name+'.metallib');shutil.copyfile(features/(name+'.metallib'),dest);hashes[dest.name]=sha(dest)
    shutil.copyfile(features/'sm120-rcp-correction-rom.bin',assets/'temporal-rcp-rom.bin');hashes['temporal-rcp-rom.bin']=sha(assets/'temporal-rcp-rom.bin')
    (source/'generated/asset_hashes.h').write_text('#pragma once\nstatic const char* NrEncoderParentRoot="'+pb['parent_root']+'";\nstatic const std::pair<const char*,const char*> NrEncoderAssetHashes[]={\n'+''.join('  {"'+n+'","'+h+'"},\n' for n,h in sorted(hashes.items()))+'};\n')
    cmd=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations','-c',str(host),'-o',str(a.output/'nr_encoder_runtime.o')]
    with (a.output/'compile.log').open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    assert sha(temporal)==temporal_sha,'peer state header changed during build'
    result={'schema':'neuralstudio.nr.temporal-encoder-build.v1','result':'compiled_experimental' if r.returncode==0 else 'compile_failed','parent_root':pb['receipt_root_sha256'],'features_root':fb['receipt_root_sha256'],'temporal_header_sha256':temporal_sha,'assets':hashes,'source_files':{str(p.relative_to(source)):sha(p) for p in source.rglob('*') if p.is_file()},'command':cmd,'exit_code':r.returncode,'runtime_parity':False,'production_changed':False}
    if r.returncode==0:result['object_sha256']=sha(a.output/'nr_encoder_runtime.o')
    result['receipt_root_sha256']=canonical(result);(a.output/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':result['result'],'root':result['receipt_root_sha256']}))
    if r.returncode:raise SystemExit(3)
if __name__=='__main__':main()
