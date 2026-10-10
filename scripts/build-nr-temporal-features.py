#!/usr/bin/env python3
"""Freeze shared peer helpers and compile an unpromoted temporal feature candidate."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rtx-fitted',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[1];assert not a.output.exists();a.output.mkdir(parents=True);source=a.output/'source';source.mkdir()
    rt=a.peer_root/'src/nr_ptx/nr_ptx_rt.metal';sampler=a.peer_root/'src/nr_temporal/nr_tex_sample.metal';rom=a.peer_root/'build/nr_ptx/tables/sm120-rcp-correction-rom.bin'
    pins={str(x):sha(x) for x in (rt,sampler,rom)};text=rt.read_text();start=text.index('inline float nr_rne_rcp_f32(');end=text.index('inline float nr_rcp_approx2(',start)
    (source/'nr_pre_temporal_support.h').write_text('#pragma once\n'+text[start:end]);shutil.copyfile(sampler,source/sampler.name);shutil.copyfile(repo/'src/nr_pre/pre_temporal_features.metal',source/'pre_temporal_features.metal');shutil.copyfile(rom,a.output/rom.name)
    commands=[]
    variants=[('hardware',[]),('q8-truncate',['-DNR_PRE_HISTORY_Q8=1','-DNR_TEX_Q8_MODE=0']),('q8-round',['-DNR_PRE_HISTORY_Q8=1','-DNR_TEX_Q8_MODE=1'])]
    if a.rtx_fitted:variants.append(('rtx-fitted',['-DNR_PRE_HISTORY_RTX=1']))
    for name,flags in variants:
        air=a.output/(name+'.air');lib=a.output/(name+'.metallib')
        for cmd in [['xcrun','-sdk','macosx','metal','-std=metal3.1','-fno-fast-math','-ffp-contract=off',*flags,'-c',str(source/'pre_temporal_features.metal'),'-o',str(air)],['xcrun','-sdk','macosx','metallib',str(air),'-o',str(lib)]]:
            with (a.output/(name+('-compile.log' if '-c' in cmd else '-link.log'))).open('x') as log:r=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            commands.append({'command':cmd,'exit_code':r.returncode})
            if r.returncode:break
        if r.returncode:break
    for p,h in pins.items():assert sha(Path(p))==h,'peer helper changed during freeze'
    result={'schema':'neuralstudio.nr.temporal-feature-build.v1','result':'compiled_candidate' if r.returncode==0 else 'compile_failed','peer_inputs':pins,'commands':commands,'files':{str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()},'native_execution':False,'parity_accepted':False,'production_changed':False,'scope':'Fixed full-size/no-depth/no-mask pre history composition, consumes freshly generated base features for the current noise index. Hardware and Q8 samplers remain experimental; not wired into the accepted runtime.'}
    result['receipt_root_sha256']=canonical(result);(a.output/'build.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':result['result'],'root':result['receipt_root_sha256']}))
    if r.returncode:raise SystemExit(3)
if __name__=='__main__':main()
