#!/usr/bin/env python3
"""Diagnostic decomposition of the accepted encoder, never a speedup benchmark.

Run under the shared GPU lock. Profiling enables output-only diagnostics and
retains the same correctness checks. GPU times exclude the two input blits;
wall time includes diagnostic file writes. Do not sum nested host/GPU timings.
"""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--probe-build',required=True,type=Path);p.add_argument('--runtime-build',type=Path);p.add_argument('--probe-source',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.probe_build=a.probe_build.resolve()
    runtime=a.runtime_build.resolve() if a.runtime_build else repo/'build/nr_encoder_runtime/build-v1';build=sealed(runtime/'build.json');assert sha(runtime/'nr_encoder_runtime.o')==build['object_sha256']
    probe_source=a.probe_source.resolve() if a.probe_source else a.probe_build/'source/probe.mm'
    for n,h in build['source_files'].items():assert sha(runtime/'source'/n)==h
    for n,h in build['assets'].items():assert sha(runtime/'assets'/n)==h
    original=repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le';dense=repo/'build/nr_color_cases/rtx-v1/case3-retry-v1/input-color.rgba16le'
    refs=[repo/'build/nr_encoder_fscale/live-v1/a5-a',repo/'build/nr_color_encoder/case3-live-v1/a5-a']
    assert not a.output.exists();a.output.mkdir(parents=True);assert shutil.disk_usage(a.output).free>=15*1024**3,'less than15GiB before estimated1.6GiB retained run'
    material=[runtime/'nr_encoder_runtime.o',a.probe_build/'probe',probe_source,runtime/'source/nr_encoder_runtime.h',original,dense]+[runtime/'assets'/n for n in build['assets']]+[runtime/'source'/n for n in build['source_files']]
    paths={'block0-skip':'block0/skip.e4m3',**{f'block{b}-native':f'encoder/block{b}-native.e4m3' for b in (4,8,14,22)},'block22-middle':'encoder/block22-middle.e4m3'}
    for ref in refs:material.extend(ref/name for name in paths.values())
    pins={str(f):sha(f) for f in material};command=[str(a.probe_build/'probe'),str(runtime/'assets'),str(original),str(refs[0]),str(dense),str(refs[1]),'a5',str(a.output/'run'),'--profile']
    plan={'schema':'neuralstudio.nr.encoder-profile-plan.v1','runtime_build_root':build['receipt_root_sha256'],'artifacts':pins,'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'timing_scope':'diagnostic, under ambient desktop load; not quiet timing or speedup acceptance'}
    plan['receipt_root_sha256']=canonical(plan);(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (a.output/'run.log').open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    (a.output/'execution.json').write_text(json.dumps({'exit_code':r.returncode,'ended_utc':datetime.now(timezone.utc).isoformat()},indent=2)+'\n');assert r.returncode==0,'preserve failed profiling run'
    run=json.loads((a.output/'run/run.json').read_text());assert run['result']=='pass' and len(run['frames'])==2
    results=[]
    for i,frame in enumerate(run['frames']):
        assert frame['mismatching_bytes']==0 and frame['guarded_outputs_checked']==1039 and frame['coverage_values_checked']==5900107776
        for name,meta in frame['ports'].items():assert sha(a.output/f'run/frame{i}'/(name+'.e4m3'))==meta['sha256']==sha(refs[i]/paths[name])
        # Encoder profile is cumulative for the whole API call. Adding the
        # separate Block0 profile again would double-count the prefix.
        pp=a.output/f'run/diagnostics-frame{i}/encoder/profile.json';profile=json.loads(pp.read_text());gpu=defaultdict(float);phase=defaultdict(float)
        for row in profile['records']:
            assert row['gpu_ms']>=0;gpu[row['label']]+=row['gpu_ms']
        for row in profile['phases']:phase[row['label']]+=row['wall_ms']
        total=sum(gpu.values());assert total>0
        results.append({'frame':i,'input_sha256':frame['input_sha256'],'poison':frame['poison'],'api_wall_ms_including_diagnostics':frame['wall_ms_diagnostic_only'],'instrumented_model_gpu_ms':total,'gpu_ms_by_label':dict(sorted(gpu.items(),key=lambda x:-x[1])),'phase_wall_ms_by_label':dict(sorted(phase.items(),key=lambda x:-x[1])),'profile_sha256':sha(pp)})
    for f,h in pins.items():assert sha(Path(f))==h,'artifact/input changed during profile'
    result={'schema':'neuralstudio.nr.encoder-profile-observation.v1','result':'correctness_pass_diagnostic_profile','plan_root':plan['receipt_root_sha256'],'native_pid':run['pid'],'frames':results,'shader_changes':False,'accepted_speedup':False,'limitations':['Ambient desktop load, not a quiet benchmark.','Two cases in one process; no timing confidence interval.','API wall includes diagnostic output writes and must not be treated as production latency.','Instrumented GPU sum omits initial/final texture blits.','Host phases refine scopes and overlap GPU execution; do not add these totals together.']}
    result['receipt_root_sha256']=canonical(result);(a.output/'profile-summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'root':result['receipt_root_sha256'],'frames':[{k:f[k] for k in ('frame','api_wall_ms_including_diagnostics','instrumented_model_gpu_ms')} for f in results]},indent=2))
if __name__=='__main__':main()
