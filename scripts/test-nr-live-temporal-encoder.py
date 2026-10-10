#!/usr/bin/env python3
"""Run under the shared GPU lock; preserve every result, including mismatches."""
import argparse
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
    p=argparse.ArgumentParser();p.add_argument('--probe-build',type=Path,required=True);p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--sampler',type=int,choices=range(3),default=0);p.add_argument('--poison',choices=('a5','5a'),default='a5');p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.probe_build=a.probe_build.resolve();peer=a.peer_root.resolve()
    build=sealed(a.probe_build/'build.json');assert build['exit_code']==0;encoder=Path(build['encoder_build']);enc=sealed(encoder/'build.json');assert enc['receipt_root_sha256']==build['encoder_root']
    files=[a.probe_build/'build.json',encoder/'build.json',Path(__file__)]
    for n,h in build['files'].items():assert sha(a.probe_build/n)==h;files.append(a.probe_build/n)
    for n,h in enc['assets'].items():assert sha(encoder/'assets'/n)==h;files.append(encoder/'assets'/n)
    for n,h in enc['source_files'].items():assert sha(encoder/'source'/n)==h;files.append(encoder/'source'/n)
    descriptor=peer/'build/nr_body/desc-v17/production-a5-a/chain.json';reference=repo/'build/nr_reset_history/matrix-audit-v1.json';image=repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le'
    refs=sealed(reference);assert refs['result']=='pass_bounded_temporal_reference'
    def walk(v):
        if isinstance(v,dict):
            for x in v.values():walk(x)
        elif isinstance(v,list):
            for x in v:walk(x)
        elif isinstance(v,str) and v.startswith('/'):
            path=Path(v).resolve();assert path.is_relative_to(peer/'build') or path.is_relative_to(repo/'build') or path==image.resolve(),'descriptor path scope'
            if path.is_file():files.append(path)
            elif path.is_dir():files.extend(p for p in path.glob('*.bin') if p.is_file())
    walk(json.loads(descriptor.read_text()));files += [descriptor,reference,image]
    assert not a.output.exists() and shutil.disk_usage(repo).free>16*1024**3;a.output.mkdir(parents=True)
    pins={str(p):sha(p) for p in files};command=[str(a.probe_build/'probe'),str(encoder/'assets'),str(descriptor),str(image),str(reference),str(a.sampler),a.poison,str(a.output/'run')]
    plan={'schema':'neuralstudio.nr.live-temporal-encoder-plan.v1','build_root':build['receipt_root_sha256'],'encoder_root':enc['receipt_root_sha256'],'rtx_reference_root':refs['receipt_root_sha256'],'artifacts':pins,'command':command,'started_utc':datetime.now(timezone.utc).isoformat(),'history_source':'live first full native inference -> GPU copy; no archived activation ingress'}
    plan['receipt_root_sha256']=canonical(plan);(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    with (a.output/'native.log').open('x') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    execution={'exit_code':run.returncode,'ended_utc':datetime.now(timezone.utc).isoformat()};(a.output/'execution.json').write_text(json.dumps(execution,indent=2)+'\n')
    drift=[p for p,h in pins.items() if sha(Path(p))!=h];record=json.loads((a.output/'run/run.json').read_text()) if (a.output/'run/run.json').exists() else None
    if record:
        for n,h in record['first_body_loaded'].items():
            path=n[4:] if n.startswith('lib:') else n[5:].rsplit(':',1)[0]
            if path not in pins or pins[path]!=h:drift.append('unbound loaded resource: '+path)
        for name,m in record['ports'].items():assert sha(a.output/'run'/(name+'.e4m3'))==m['sha256']
    result={'schema':'neuralstudio.nr.live-temporal-encoder-observation.v1','plan_root':plan['receipt_root_sha256'],'exit_code':run.returncode,'result':record['result'] if record else 'execution_failed','artifact_drift':drift,'native_receipt_sha256':sha(a.output/'run/run.json') if record else None,'fresh_processes':1,'full_temporal_model_accepted':False,'scope':'One original-image live-history temporal encoder experiment; all failures retained. No performance or full-model temporal promotion.'}
    if drift:result['result']='provenance_failure'
    result['receipt_root_sha256']=canonical(result);(a.output/'observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if run.returncode or drift:raise SystemExit(3)
if __name__=='__main__':main()
