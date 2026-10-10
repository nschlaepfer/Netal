#!/usr/bin/env python3
"""Locked ABBA diagnostic comparison; unchanged shader assets, no speedup promotion."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def seal(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert seal(b)==h;return r
def save(p,r):
    assert not p.exists();r['receipt_root_sha256']=seal(r);p.write_text(json.dumps(r,indent=2)+'\n');return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.candidate=a.candidate.resolve();a.matrix=a.matrix.resolve()
    baseline=repo/'build/nr_encoder_runtime/build-v1';probe=repo/'build/nr_encoder_runtime/profile-v1';bb=read(baseline/'build.json');cb=read(a.candidate/'build.json');audit=read(a.matrix/'independent-audit-v1.json')
    assert audit['result']=='pass' and audit['build_root']==cb['receipt_root_sha256'];assert bb['assets']==cb['assets'] and cb['shader_changes'] is False
    assert sha(probe/'source/probe.mm')==sha(a.matrix/'probe-source.mm'),'same profiling harness source'
    assert not a.output.exists();assert shutil.disk_usage(repo).free>22*1024**3;a.output.mkdir(parents=True)
    plan=save(a.output/'plan.json',{'schema':'neuralstudio.nr.block0-scratch-comparison-plan.v1','baseline_root':bb['receipt_root_sha256'],'candidate_root':cb['receipt_root_sha256'],'candidate_audit_root':audit['receipt_root_sha256'],'order':['baseline','candidate','candidate','baseline'],'comparison_driver_sha256':sha(Path(__file__)),'profile_driver_sha256':sha(repo/'scripts/profile-nr-encoder-runtime.py'),'scope':'Ambient-load ABBA diagnostic decomposition including output-only diagnostic writes. Not a quiet benchmark or promotion.'})
    runs=[]
    for i,role in enumerate(plan['order']):
        dest=a.output/f'{i}-{role}';is_candidate=role=='candidate'
        stats=subprocess.run(['ioreg','-r','-c','AGXAccelerator','-l'],capture_output=True,text=True,check=True)
        (a.output/f'{i}-ioreg.txt').write_text(stats.stdout)
        command=[sys.executable,str(repo/'scripts/profile-nr-encoder-runtime.py'),'--probe-build',str(a.candidate if is_candidate else probe),'--runtime-build',str(a.candidate if is_candidate else baseline),'--probe-source',str(a.matrix/'probe-source.mm' if is_candidate else probe/'source/probe.mm'),'--output',str(dest)]
        with (a.output/f'{i}.log').open('x') as log:result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        save(a.output/f'{i}-execution.json',{'command':command,'exit_code':result.returncode});assert result.returncode==0,'preserved failed run'
        summary=read(dest/'profile-summary.json');assert summary['result']=='correctness_pass_diagnostic_profile'
        if is_candidate:
            for frame in range(2):
                reuse=json.loads((dest/f'run/diagnostics-frame{frame}/block0/scratch-reuse.json').read_text())
                assert reuse=={'allocations':44,'reuses':1452,'re_poison_every_use':True}
        runs.append({'role':role,'path':str(dest),'root':summary['receipt_root_sha256'],'frames':summary['frames']});print('completed '+str(i)+' '+role,flush=True)
    # Compare every retained binary diagnostic, not just the six API ports.
    reference=a.output/'0-baseline/run';expected={str(f.relative_to(reference)):sha(f) for f in reference.rglob('*') if f.is_file() and f.suffix in ('.bin','.f16','.e4m3')};assert expected
    total=0
    for row in runs[1:]:
        directory=Path(row['path'])/'run';actual={str(f.relative_to(directory)):sha(f) for f in directory.rglob('*') if f.is_file() and f.suffix in ('.bin','.f16','.e4m3')};assert actual==expected,'diagnostic checkpoint mismatch'
        total+=sum((directory/n).stat().st_size for n in expected)
    metrics={}
    for role in ('baseline','candidate'):
        frames=[f for r in runs if r['role']==role for f in r['frames']]
        metrics[role]={k:statistics.median([f[k] for f in frames]) for k in ('api_wall_ms_including_diagnostics','instrumented_model_gpu_ms')}
        metrics[role]['block0_allocation_poison_ms']=statistics.median([f['phase_wall_ms_by_label']['block0.chunk/allocation_and_poison'] for f in frames])
    result=save(a.output/'comparison.json',{'schema':'neuralstudio.nr.block0-scratch-comparison.v1','result':'correctness_pass_diagnostic_comparison','plan_root':plan['receipt_root_sha256'],'runs':runs,'diagnostic_files_per_run':len(expected),'additional_binary_bytes_compared':total,'diagnostic_hashes':expected,'median_observations_ms':metrics,'accepted_speedup':False,'shader_changes':False,'limitations':['Ambient desktop activity; interleaving is not isolation.','Four processes, two cases each; diagnostics include file output.','No direct full-model integration claim for this encoder candidate.','Whole-group/frame performance promotion requires further evidence.']})
    print(json.dumps({'root':result['receipt_root_sha256'],'median_observations_ms':metrics},indent=2))
if __name__=='__main__':main()
