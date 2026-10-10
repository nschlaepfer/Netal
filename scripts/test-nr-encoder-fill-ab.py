#!/usr/bin/env python3
"""ABBA fresh-process test of byte-identical host initialization; reserve GPU first.

Preserves all attempts, including failures. This does not promote a candidate.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys

def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(path):
    r=json.loads(path.read_text());b=dict(r);h=b.pop('receipt_root_sha256')
    assert canonical(b)==h,path
    return r

def main():
    p=argparse.ArgumentParser();p.add_argument('--baseline',required=True,type=Path);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--guard-script',required=True,type=Path);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    bm=load(a.baseline/'manifest.json');cm=load(a.candidate/'manifest.json')
    assert bm['timing_only_host_changes'] is True and cm['host_allocation_change']=='single-pass-payload-poison'
    assert bm['profiling_parent_root']==cm['profiling_parent_root'] and bm['detailed_host_phases'] and cm['detailed_host_phases']
    for name,h in bm['files'].items():
        if name.endswith('.metallib'):assert cm['files'][name]==h,name
    records=[];steps=[]
    def run(command,label):
        with (a.output/(label+'.log')).open('x') as log:r=subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT)
        steps.append({'label':label,'command':[str(x) for x in command],'exit_code':r.returncode})
        (a.output/'steps.json').write_text(json.dumps(steps,indent=2)+'\n')
        assert r.returncode==0,'failed '+label+'; preserve evidence'
    for label,kind,bundle in [('baseline-a','baseline',a.baseline),('candidate-a','candidate',a.candidate),('candidate-b','candidate',a.candidate),('baseline-b','baseline',a.baseline)]:
        run([sys.executable,a.guard_script.resolve(),'--me','Codex'],label+'-reservation-check')
        assert shutil.disk_usage(a.output).free>20*1024**3,'less than20GiB free; preserve prior results, do not clean'
        attempt=a.output/label
        run([sys.executable,repo/'scripts/run-nr-encoder-profile.py','--attempt',attempt,'--bundle',bundle],label+'-run')
        run([sys.executable,repo/'scripts/verify-nr-live-encoder.py',attempt,'--output',attempt/'verification.json'],label+'-verify')
        run([sys.executable,repo/'scripts/analyze-nr-encoder-profile.py',attempt,'--output',attempt/'profile-summary.json'],label+'-profile')
        profile=load(attempt/'profile-summary.json');verified=load(attempt/'verification.json')
        assert verified['result']=='pass' and profile['result']=='pass'
        records.append({'label':label,'kind':kind,'acceptance_root':verified['receipt_root_sha256'],'profile_root':profile['receipt_root_sha256'],'runs':profile['runs']})
        print(label+' verified',flush=True)
    comparisons={}
    for component in ('block0','encoder'):
        values={}
        for kind in ('baseline','candidate'):
            rows=[run['components'][component] for r in records if r['kind']==kind for run in r['runs']]
            assert len(rows)==8
            values[kind]={metric:median(row[metric] for row in rows) for metric in ('process_wall_ms','gpu_ms')}
            values[kind]['allocation_and_poison_ms']=median(sum(v for k,v in row['phase_wall_ms'].items() if k.endswith('/allocation_and_poison')) for row in rows)
        values['process_wall_reduction_fraction']=1-values['candidate']['process_wall_ms']/values['baseline']['process_wall_ms']
        comparisons[component]=values
    result={'schema':'neuralstudio.nr.encoder-fill-abba.v1','baseline_root':bm['receipt_root_sha256'],'candidate_root':cm['receipt_root_sha256'],
            'records':records,'comparisons':comparisons,'result':'pass','native_chains':16,'candidate_promoted':False,'reservation_guard':str(a.guard_script.resolve()),
            'scope':'Pinned image; ABBA batches, two four-process-pair matrices per variant. All checkpoints, poison and coverage retained. Desktop activity is observed, not controlled. Not game FPS.'}
    result['receipt_root_sha256']=canonical(result);(a.output/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'root':result['receipt_root_sha256'],'comparisons':comparisons},indent=2))

if __name__=='__main__':main()
