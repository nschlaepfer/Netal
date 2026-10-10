#!/usr/bin/env python3
"""Summarize verified timing-only encoder runs without confusing GPU and wall."""
import argparse
from collections import defaultdict
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
from statistics import median
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def load(p):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h,p
    return r
def summarize_phases(records):
    totals=defaultdict(float)
    for record in records:
        value=record['wall_ms']
        if not isinstance(value,(int,float)) or not math.isfinite(value) or value<0:
            raise ValueError('invalid phase duration')
        if not isinstance(record['label'],str) or not record['label']:
            raise ValueError('invalid phase label')
        totals[record['label']]+=value
    return dict(totals)
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    acceptance=load(a.attempt/'verification.json');assert acceptance['result']=='pass'
    candidate=acceptance['candidate_bundle'];bundle=Path(candidate['path']);manifest=load(bundle/'manifest.json')
    assert (manifest['timing_only_host_changes'] is True or manifest.get('host_allocation_change')=='single-pass-payload-poison') and manifest['profiling_parent_root']=='9f836feaf6cb3f05c85ff30b70eabdf66c8db58860d0d8347f0cdedbe924c9f0'
    assert manifest['receipt_root_sha256']==candidate['root'] and sha(bundle/'manifest.json')==candidate['manifest_sha256']
    repo=Path(__file__).resolve().parents[1];parent=repo/'build/nr_encoder_fixed/bundle-v1'
    for p in parent.rglob('*.metallib'):assert sha(p)==sha(bundle/p.relative_to(parent))
    commands=json.loads((a.attempt/'commands.json').read_text());runs=[];groups=defaultdict(list);phase_groups=defaultdict(list)
    for name in ('a5-a','a5-b','5a-a','5a-b'):
        item={'run':name,'components':{}}
        for component in ('block0','encoder'):
            path=a.attempt/name/component;profile=load(path/'profile.json');native=load(path/'run.json');assert profile['pid']==native['pid'] and native['result']=='pass' and profile['shader_changes'] is False
            by_label=defaultdict(lambda:{'gpu_ms':0.0,'submit_wait_ms':0.0,'host_scope_ms':0.0,'commands':0,'host_scopes':0})
            for record in profile['records']:
                g=by_label[record['label']]
                for field in ('gpu_ms','submit_wait_ms','host_scope_ms'):assert math.isfinite(record[field]) and record[field]>=0;g[field]+=record[field]
                g['commands']+=record['submit_wait_ms']>0;g['host_scopes']+=record['host_scope_ms']>0
            cmd=[c for c in commands if c['label']==component and any(str(path.resolve())==str(v) for v in c['command'])];assert len(cmd)==1
            wall=(datetime.fromisoformat(cmd[0]['end_utc'])-datetime.fromisoformat(cmd[0]['start_utc'])).total_seconds()*1000
            gpu=sum(v['gpu_ms'] for v in by_label.values());wait=sum(v['submit_wait_ms'] for v in by_label.values());host=sum(v['host_scope_ms'] for v in by_label.values())
            assert gpu>0 and wall>=gpu
            phases=summarize_phases(profile.get('phases',[]))
            item['components'][component]={'process_wall_ms':wall,'gpu_ms':gpu,'submit_wait_ms':wait,'host_scopes_ms':host,'phase_wall_ms':phases,
                'outside_host_scopes_ms':wall-host,'profile_sha256':sha(path/'profile.json'),'labels':dict(by_label)}
            for label,row in by_label.items():groups[component+'/'+label].append(row)
            for label,value in phases.items():phase_groups[component+'/'+label].append(value)
        runs.append(item)
    hotspots=[]
    for label,records in groups.items():
        assert len(records)==4
        hotspots.append({'label':label,'median_gpu_ms':median(x['gpu_ms'] for x in records),'median_host_scope_ms':median(x['host_scope_ms'] for x in records),
                         'gpu_ms_runs':[x['gpu_ms'] for x in records],'host_scope_ms_runs':[x['host_scope_ms'] for x in records]})
    hotspots.sort(key=lambda x:x['median_gpu_ms'],reverse=True)
    phase_summary=[]
    for label,values in phase_groups.items():
        assert len(values)==4
        phase_summary.append({'label':label,'median_wall_ms':median(values),'wall_ms_runs':values})
    phase_summary.sort(key=lambda x:x['median_wall_ms'],reverse=True)
    r={'schema':'neuralstudio.nr.verified-encoder-profile.v1','acceptance_root':acceptance['receipt_root_sha256'],'candidate_root':manifest['receipt_root_sha256'],'phase_summary':phase_summary,
       'runs':runs,'hotspots':hotspots,'shaders_identical_to_reference':True,'host_allocation_change':manifest.get('host_allocation_change'),'result':'pass',
       'scope':'Timing-only instrumented pinned-frame acceptance runs, not a game/production benchmark. GPU durations and CPU host scopes overlap and must not be added. Process wall includes launch, library setup, validation and IO.'}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'components':[{k:v for k,v in x.items() if k not in ('labels','phase_wall_ms')} for run in runs for x in run['components'].values()],'top_gpu':hotspots[:7],'top_host_phases':phase_summary[:7]},indent=2))
if __name__=='__main__':main()
