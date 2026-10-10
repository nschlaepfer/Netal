#!/usr/bin/env python3
"""Review pinned v10 commit ordering and bind the five observed fault controls.

Failed-frame counters are not serialized by this harness; do not invent them.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,label):
    if not ok:raise ValueError(label)
def audit(peer):
    directory=peer/'build/nr_full/temporal-fault-v2';summary=read(directory/'summary.json');body=dict(summary);root=body.pop('receipt_root_sha256')
    require(hashlib.sha256(json.dumps(body,indent=1).encode()).hexdigest()==root=='3660eb268f13408d80167a17355db6a05c777bb7c2a371fa8cb512d5a7bbbf41','fault summary seal')
    artifacts={}
    for build in ('v13t','v13t-fault'):
        d=peer/'build/nr_full'/build
        for line in (d/'build-sha256.txt').read_text().splitlines():
            h,p=line.split(None,1);path=peer/p.strip();require(sha(path)==h,'frozen build artifact');artifacts[str(path)]=h
    source=peer/'build/nr_full/v13t/source/src/nr_body/nr_body_runtime_v10.mm'
    require(sha(source)=='8f02851c1ec1ffdcf7f1ff499e33bd2ebedd436f343ddcd0d5aff9caa7ddb038','reviewed v10 source')
    s=source.read_text();commit='in.temporal->history_sha256 = pend.hist; in.temporal->last_output_sha256 = pend.out; in.temporal->frames_since_reset ='
    require(s.count(commit)==1 and len(re.findall(r'in\.temporal->history_sha256\s*=(?!=)',s))==1,'one state commit site')
    require(s.index('res.constants_unchanged_after_frame = true;')<s.index('model constant changed since load:')<s.index('res.temporal_state_advanced = fails.empty() && pend.ok;')<s.index(commit),'commit after final rehash')
    require('if (res.temporal_state_advanced)' in s and 'pend.ok = (hdiff == 0 && osha == hsha)' in s,'commit guarded by independent history equality and all failures')
    require((peer/'build/nr_full/v13t/build-defines.txt').read_text().strip()=='defines: none','production no hooks')
    require((peer/'build/nr_full/v13t-fault/build-defines.txt').read_text().strip()=='defines: -DNR_BODY_FAULT_TESTS','fault build explicit')
    faults=['constant-late-mutation','ex2-table-fault','history-copy-mismatch','ex2-table-row-corrupt','history-texel-corrupt']
    require([c['fault'] for c in summary['cases']]==faults,'complete fault set')
    rows=[];pids=set()
    for c in summary['cases']:
        d=directory/c['fault'];r=read(d/'receipt.json');require(sha(d/'receipt.json')==c['receipt_sha256'],'fault native receipt')
        require(r['executable_sha256']==summary['executable_sha256']==sha(peer/'build/nr_full/v13t-fault/nr_full_frame'),'executed test build')
        require(r['pid']>0 and r['pid'] not in pids,'distinct fault processes');pids.add(r['pid'])
        require(r['result']=='fail' and len(r['frames'])==3,'failed three-call observation')
        a,b,z=r['frames'];require(a['frame']==0 and a['result']=='pass' and a['temporal_state_advanced'] and a['body']['frames_since_reset']==1,'clean initial state')
        require(b['frame']==1 and b['result']=='fail' and b['error']==c['frame1_error'] and 'computed check(s) failed' in b['error'],'runtime check rejected fault')
        require(z['frame']==2 and z['result']=='fail' and z['error']==c['frame2_error']=='encoder frame rejected: stale/substituted temporal history','follow-on rejection')
        if c['fault']=='constant-late-mutation':require('1 computed check(s) failed: model constant changed since load:' in b['error'],'late-only rehash failure')
        if c['fault']=='ex2-table-row-corrupt':require('argument mismatches' in b['error'] and 'model constant changed since load:' in b['error'],'real table fault exercised')
        if c['fault']=='history-texel-corrupt':require('(3 half values;' in b['error'],'actual history corruption exercised')
        rows.append({'fault':c['fault'],'pid':r['pid'],'receipt_sha256':c['receipt_sha256'],'frame1_rejection':b['error'],'next_call_rejected':True,'failed_frame_state_fields_present':'temporal_state_advanced' in b})
    return {'schema':'neuralstudio.nr.deferred-commit-independent-audit.v1','result':'pass_source_order_and_observed_rejections','fault_summary_root':root,'frozen_artifacts':artifacts,'runs':rows,'single_commit_after_all_checks_verified_in_source':True,'five_runtime_fault_rejections_observed':True,'failed_frame_counters_directly_observed':False,'limits':['Fault harness discards partial frame detail when it throws: failed frames retain error text but not the numeric code, actual state counter or EX2 counters.','No-advance behavior is supported by pinned source ordering and subsequent stale-history rejection, not by invented failed-frame counter fields.','This does not establish recovery after a fault; retry/reset recovery needs its own test.'],'full_goal_complete':False,'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior audit')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('frozen_artifacts','runs')},indent=2))
if __name__=='__main__':main()
