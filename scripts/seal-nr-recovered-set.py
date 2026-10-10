#!/usr/bin/env python3
"""Seal recovered v17 cases, retaining the stopped-matrix/auditor counterexample."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]
CASES={
 'black3':('black3-v17-audit-v3.json','7108932820d2693436603ee00794b60684d345477a383ed98f33bc14b1efe0d8'),
 'checker3':('varied-v1/checker3-audit.json','7b5d362b70bae67199c49ca0917e37b05a8815e4a0f2074aa1e2a415b7f8ce0b'),
 'images3':('varied-v1/images3-audit.json','0220e9fb7250093b76ec5103ef6f63072cf098f03cd3fd6a896d658216ace44f'),
 'midreset8':('varied-v1/midreset8-audit-v2.json','26ac50f34b78b115e2fca6e857d44b8b5aca65332b1a87b0d5864334a14daa9a'),
 'uniform3':('varied-v2/uniform3-audit.json','b0baf22c74f3788ad94ae7e75033aeb76c3468b0b2fd62c5a64fff5e6c9cfce2'),
 'spatial3':('varied-v2/spatial3-audit.json','cadab1abbf086f1a6d7c6e55aa275805db34ad7a10216c3ed8faaf44d291016f'),
 'combined3':('varied-v2/combined3-audit.json','77e6cebef3eeba9c41f520633a3734f8ecffcb8cadc37bba1cd6436134620472'),
 'dense':('varied-v2/dense-audit.json','b32a65cd5efe87e4ceb8c15b73bc90ccea73e33ed7079a87431da3be3ec3f807')}


def sha(path):
    with Path(path).open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def canonical(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def need(ok,message):
    if not ok:raise ValueError(message)


def seal():
    base=ROOT/'build/nr_recovery_runs'; pins={}; rows=[]; pids=set(); frames=ports=raw=0
    verifier_sources={}
    for commit in ('dd13115','c39985e'):
        data=subprocess.check_output(['git','show',commit+':scripts/audit-nr-recovery-run.py'],cwd=ROOT)
        verifier_sources[commit]=hashlib.sha256(data).hexdigest()
    for case,(relative,expected) in CASES.items():
        path=base/relative;r=json.loads(path.read_text());h=r.pop('receipt_root_sha256')
        need(canonical(r)==h==expected,'case seal '+case)
        need(r['body_version']==17 and r['configuration']==case and r['result']=='pass_bounded_live_output_ports_history','scoped case pass')
        need(not r['full_goal_complete'] and not r['performance_claim'] and not r['strict_intermediate_checkpoint_acceptance'],'no implicit promotion')
        need(r['verifier_sha256'] in verifier_sources.values(),'retained verifier source identity')
        if case=='midreset8':need(r['verifier_sha256']==verifier_sources['c39985e'],'corrected reset-local verifier required')
        count=8 if case=='midreset8' else 1 if case=='dense' else 3
        ids={x['pid'] for x in r['rows']}
        need(len(ids)==r['fresh_processes']==4 and not ids&pids,'distinct fresh processes across cases');pids|=ids
        need(len(r['rows'])==r['frames']==4*count,'complete case frame count')
        for pid in ids:need(sorted(x['frame'] for x in r['rows'] if x['pid']==pid)==list(range(count)),'every frame in each process')
        for i in range(count):need(Counter(x['poison'] for x in r['rows'] if x['frame']==i)==Counter({165:2,90:2}),'two repeats of each poison per frame')
        need(r['actual_output_history_bytes_checked']==4*count*3*16588800 and r['encoder_boundary_hashes_checked']==4*count*6,'byte/port extent')
        for name,value in r['artifacts'].items():
            need(name not in pins or pins[name]==value,'conflicting artifact identities');pins[name]=value
        pins[str(path)]=sha(path);frames+=r['frames'];ports+=r['encoder_boundary_hashes_checked'];raw+=r['actual_output_history_bytes_checked']
        rows.append({'case':case,'audit_root':h,'audit_file_sha256':sha(path),'frames':r['frames'],'verifier_sha256':r['verifier_sha256']})
    failure_path=base/'varied-v1/midreset8-audit.json';failure=json.loads(failure_path.read_text());h=failure.pop('receipt_root_sha256')
    need(canonical(failure)==h=='13fdbee152f1d53c631323cf3ae0b1eddd22d6648b982952d313bbc1cb4a7c8d' and failure['result']=='fail','original audit failure retained')
    stopped=base/'varied-v1/matrix-summary.json';continued=base/'varied-v2/matrix-summary.json'
    need(json.loads(stopped.read_text())['all_cases_passed'] is False,'stopped run not relabelled')
    need(json.loads(continued.read_text())['all_cases_passed'] is True,'remaining run completed')
    for path in (failure_path,stopped,continued):pins[str(path)]=sha(path)
    for path,expected in pins.items():need(sha(path)==expected,'current retained byte identity: '+path)
    need((len(pids),frames,ports,raw)==(32,108,648,5374771200),'complete eight-case extent')
    return {'schema':'neuralstudio.nr.recovered-set.v1','result':'pass_bounded_live_outputs_ports_history',
        'cases':rows,'fresh_processes':32,'frames':108,'encoder_boundary_hashes_checked':648,
        'actual_output_history_bytes_checked':raw,'retained_artifacts_rehashed':len(pins),'verifier_source_commits':verifier_sources,
        'retained_auditor_failure_root':h,'failure_resolution':'reset-local encoder index verified against unchanged harness line83; same outputs reaudited, no frames rerun',
        'full_strong_checkpoint_suite_accepted':False,'performance_baseline_accepted':False,'runtime_promoted':False,'full_goal_complete':False,
        'limits':['Eight fixed input/control sequences at1920x1080; not universal input-domain or long-run/game validation.',
                  'Production omissions remain explicit; this does not replace strong intermediate checkpoint evidence.',
                  'Black3 v11 control is separate and is not counted in these v17 totals. No timing or speedup claim.'],
        'verifier_sha256':sha(Path(__file__))}


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve existing seal')
    r=seal();r['receipt_root_sha256']=canonical(r)
    with a.out.open('x') as stream:json.dump(r,stream,indent=2);stream.write('\n')
    print(json.dumps(r,indent=2))


if __name__=='__main__':main()
