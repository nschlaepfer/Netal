#!/usr/bin/env python3
"""Independently check direct failure-state fields and actual recovery bytes."""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(v,label):
    if not v:raise ValueError(label)
def audit(peer):
    d=peer/'build/nr_full/temporal-fault-v3';summary=read(d/'summary.json');body=dict(summary);root=body.pop('receipt_root_sha256')
    require(hashlib.sha256(json.dumps(body,indent=1).encode()).hexdigest()==root=='92be912db59db4db3dc3a978b811c67d14ccbda4a65fdec5541a913b0cd1d2f1','summary seal')
    build=peer/'build/nr_full/v15t-fault';pins={}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);p=peer/p.strip();require(sha(p)==h,'frozen fault build');pins[str(p)]=h
    require((build/'build-defines.txt').read_text().strip()=='defines: -DNR_BODY_FAULT_TESTS','test-only hooks')
    runtime=build/'source/src/nr_body/nr_body_runtime_v11.mm';harness=build/'source/src/nr_body/nr_full_frame.mm'
    require(sha(runtime)=='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9' and sha(harness)=='285261f0542255350c0c44d6768276cb83cbb02d69e14495428c8a4182e7af8a','reviewed sources')
    text=runtime.read_text();require('require(ctx.damaged.empty()' in text and 'if (gpu_started) mark_damage' in text and 'mark_damage("code 3:' in text,'damaged-context fail-closed paths')
    hs=harness.read_text();require('bctx = NrBodyContext{}' in hs and 'tstate = NrTemporalState{}' in hs and 'ectx = NrEncoderContext{}' in hs,'fresh recovery contexts in source')
    faults=['constant-late-mutation','ex2-table-fault','history-copy-mismatch','ex2-table-row-corrupt','history-texel-corrupt'];require([c['fault'] for c in summary['cases']]==faults,'all fault cases')
    first='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2';second='d48bf48553e61c9e76d23b008dc1192862a11894e1465a6366bf1476473c9ac0';rows=[];pids=set()
    for claimed in summary['cases']:
        folder=d/claimed['fault'];r=read(folder/'receipt.json');require(sha(folder/'receipt.json')==claimed['receipt_sha256'],'native receipt binding')
        require(r['executable_sha256']==summary['harness']['executable_sha256']==sha(build/'nr_full_frame'),'executed test binary')
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v6' and r['result']=='fail' and len(r['frames'])==6,'full recorded failure/recovery protocol')
        require(r['pid']>0 and r['pid'] not in pids,'fresh fault process');pids.add(r['pid'])
        f=r['frames'];require([x['frame'] for x in f]==list(range(6)),'frame order')
        failed=f[1];require(failed['result']=='fail' and failed['body']['code']==3 and failed['body']['check_failures'] and not failed['temporal_state_advanced'] and failed['frames_since_reset_after']==failed['body']['frames_since_reset']==1,'direct failed-frame code and no advance')
        expected_arg=1 if claimed['fault'] in ('ex2-table-fault','ex2-table-row-corrupt') else 0
        require(failed['ex2_half_faults']=={'argument':expected_arg,'domain':0},'observed EX2 counts')
        expected_diff={'history-texel-corrupt':3,'history-copy-mismatch':1}.get(claimed['fault'],0)
        if claimed['fault']=='history-copy-mismatch':
            # The forced scalar is applied after the diagnostic field was assigned.
            require('(1 half values;' in failed['body']['error'],'forced history failure observed')
        elif claimed['fault']!='ex2-table-row-corrupt':require(failed['history_copy_value_differences']==expected_diff,'actual history differences')
        require(f[2]['result']=='fail' and f[2]['encoder']['code']==2,'damaged temporal continuation rejected')
        require(f[3]['result']=='fail' and f[3]['body']['code']==2 and 'context damaged' in f[3]['body']['error'] and f[3]['body']['checks_enforced']==0 and not f[3]['temporal_state_advanced'],'same-context reset rejected at body admission')
        for index,wanted in ((0,first),(4,first),(5,second)):
            frame=f[index];require(frame['result']=='pass' and frame['body']['code']==0 and frame['temporal_state_advanced'] and frame['history_copy_value_differences']==0,'clean recovered frame')
            require(frame['reference']['mismatching_values']==0 and frame['reference']['cpu_comparison_only'] and sha(frame['reference']['file'])==wanted,'actual reference identity')
            for name in (f'diag/frame-{index}/tex_out_155.bin',f'diag/frame-{index}/tex_out_156.bin'):
                p=folder/name;require(p.stat().st_size==16588800 and sha(p)==wanted,'actual recovery output/history payload')
            require(frame['surface_sha256_caller_readback']==frame['body']['history_sha256_after']==wanted,'reported output and history bind to bytes')
        require(f[4]['frames_since_reset_after']==1 and f[5]['frames_since_reset_after']==2 and f[5]['body']['history_sha256_before']==first,'recovered temporal progression')
        require(r['recovery']['result']=='pass' and len(r['recovery']['steps'])==3 and all(x['ok'] for x in r['recovery']['steps']),'reported recovery agrees on this dataset')
        rows.append({'fault':claimed['fault'],'pid':r['pid'],'receipt_sha256':claimed['receipt_sha256'],'failed_body_code':3,'failed_state_counter':1,'failed_advance':False,'EX2_argument_faults':expected_arg,'fresh_reset_and_continuation_bytes_match':True})
    return {'schema':'neuralstudio.nr.recovery-independent-audit.v1','result':'pass_observed_fault_and_recovery_bytes','summary_root':root,'frozen_artifacts':pins,'runs':rows,'failed_frame_fields_directly_observed':True,'full_goal_complete':False,'limits':['Five fault patterns and original-image/zero-motion recovery only.','Same-context reset is refused by the body; the harness may compute the reset encoder first.','Recovery reporter uses no-exception as success even though do_frame can record a comparison failure without throwing; current recovered bytes pass, but that reporter needs a negative comparison test.','contexts_prepared_once remains true despite explicit recovery context recreation; that metadata must not be used as proof of one context in these recovery runs.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior audit')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','frozen_artifacts')},indent=2))
if __name__=='__main__':main()
