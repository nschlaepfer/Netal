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
    d=peer/'build/nr_full/temporal-fault-v5';summary=read(d/'summary.json');body=dict(summary);root=body.pop('receipt_root_sha256')
    require(hashlib.sha256(json.dumps(body,indent=1).encode()).hexdigest()==root=='568861ad109623101c7543c9066eab22d80f10726bf63850f62135517cfed806','summary seal')
    build=peer/'build/nr_full/v21t-fault';pins={}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);p=peer/p.strip();require(sha(p)==h,'frozen fault build');pins[str(p)]=h
    require((build/'build-defines.txt').read_text().strip()=='defines: -DNR_BODY_FAULT_TESTS','test-only hooks')
    runtime=build/'source/src/nr_body/nr_body_runtime_v11.mm';harness=build/'source/src/nr_body/nr_full_frame.mm'
    require(sha(runtime)=='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9' and sha(harness)=='ade17848864e8fbf10c3240eb79423b39c784f45b8f262621816b2c3e4840338','reviewed sources')
    text=runtime.read_text();require('require(ctx.damaged.empty()' in text and 'if (gpu_started) mark_damage' in text and 'mark_damage("code 3:' in text,'damaged-context fail-closed paths')
    hs=harness.read_text();require('bctx = NrBodyContext{}' in hs and 'tstate = NrTemporalState{}' in hs and 'ectx = NrEncoderContext{}' in hs,'fresh recovery contexts in source')
    repo=Path(__file__).resolve().parents[1];enc=repo/'build/nr_encoder_temporal/half-fma-v1'
    require(sha(build/'source/encoder/nr_encoder_runtime.o')==summary['harness']['encoder_object_sha256']=='830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3','candidate encoder object')
    eb=read(enc/'build.json');body=dict(eb);er=body.pop('receipt_root_sha256')
    require(canonical(body)==er=='0a57b3bac8616c7103ae30a2df5a91296f3bda2ab9f84960725bb943df053b2e','candidate build seal')
    for name,h in eb['assets'].items():require(sha(enc/'assets'/name)==h,'candidate constant asset')
    prod=peer/'build/nr_full/v21t'
    require((prod/'build-defines.txt').read_text().strip()=='defines: none','production has no test hooks')
    for line in (prod/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);p=peer/p.strip();require(sha(p)==h,'production source/build pin');pins[str(p)]=h
    pr=read(d/'production-build-rejects-test-fault/receipt.json')
    require(pr['executable_sha256']==summary['harness']['production_executable_sha256']==sha(prod/'nr_full_frame'),'production executable binding')
    require(pr['result']=='fail' and len(pr['frames'])==2,'production rejection sequence')
    pf=pr['frames'][1]
    require(pf['test_fault']=='ex2-table-fault' and pf['body']['code']==2 and pf['body']['error']=='live input rejected: test_fault set on a production build','body admission rejects fault hook')
    require(not pf['temporal_state_advanced'] and pf['frames_since_reset_after']==1,'production rejected frame does not advance')
    neg=peer/'build/nr_temporal/negctl-v11-halffma/summary.txt';text=neg.read_text();pins[str(neg)]=sha(neg)
    require(sum(x.startswith('PASS ') for x in text.splitlines())==20 and 'negative controls passed 20/20' in text and 'FAIL ' not in text,'fresh body admission negatives')
    require(any(x.startswith('PASS temporal-ex2-table-wrong-sha:') and 'ex2 half-domain table: size/sha mismatch' in x for x in text.splitlines()),'actual table rejection reached')
    faults=['constant-late-mutation','ex2-table-fault','history-copy-mismatch','ex2-table-row-corrupt','history-texel-corrupt'];require([c['fault'] for c in summary['cases']]==faults,'all fault cases')
    first='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2';second='d48bf48553e61c9e76d23b008dc1192862a11894e1465a6366bf1476473c9ac0';rows=[];pids=set()
    for claimed in summary['cases']:
        folder=d/claimed['fault'];r=read(folder/'receipt.json');require(sha(folder/'receipt.json')==claimed['receipt_sha256'],'native receipt binding')
        require(r['executable_sha256']==summary['harness']['fault_executable_sha256']==sha(build/'nr_full_frame'),'executed test binary')
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v8' and r['result']=='fail' and len(r['frames'])==7,'full recorded failure/recovery protocol')
        require(r['pid']>0 and r['pid'] not in pids,'fresh fault process');pids.add(r['pid'])
        require(Path(r['encoder_asset_root']).resolve()==enc.resolve()/'assets','executed candidate assets')
        require(r['body_descriptor_sha256']==summary['harness']['descriptor_sha256']==sha(peer/'build/nr_body/desc-v23t/chain.json'),'same descriptor')
        f=r['frames'];require([x['frame'] for x in f]==list(range(7)),'frame order')
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
        require(r['recovery']['result']=='pass' and len(r['recovery']['steps'])==4 and all(x['ok'] for x in r['recovery']['steps']),'reported recovery agrees on this dataset')
        require(r['context_preparations']==3 and not r['contexts_prepared_once'],'actual recovery context count')
        wrong=f[6];require(wrong['result']=='fail' and wrong['body']['code']==0 and wrong['reference']['cpu_comparison_only'] and wrong['reference']['mismatching_values']>0,'wrong-reference comparison fails without runtime rejection')
        require(wrong['surface_sha256_caller_readback']==first and sha(wrong['reference']['file'])==second,'wrong reference never became runtime output')
        for n in ('tex_out_155.bin','tex_out_156.bin'):require(sha(folder/'diag/frame-6'/n)==first,'wrong-reference control retains correct actual bytes')
        require([x['observed'] for x in r['recovery']['steps']]==['rejected','pass','pass','comparison-fail'],'distinct recovery outcomes')
        rows.append({'fault':claimed['fault'],'pid':r['pid'],'receipt_sha256':claimed['receipt_sha256'],'failed_body_code':3,'failed_state_counter':1,'failed_advance':False,'EX2_argument_faults':expected_arg,'fresh_reset_and_continuation_bytes_match':True})
    return {'schema':'neuralstudio.nr.half-fma-recovery-independent-audit.v1','result':'pass_observed_fault_and_recovery_bytes','summary_root':root,'frozen_artifacts':pins,'runs':rows,'failed_frame_fields_directly_observed':True,'body_negative_admission_controls':20,'production_fault_hook_rejection':True,'encoder_candidate_root':er,'full_goal_complete':False,'limits':['Five fault patterns and original-image/zero-motion recovery only.','Same-context reset is refused by the body; the harness may compute the reset encoder first.','The wrong-reference control validates comparator reporting, not an additional model inference domain.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior audit')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','frozen_artifacts')},indent=2))
if __name__=='__main__':main()
