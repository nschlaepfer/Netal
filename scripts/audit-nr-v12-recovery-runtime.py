#!/usr/bin/env python3
"""Separate observed recovery behaviour from the preserved v25t label defect."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();peer=a.peer.resolve()
    suite=peer/'build/nr_body/v12-evidence-v1';build=peer/'build/nr_full/v25t-fault';pins={}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,n=line.split(None,1);path=peer/n.strip();assert sha(path)==h;pins[str(path)]=h
    assert (build/'build-defines.txt').read_text().strip()=='defines: -DNR_BODY_FAULT_TESTS'
    assert sha(build/'source/encoder/nr_encoder_runtime.o')=='830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3'
    faults=['ex2-table-fault','history-copy-mismatch','ex2-table-row-corrupt','history-texel-corrupt','constant-late-mutation','validator-skipped','coverage-compensating','restage-corrupt']
    expected=['a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','d48bf48553e61c9e76d23b008dc1192862a11894e1465a6366bf1476473c9ac0']
    rows=[];pids=set();bytes_checked=0
    for fault in faults:
        d=suite/'harness'/('fault-'+fault);path=d/'receipt.json';r=read(path);pins[str(path)]=sha(path)
        assert r['executable_sha256']==sha(build/'nr_full_frame') and r['pid'] not in pids;pids.add(r['pid'])
        assert r['result']=='fail' and len(r['frames'])==7;f=r['frames']
        assert f[1]['body']['code']==3 and f[1]['body']['check_failures'] and not f[1]['temporal_state_advanced'] and f[1]['frames_since_reset_after']==1
        assert f[2]['encoder']['code']==2 and not f[2].get('temporal_state_advanced',False)
        assert f[3]['body']['code']==2 and 'context damaged' in f[3]['body']['error'] and f[3]['body']['checks_enforced']==0 and not f[3]['temporal_state_advanced']
        for index,wanted in [(0,expected[0]),(4,expected[0]),(5,expected[1]),(6,expected[0])]:
            frame=f[index];assert frame['body']['code']==0 and frame['surface_sha256_caller_readback']==frame['body']['history_sha256_after']==wanted
            for n in ('tex_out_155.bin','tex_out_156.bin'):
                actual=d/f'diag/frame-{index}'/n;assert actual.stat().st_size==16588800 and sha(actual)==wanted;pins[str(actual)]=sha(actual);bytes_checked+=actual.stat().st_size
        assert f[4]['result']==f[5]['result']=='pass' and f[4]['frames_since_reset_after']==1 and f[5]['frames_since_reset_after']==2
        assert f[6]['result']=='fail' and f[6]['reference']['mismatching_values']>0 and f[6]['reference']['cpu_comparison_only'] and sha(Path(f[6]['reference']['file']))==expected[1]
        assert r['context_preparations']==3 and not r['contexts_prepared_once']
        steps=r['recovery']['steps'];assert len(steps)==4 and not steps[0]['ok'] and steps[0]['observed']=='execution_failed' and steps[0]['expected']=='rejected'
        assert [s['observed'] for s in steps[1:]]==['pass','pass','comparison-fail'] and all(s['ok'] for s in steps[1:])
        rows.append({'fault':fault,'pid':r['pid'],'raw_runtime_recovery_checks_pass':True,'label_defect':'step A body admission rejection labelled execution_failed','receipt_sha256':pins[str(path)]})
    for path,h in pins.items():assert sha(Path(path))==h
    result={'schema':'neuralstudio.nr.v12-recovery-runtime-review.v1','result':'observed_runtime_recovery_pass_with_preserved_label_failures','runs':rows,'actual_output_history_bytes_checked':bytes_checked,'pins':pins,
            'original_suite_failure_reclassified':False,'resident_acceptance':False,'limits':['Eight fault/recovery patterns, original image and zero motion.','This supplement does not overwrite the failed v25t suite or prove unresolved strong decoder checkpoints.','A corrected reporter needs its own new receipts.'],'verifier_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ('pins','runs')},indent=2))


if __name__=='__main__':main()
