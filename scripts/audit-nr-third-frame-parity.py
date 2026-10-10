#!/usr/bin/env python3
"""Bind the prior live Mac state audit to the subsequently captured RTX frame2.

Comparison only: never reruns inference or supplies these bytes to a runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path

def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,label):
    if not ok:raise ValueError(label)
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'receipt seal');return r
def audit(peer):
    repo=Path(__file__).resolve().parents[1]
    native=sealed(repo/'build/nr_connected_temporal_audit/v4-state-audit-v2.json')
    rtx=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json')
    require(native['receipt_root_sha256']=='87465f406240ed0c8eb0a43f8c97095c219afc6c48739d1dbf1b8887da8b3589','reviewed native state parent')
    require(rtx['receipt_root_sha256']=='e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657' and rtx['fresh_processes']==4 and rtx['evaluations']==12,'repeated RTX parent')
    for p,h in native['artifact_hashes'].items():require(sha(p)==h,'native evidence drift '+p)
    reference=repo/'build/nr_three_frame_history/receipts-v1/temporal-reset100-a5-a/frame-2-output.rgba16le'
    require(reference.stat().st_size==16588800 and sha(reference)==rtx['outputs'][2],'complete retrieved RTX reference')
    wanted=reference.read_bytes();ports={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    rows=[]
    for pair in ('a5-a5','5a-5a','a5-5a','5a-a5'):
        d=peer/'build/nr_full/temporal-seq-v4'/pair;r=read(d/'receipt.json');f=r['frames'][2]
        require(r['pid'] in {v['pid'] for v in native['runs']},'audited native process')
        require(f['no_reference_no_parity_meaning'] and f['reference']['file']=='','original runtime did not consume this later reference')
        for name in ('final-2.rgba16le','diag/frame-2/tex_out_155.bin','diag/frame-2/tex_out_156.bin'):
            require((d/name).read_bytes()==wanted,'all actual third output/history bytes versus RTX: '+name)
        require(f['body']['history_sha256_before']==rtx['outputs'][1] and f['body']['history_sha256_after']==rtx['outputs'][2],'RTX history transition')
        for name,target in ports.items():require(f['regions'][name]['sha256']==rtx['boundary_ports'][5][target],'third encoder RTX boundary '+name)
        rows.append({'pair':pair,'pid':r['pid'],'poison':f['poison'],'native_receipt_sha256':sha(d/'receipt.json'),'third_output_and_history_sha256':rtx['outputs'][2],'third_encoder_ports_match_RTX':True})
    return {'schema':'neuralstudio.nr.third-frame-parity-supplement.v1','result':'pass_bounded_three_frame_output_history_parity','native_state_audit_root':native['receipt_root_sha256'],'RTX_matrix_root':rtx['receipt_root_sha256'],'reference_sha256':sha(reference),'runs':rows,'third_output_post_history_bytes_compared':4*3*16588800,'third_encoder_boundary_hashes_compared':24,'runtime_expected_byte_ingress':False,'full_goal_complete':False,'production_promotion':False,'scope':'Original1920x1080 image repeated three times, reset[1,0,0], zero motion; four Mac and four RTX processes, dual poison, actual outputs and histories. No changing-image, nonzero-motion, longer-sequence or performance claim.','remaining_evidence_limits':native['limits'][1:],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior supplement')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
