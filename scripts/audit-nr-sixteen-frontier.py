#!/usr/bin/env python3
"""Retain the first long-history mismatch; runtime success is not parity."""
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
    repo=Path(__file__).resolve().parents[1];rtx=read(repo/'build/nr_sixteen/matrix-audit-v1.json');b=dict(rtx);h=b.pop('receipt_root_sha256');require(canonical(b)==h=='4feea5197385aecc8a9784f3ac955cd9362ec9f1f3b7ebfe2c6d7aaf55998576','RTX matrix seal')
    names=['replica-1-pilot','replica-2-a5','replica-3-5a','replica-4-5a-a5'];mapping={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'};rows=[];pids=set();signature=None
    for name in names:
        d=peer/'build/nr_full/temporal-sixteen-v1'/name;r=read(d/'receipt.json');require(r['pid']>0 and r['pid'] not in pids,'fresh native process');pids.add(r['pid'])
        require(r['result']=='pass' and len(r['frames'])==16,'runtime-check-only pass')
        observed=[];frames=[]
        for i,f in enumerate(r['frames']):
            actual=sha(d/f'final-{i}.rgba16le');require(actual==f['surface_sha256_caller_readback']==f['body']['history_sha256_after'],'actual native output binding')
            require(sha(d/f'diag/frame-{i}/tex_out_156.bin')==actual,'actual history binding')
            ports=[key for key,target in mapping.items() if f['regions'][key]['sha256']!=rtx['boundary_ports'][2*i+1][target]]
            observed.append(actual);frames.append({'frame':i,'output_matches_RTX':actual==rtx['outputs'][i],'bad_encoder_ports':ports})
        if signature is not None:require(observed==signature,'native divergence reproduces')
        signature=observed
        require(all(f['output_matches_RTX'] and not f['bad_encoder_ports'] for f in frames[:12]),'first12 matched')
        require(not frames[12]['output_matches_RTX'] and 'block0_skip' not in frames[12]['bad_encoder_ports'] and len(frames[12]['bad_encoder_ports'])==5,'first divergence inside early encoder')
        require(all(not f['output_matches_RTX'] for f in frames[12:]),'later divergence preserved')
        rows.append({'replica':name,'pid':r['pid'],'receipt_sha256':sha(d/'receipt.json'),'frames':frames})
    return {'schema':'neuralstudio.nr.sixteen-frontier.v1','result':'reproduced_frame12_encoder_counterexample','RTX_matrix_root':h,'runs':rows,'first_divergent_frame':12,'matching_initial_frames_per_replica':12,'frame12_RTX':rtx['outputs'][12],'frame12_Mac':signature[12],'block0_skip_at_first_divergence_matches':True,'sixteen_frame_parity':False,'full_goal_complete':False,'next_test':'encoder temporal sampler hardware vs frozen fitted implementation, unchanged body/post; hypothesis not yet accepted','verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve evidence')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'},indent=2))
if __name__=='__main__':main()
