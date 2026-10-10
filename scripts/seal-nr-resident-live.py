#!/usr/bin/env python3
"""Seal the eight live output/port/history audits without promoting omitted checks."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
ROOTS={
 'black3':'de29f85850789236dc39f77aa0b991492f5910440d58e952474d6fd91724a969',
 'checker3':'88c7c65c73364b7b4c540b8bfbb6e5d7f0eee623772ed99d11e3ac0c073dbc06',
 'dense':'f052e0bc99c26399b50b61f313d2bb0b75f15f4d827fe2963e299567764bc3ca',
 'images3':'7390ea36af7f68b16270b7a2b3fbaece552a84c4ab565c3b9b50f0b93efe07f1',
 'midreset8':'9816f88eedb01651329014b110deea3bbdb81fee5ad55eb0bbf169443e391776',
 'uniform3':'bfb9758a14ed568e0cf6d35948dc24e5a5f2ed21934059fc4a284923fa65fa77',
 'spatial3':'4e728b11bc28533521d2c6312fdac13e6d611bbb0fc350ad2653fef6a54b66ed',
 'combined3':'f8ec10be026815ffc841ab9f6ef82544821b5ae288cc5820f0fdbc8bb01f46b8'}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1];folder=repo/'build/nr_resident_review'
    rows=[];pids=set();frames=0;bytes_checked=0;ports=0
    for name,root in ROOTS.items():
        path=folder/f'live-{name}-v1.json';r=json.loads(path.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==root
        assert r['result']=='pass_live_outputs_ports_history_with_explicit_omissions' and r['configuration']==name and not r['strong_live_intermediate_acceptance'] and not r['production_promotion']
        assert r['verifier_sha256']==sha(repo/'scripts/audit-nr-resident-live.py')
        rpids={f['pid'] for f in r['runs']};assert len(rpids)==4 and not(pids&rpids);pids|=rpids
        for file,digest in r['artifacts'].items():assert sha(Path(file))==digest,'artifact drift '+file
        frames+=len(r['runs']);bytes_checked+=r['actual_final_post_history_bytes_checked'];ports+=r['encoder_boundary_hashes_checked']
        rows.append({'case':name,'audit_root':h,'frames':len(r['runs']),'matrix':r['matrix']})
    assert len(pids)==32 and frames==108 and ports==648
    r={'schema':'neuralstudio.nr.resident-live-set.v1','result':'pass_bounded_live_outputs_ports_history','cases':rows,'fresh_processes':len(pids),'frames':frames,'actual_output_post_history_bytes_rehashed':bytes_checked,'RTX_encoder_port_hashes_checked':ports,
       'strong_live_checkpoint_acceptance':False,'resident_promoted':False,'speedup_accepted':False,'full_goal_complete':False,
       'next_gate':'Strong per-launch live checked pairs on the same inputs, followed by telemetry parity and fair timed comparisons.',
       'scope':'Eight fixed recipes only. Intermediate/pool snapshots omitted in this production-style mode remain explicitly unaccepted; 64-bit digests are supplemental.',
       'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
