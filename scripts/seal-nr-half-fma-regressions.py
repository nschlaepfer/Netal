#!/usr/bin/env python3
"""Seal only the complete requested seven-case independently audited matrix."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
AUDITS={
 'images3':('images3-audit-v3.json','3a70a921861777cc1f32cd9ad0dca1200f5bb04be655412d03fae1c135a2d135'),
 'midreset8':('midreset8-audit-v2.json','ada1c551de1d13880d67185cb92c19e5163cb432c0ad3fa3f9cee59d6dad10bc'),
 'uniform3':('uniform3-audit-v2.json','8593696ae25de602640db55f4c367c5a38865e45a43acc40170e0995bebd9c45'),
 'spatial3':('spatial3-audit-v1.json','1e0bc8fa1bc72cd55ca0f666c8d615b238aae73658d76c2720c5d88ecc2611d6'),
 'black3':('black3-audit-v1.json','181d315117c9cc547df12f855876d1a50e28820290ad970cccc5151a1829ed91'),
 'checker3':('checker3-audit-v1.json','88d2ea852c257cca3b89f7175db58894fc83569b5658dac94ff47366d8bf1c4a'),
 'dense':('dense-audit-v1.json','e6c19197a61b7307cd3a8be3461902fde1f26aeeea5e4ca43846cc1ec2863eda'),
}


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];folder=repo/'build/nr_half_fma_regression';rows=[];pids=set();frames=0;size=0
    verifier=repo/'scripts/audit-nr-half-fma-regression.py'
    for name,(file,root) in AUDITS.items():
        r=json.loads((folder/file).read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==root
        assert r['configuration']==name and r['result']=='pass_bounded_prior_mode_regression' and r['verifier_sha256']==sha(verifier)
        rpids={x['pid'] for x in r['runs']};assert len(rpids)==4 and not(pids&rpids);pids|=rpids
        for path,digest in r['artifacts'].items():assert sha(Path(path))==digest,'post-audit artifact drift '+path
        frames+=len(r['runs']);size+=r['actual_final_post_history_bytes_checked']
        rows.append({'configuration':name,'audit_root':root,'audit_file':str(folder/file),'frames':len(r['runs']),'matrix':r['matrix'],'encoder_root':r['encoder_root']})
    assert len(pids)==28 and frames==96
    r={'schema':'neuralstudio.nr.half-fma-regression-set.v1','result':'pass_bounded_seven_case_regression','configurations':rows,
       'fresh_processes':28,'native_frames':frames,'native_final_post_history_bytes_checked':size,'artifacts_rehashed_after_audits':True,
       'full_goal_complete':False,'production_promotion':False,'next_gate':'Re-run the existing test-only fault/recovery controls on the candidate joint build; combined changing-image/spatial reference is a separate new case.',
       'excluded':['black/checker missing-motion-path pre-inference failures','original-image all-reset extra: not dense COLOR CASE3'],
       'scope':'Seven previously accepted configurations only. No general sequence, arbitrary motion, resident integration or speed claim.',
       'verifier_sha256':sha(verifier),'sealer_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
