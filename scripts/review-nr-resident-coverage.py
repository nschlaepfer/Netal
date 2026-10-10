#!/usr/bin/env python3
"""Counterexamples to aggregate per-address counts as a per-launch coverage proof."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path


def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    contract=a.peer/'docs/NR_RESIDENT_ENCODE_CONTRACT.md';text=contract.read_text()
    assert 'expected-multiplicity' in text and 'a word written by k launches must show k' in text
    false_accepts=[]
    for counts in itertools.product(range(3),repeat=2):
        aggregate=sum(counts)==2;per_launch=counts==(1,1)
        if aggregate and not per_launch:false_accepts.append({'expected_per_launch':[1,1],'actual_per_launch':list(counts),'aggregate_expected':2,'aggregate_actual':sum(counts),'aggregate_passes':True,'per_launch_passes':False})
    assert len(false_accepts)==2
    padding={'expected_per_launch':[0,1],'actual_per_launch':[1,0],'aggregate_expected':1,'aggregate_actual':1,'aggregate_passes':True,'per_launch_passes':False}
    chain_path=a.peer/'build/nr_full/combined3-v1/pilot/diag/frame-0/chain-run.json';chain=json.loads(chain_path.read_text())
    kinds={}
    for launch in chain['launches']:kinds[launch['kind']]=kinds.get(launch['kind'],0)+1
    r={'schema':'neuralstudio.nr.resident-coverage-review.v1','result':'aggregate_counts_insufficient_for_per_launch_guarantee',
       'contract_sha256':hashlib.sha256(contract.read_bytes()).hexdigest(),'counterexamples':false_accepts+[padding],
       'observed_chain_record_count':len(chain['launches']),'observed_chain_kinds':kinds,
       'chain_sha256':hashlib.sha256(chain_path.read_bytes()).hexdigest(),
       'remedies':['Partition a coverage slab by logical launch/output/index, with no cross-launch aliasing of counters.',
       'Or validate each launch before its counters are reused, using an ordered GPU checker that latches failures and first-failure identity. Final CPU validation reads that result.'],
       'limitations':'The examples prove insufficiency of summing overlapping per-address counts. Disjoint per-launch slices are not rejected. No GPU execution, performance claim, or peer-file change.',
       'reviewer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    r['receipt_root_sha256']=canonical(r);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
