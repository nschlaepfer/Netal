#!/usr/bin/env python3
"""Enable the already compiled color recipes in the sealed three-call observer."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

PARENT='88d2006591a451aac66a16547134c80a7e1c7ae2c714978a2ec4931a1e1fffe6'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed parent anchor '+a[:80])
    return s.replace(a,b)
def runner(s):
    s=once(s,"[ValidateRange(0,1)][int]$SecondReset=1,","[ValidateRange(0,1)][int]$SecondReset=1,[ValidateRange(0,2)][int]$ColorCase=0,")
    s=once(s,"$env:NEURALSTUDIO_NR_COLOR_CASE='0'","$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$ColorCase")
    s=once(s,'resets=@(1,$SecondReset,$SecondReset);','resets=@(1,$SecondReset,$SecondReset);color_case=$ColorCase;')
    s=once(s,"  $record.first_matches_control=($cap.output_sha256 -eq 'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2')", "  $firstPins=@('a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','1391787165701ef795a2259432a8672ec3bba779b9c09c7abe602848d8677f29','4e3df0c2d6c15ba20f35bbd2f3374d98d52687bd6345e4fe1191f33aafe16379')\n  $record.first_matches_control=($cap.output_sha256 -eq $firstPins[$ColorCase])")
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in parent['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    (a.output/'run-nr-reset-sequence.ps1').write_text(runner((a.parent/'run-nr-reset-sequence.ps1').read_text()))
    r={'schema':'neuralstudio.nr.varied-temporal-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':3,'case_ids':{'0':'original','1':'black','2':'checker'},'supported_resets':[[1,1,1],[1,0,0]],'snapshot_launch_ids':[2,24,158,180,314,336],'harness_proxy_model_unchanged':True,'expected_bytes_staged':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
