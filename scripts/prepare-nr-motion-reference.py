#!/usr/bin/env python3
"""Prepare bounded zero/uniform-fractional-motion RTX observations."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess

PARENT='d5da3e2d6d3ff2ad2545bc1b50bb181ff389bdf8ee663cdb8f2800dcf00543e8'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed parent anchor '+a[:80])
    return s.replace(a,b)
def host(s):
    s=once(s,'      NrColorCase(color_host,kWidth,kHeight,case_id);', '''      NrColorCase(color_host,kWidth,kHeight,case_id);
      char motion_text[8]{};
      const DWORD motion_len=GetEnvironmentVariableA("NEURALSTUDIO_NR_MOTION_CASE",motion_text,sizeof(motion_text));
      if(motion_len!=1 || (motion_text[0]!='0' && motion_text[0]!='1'))throw std::runtime_error("explicit motion case0/1 required");
      if(motion_text[0]=='1')for(size_t i=0;i<motion_host.size();i+=2){motion_host[i]=0x3400u;motion_host[i+1]=0xb800u;}
      std::cout<<"runtime.motion_case="<<motion_text[0]<<'\\n';''')
    return once(s,'rgba16-color-cases-v1_depth-bands-v1_motion-zero','rgba16-color-cases-v1_depth-bands-v1_motion-cases-v1')
def runner(s):
    if '[int]$MotionCase' in s:raise ValueError('motion wrapper already transformed')
    s=once(s,'[ValidateRange(0,2)][int]$ColorCase=0,','[ValidateRange(0,2)][int]$ColorCase=0,[ValidateRange(0,1)][int]$MotionCase=0,')
    s=once(s,'$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$ColorCase;','$env:NEURALSTUDIO_NR_MOTION_CASE=[string]$MotionCase;$env:NEURALSTUDIO_NR_COLOR_CASE=[string]$ColorCase;')
    return once(s,'color_case=$ColorCase;','color_case=$ColorCase;motion_case=$MotionCase;')
def capture(s):
    assert s.count('rgba16-color-cases-v1_depth-bands-v1_motion-zero')==2
    return s.replace('rgba16-color-cases-v1_depth-bands-v1_motion-zero','rgba16-color-cases-v1_depth-bands-v1_motion-cases-v1')
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in parent['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    for n,f in [('color_cases_trace.cpp',host),('run-nr-reset-sequence.ps1',runner),('capture-color.ps1',capture)]:(a.output/n).write_text(f((a.parent/n).read_text()))
    motion=struct.pack('<HH',0x3400,0xb800)*(1920*1080)
    r={'schema':'neuralstudio.nr.motion-reference-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':3,'motion_cases':{'0':'zero','1':'uniform RG16F(0.25,-0.5), unchanged MVecScaleX/Y=1'},'motion1_sha256':hashlib.sha256(motion).hexdigest(),'motion_bytes':len(motion),'proxy_unchanged':True,'model_unchanged':True,'expected_bytes_staged':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'motion1_sha256':r['motion1_sha256']}))
if __name__=='__main__':main()
