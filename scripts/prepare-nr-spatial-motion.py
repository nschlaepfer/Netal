#!/usr/bin/env python3
"""Prepare a signed spatial motion field, preserving zero/uniform controls."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np

PARENT='c7afbe92e4905160a6ae015deae7f64dd8b43f4000c36c2b2bed4532c3627026'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def once(s,a,b):
    if s.count(a)!=1:raise ValueError('changed source anchor '+a[:90])
    return s.replace(a,b)
def recipe():
    x=np.arange(1920)[None,:];y=np.arange(1080)[:,None]
    return np.stack((((37*x+17*y)%257-128)/16,((11*x+53*y+73)%257-128)/16),axis=2).astype('<f2').tobytes()
def host(s):
    s=once(s,"(motion_text[0]!='0' && motion_text[0]!='1')","(motion_text[0]!='0' && motion_text[0]!='1' && motion_text[0]!='2')")
    s=once(s,'explicit motion case0/1 required','explicit motion case0/1/2 required')
    s=once(s,'      std::cout<<"runtime.motion_case="', '''      if(motion_text[0]=='2'){
        // Exact binary16 encoding of q/16, q in [-128,128]; no float conversion or expected output.
        auto encode=[](int q)->uint16_t{if(q==0)return 0;const unsigned m=unsigned(q<0?-q:q);unsigned e=0;while((1u<<(e+1u))<=m)++e;return uint16_t((q<0?0x8000u:0u)|((e+11u)<<10u)|((m-(1u<<e))<<(10u-e)));};
        for(unsigned y=0;y<kHeight;++y)for(unsigned x=0;x<kWidth;++x){const size_t i=(size_t(y)*kWidth+x)*2;motion_host[i]=encode(int((37u*x+17u*y)%257u)-128);motion_host[i+1]=encode(int((11u*x+53u*y+73u)%257u)-128);}
      }
      std::cout<<"runtime.motion_case="''')
    return once(s,'motion-cases-v1','motion-cases-v2')
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    parent=json.loads((a.parent/'source-receipt.json').read_text());b=dict(parent);h=b.pop('receipt_root_sha256');assert canonical(b)==h==PARENT
    for n,d in parent['files'].items():assert sha(a.parent/n)==d,n
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    (a.output/'color_cases_trace.cpp').write_text(host((a.parent/'color_cases_trace.cpp').read_text()))
    run=once((a.parent/'run-nr-reset-sequence.ps1').read_text(),'[ValidateRange(0,1)][int]$MotionCase=0','[ValidateRange(0,2)][int]$MotionCase=0');(a.output/'run-nr-reset-sequence.ps1').write_text(run)
    cap=(a.parent/'capture-color.ps1').read_text();assert cap.count('motion-cases-v1')==2;(a.output/'capture-color.ps1').write_text(cap.replace('motion-cases-v1','motion-cases-v2'))
    data=recipe();r={'schema':'neuralstudio.nr.spatial-motion-source.v1','parent_source_root':h,'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[1],text=True).strip(),'frames':3,'motion_case':2,'motion_formula':{'x':'(((37*x+17*y)%257)-128)/16','y':'(((11*x+53*y+73)%257)-128)/16'},'range':[-8,8],'step':0.0625,'motion2_sha256':hashlib.sha256(data).hexdigest(),'motion_bytes':len(data),'proxy_model_unchanged':True,'expected_bytes_staged':False,'executed':False,'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    r['receipt_root_sha256']=canonical(r);(a.output/'source-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'motion2_sha256':r['motion2_sha256']}))
if __name__=='__main__':main()
