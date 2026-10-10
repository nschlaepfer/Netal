#!/usr/bin/env python3
"""Seal four fresh native arithmetic-domain runs; reject weakened gate receipts."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
N=1<<24
EXPECTED='daa0f10f8a04d625c38b277bfc9fef2d4fd68622fd9f6203024e9d7d0415975a'
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(ok,label):
    if not ok:raise ValueError(label)
def validate(r):
    require(r['schema']=='neuralstudio.nr.noise-math-exact-mac.v1','schema')
    require(r['result']=='pass' and r['records']==N and r['fields']==6,'run result/extent')
    require(r['output_sha256']==EXPECTED,'output identity')
    require(r['mismatching_words']==0 and r['field_mismatches']==[0]*6,'arithmetic differences')
    require(r['coverage_errors']==0 and r['canary_errors']==0 and r['corrections_immutable'] is True,'memory gates')
    require(r['model_inputs'] is False and r['expected_gpu_input'] is False and r['radius_lookup'] is False,'ingress gates')
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('corrections',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(not a.output.exists(),'preserve receipt')
    c=json.loads((a.corrections/'receipt.json').read_text());cr=dict(c);root=cr.pop('receipt_root_sha256')
    require(canonical(cr)==root=='18e80b3b58597924061e94ea77f8558609407805e2fb31249c572cc6d6be0aea','correction root')
    for name,row in c['files'].items():require(sha(a.corrections/name)==row['sha256'] and (a.corrections/name).stat().st_size==row['bytes'],'correction identity')
    runs=[]
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            path=a.attempt/f'{poison}-{ordinal}';r=json.loads((path/'run.json').read_text());validate(r)
            require(r['poison']==int(poison,16),'poison identity')
            require((path/'output.bin').stat().st_size==N*6*4 and sha(path/'output.bin')==EXPECTED,'actual output bytes')
            cov=np.memmap(path/'coverage.bin',dtype='<u4',mode='r');require(cov.size==N*6 and np.all(cov==1),'actual coverage')
            runs.append({'run':path.name,**r,'receipt_sha256':sha(path/'run.json'),'coverage_sha256':sha(path/'coverage.bin')})
    require(len({r['pid'] for r in runs})==4,'four fresh processes')
    negative=[]
    for key,value in [('result','fail'),('records',N-1),('output_sha256','0'*64),('mismatching_words',1),('field_mismatches',[0,0,1,0,0,0]),('coverage_errors',1),('canary_errors',1),('corrections_immutable',False),('model_inputs',True),('expected_gpu_input',True),('radius_lookup',True)]:
        altered=copy.deepcopy(runs[0]);altered[key]=value
        try:validate(altered)
        except ValueError:negative.append({'case':key,'rejected':True})
        else:raise ValueError('weakened gate accepted: '+key)
    files={str(p.relative_to(a.attempt)):sha(p) for p in sorted((a.attempt/'source').rglob('*')) if p.is_file()}
    files.update({name:sha(a.attempt/name) for name in ('math.metallib','probe')})
    r={'schema':'neuralstudio.nr.noise-math-native-verification.v1','correction_root':root,'parent_rtx_root':c['parent_rtx_root'],
       'runs':runs,'files':files,'negative_audit':negative,'compared_bytes':4*N*6*4,'result':'pass',
       'scope':'Complete positive 24-bit unit-grid LG2 and scaled-argument sin/cos; normalized SQRT domains and all induced radius arguments. Not all FP32 inputs, model inference, or performance acceptance.',
       'model_parity':False,'radius_lookup':False,'expected_model_intermediate_ingress':False}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'compared_bytes':r['compared_bytes'],'negative_cases':len(negative),'result':'pass'},indent=2))
if __name__=='__main__':main()
