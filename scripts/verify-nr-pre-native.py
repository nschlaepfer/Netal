#!/usr/bin/env python3
"""Accept a complete native Block0 group only with four fresh two-poison runs."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import numpy as np
EXPECTED={'input_sha256':'81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342',
          'weight_sha256':'5fe2ab86289f7b21253d00368ce7a95f005ec3c8476abd7db22550b918962a5c',
          'feature_sha256':'49abcec0ab2e1068a03339564aa99c5358c0720eb06fe4930ccc803aa2b0a06f',
          'skip_sha256':'26080cceab714d2cdb03532c6e911693283ef178127660351e59c99e612799c6',
          'encoder_sha256':'de4789425897a4ffa3d5646670c12385d25109dcad60609c1729dca2da4a33a0',
          'delta_sha256':'768da747f3c7c799d768cd4410195934292cc838fbc083ec6a3ab9c04184f37b',
          'escape_sha256':'c2e095109b1f220ed289ea09f5b2d10a4df9f15f5366355e7fdab30b455d40f5'}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(ok,label):
    if not ok:raise ValueError(label)
def validate(r):
    require(r['schema']=='neuralstudio.nr.pre-native-candidate.v1' and r['result']=='pass','run status')
    for k,v in EXPECTED.items():require(r[k]==v,'identity '+k)
    for k in ('skip_mismatching_bytes','encoder_mismatching_bytes','coverage_errors','canary_errors','immutable_errors'):require(r[k]==0,k)
    require(r['expected_gpu_input'] is False and r['live_device_handoffs'] is True and r['instruction_corrections'] is True,'live producer provenance')
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--math-receipt',required=True,type=Path);p.add_argument('--rtx-receipt',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(not a.output.exists(),'preserve receipt');math=json.loads(a.math_receipt.read_text());rtx=json.loads(a.rtx_receipt.read_text())
    for parent,expected in ((math,'6cafbf4a8efe30130f82c831e3e1b4d9639d2afe5ff223d6acb167f13ae45ec2'),(rtx,'a1cea9cba2609b9e2ff9f61a63534ff1b83d4f09fe41a57c3395b88ba5b64d20')):
        body=dict(parent);root=body.pop('receipt_root_sha256');require(canonical(body)==root==expected and parent['result']=='pass','accepted parent')
    manifest=json.loads((a.attempt/'source/source-manifest.json').read_text())
    for name,h in manifest['files'].items():require(sha(a.attempt/'source'/name)==h,'frozen source '+name)
    runs=[];first_windows=[]
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            path=a.attempt/f'{poison}-{ordinal}';r=json.loads((path/'run.json').read_text());validate(r);require(r['poison']==int(poison,16),'poison identity')
            for name,key,size in [('features-shared.f16','feature_sha256',70778880),('skip.e4m3','skip_sha256',70778880),('encoder.e4m3','encoder_sha256',17694720)]:
                f=path/name;require(f.stat().st_size==size and sha(f)==EXPECTED[key],'actual data '+name)
            for name,n in [('skip',70778880),('encoder',17694720)]:
                cov=np.memmap(path/(name+'.coverage.bin'),dtype='<u4',mode='r');require(cov.size==n and np.all(cov==1),'actual output coverage '+name)
            stages={p.name:sha(p) for p in sorted(path.glob('first-window-*'))};require(len(stages)==11,'all first-window stage witnesses');first_windows.append(stages)
            runs.append({'run':path.name,**r,'receipt_sha256':sha(path/'run.json')})
    require(len({r['pid'] for r in runs})==4,'four fresh processes');require(all(x==first_windows[0] for x in first_windows),'stage witness repeat/poison equality')
    negative=[]
    changes=[('result','fail'),('skip_mismatching_bytes',1),('encoder_mismatching_bytes',1),('coverage_errors',1),('canary_errors',1),('immutable_errors',1),('expected_gpu_input',True),('live_device_handoffs',False),('instruction_corrections',False)]+[(k,'0'*64) for k in EXPECTED]
    for k,v in changes:
        bad=copy.deepcopy(runs[0]);bad[k]=v
        try:validate(bad)
        except ValueError:negative.append({'case':k,'rejected':True})
        else:raise ValueError('weakened gate accepted '+k)
    files={name:sha(a.attempt/name) for name in ('source/source-manifest.json','pre.metallib','probe')}
    r={'schema':'neuralstudio.nr.native-block0-verification.v1','native_math_root':math['receipt_root_sha256'],'rtx_replay_root':rtx['receipt_root_sha256'],
       'runs':runs,'files':files,'first_window_stage_hashes':first_windows[0],'negative_audit':negative,
       'compared_output_bytes':4*(70778880+17694720),'compared_feature_bytes':4*70778880,'result':'pass',
       'scope':'Complete native pinned color-only frame0 Block0, original image to both native outputs. Not complete NR, other images/configurations, temporal state or performance acceptance.',
       'recorded_model_intermediate_gpu_input':False,'full_model_inference':False,'optimized':False}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'output_bytes':r['compared_output_bytes'],'feature_bytes':r['compared_feature_bytes'],'result':'pass'},indent=2))
if __name__=='__main__':main()
