#!/usr/bin/env python3
"""Independently check every lane, transfer hash and fresh two-poison process."""
import argparse
import hashlib
import json
from pathlib import Path
import struct

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):return json.loads(path.read_text(encoding='utf-8-sig'))

def main():
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('source',type=Path)
    p.add_argument('results',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    assert not a.output.exists(), 'Do not overwrite verification evidence'
    spec=read(a.spec);inputs=read(a.source/'inputs.json');job=read(a.results/'receipt.json')
    errors=[]
    def check(ok,label):
        if not ok:errors.append(label)
    check(inputs['spec_sha256']==sha(a.spec),'spec hash')
    root=dict(inputs);claimed=root.pop('receipt_root_sha256')
    check(hashlib.sha256(json.dumps(root,sort_keys=True,separators=(',',':')).encode()).hexdigest()==claimed,'input receipt root')
    check(job['inputs_root_sha256']==claimed,'remote source root')
    for name,digest in inputs['files'].items():check(sha(a.source/name)==digest,'source '+name)
    check(sha(a.results/'sass.txt')==job['sass_sha256'],'SASS hash')
    check('QMMA' in (a.results/'sass.txt').read_text(encoding='utf-16'),'actual QMMA SASS')
    for item in job['artifacts']:
        path=a.results/'output'/item['name']
        check(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'transfer '+item['name'])
    cases=spec.get('cases',[spec]);lanes=[lane for case in cases for lane in case['lanes']];count=len(lanes)*2
    operand_words=[int(x,16) for lane in lanes for k in ('a','b','c') for x in lane[k]]
    check((a.source/'operands.bin').read_bytes()==struct.pack(f'<{len(operand_words)}I',*operand_words),'operand mapping')
    hypotheses=({k:[int(x,16) for lane in lanes for x in lane['d'][k]] for k in spec['candidates']} if 'cases' in spec else
                {k:[int(x,16) for lane in lanes for x in lane[k]] for k in ('d_seq','d_c_last')})
    runs=[];pids=[];hashes=[]
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            prefix=a.results/'output'/f'{poison}-{ordinal}';r=read(prefix.with_suffix('.json'))
            data=prefix.with_suffix('.bin').read_bytes();words=struct.unpack(f'<{count}I',data)
            check(r['result']=='pass' and r['coverage_errors']==0 and r['canary_errors']==0 and r['input_immutable'],'gates '+prefix.name)
            check(r['expected_gpu_input'] is False and r['poison']==int(poison,16),'configuration '+prefix.name)
            check(prefix.with_suffix('.input-after.bin').read_bytes()==(a.source/'operands.bin').read_bytes(),'input immutable '+prefix.name)
            check(struct.unpack(f'<{count}I',prefix.with_suffix('.coverage.bin').read_bytes())==(1,)*count,'coverage '+prefix.name)
            pids.append(r['pid']);hashes.append(sha(prefix.with_suffix('.bin')))
            runs.append({'run':prefix.name,'pid':r['pid'],'output_sha256':hashes[-1],
                         'hypothesis_mismatches':{k:[{'case':i//64,'lane':(i//2)%32,'register':i%2,'actual':f'{x:08x}','hypothesis':f'{v[i]:08x}'} for i,x in enumerate(words) if x!=v[i]] for k,v in hypotheses.items()},
                         'lanes':[{'case':i//32,'lane':i%32,'d':[f'{words[i*2]:08x}',f'{words[i*2+1]:08x}']} for i in range(len(lanes))]})
    check(len(set(pids))==4,'four fresh processes');check(len(set(hashes))==1,'repeat and two poison invariance')
    receipt={'schema':'neuralstudio.nr.qmma-operands-verification.v1','scope':f'{len(cases)} QMMA operand sets, not a general rule or model parity',
             'spec_sha256':sha(a.spec),'inputs_root_sha256':claimed,'job_receipt_sha256':sha(a.results/'receipt.json'),
             'runs':runs,'errors':errors,'result':'fail' if errors else 'pass','full_model_inference':False}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'root':receipt['receipt_root_sha256'],'errors':errors,'runs':[{'run':r['run'],'output_sha256':r['output_sha256'],'hypothesis_mismatch_counts':{k:len(v) for k,v in r['hypothesis_mismatches'].items()}} for r in runs]},indent=2))
    return bool(errors)

if __name__=='__main__':raise SystemExit(main())
