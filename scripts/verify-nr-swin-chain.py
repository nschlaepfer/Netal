#!/usr/bin/env python3
"""Accept only a fresh, dual-poison, full-grid live Block-1/2/3 prefix."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return sha(path.read_bytes())
def require(ok,message):
    if not ok:raise ValueError(message)
def checked_receipt(path):
    r=json.loads(path.read_text());body=dict(r);root=body.pop('receipt_root_sha256')
    require(root==sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode()),'receipt root drift')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--reference',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'preserve verification')
    repo=Path(__file__).resolve().parents[1]
    r=checked_receipt(a.reference/'receipt.json')
    require(r['runtime_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e','wrong model')
    require(r['graph_root_sha256']=='3de475044e840d59f0235f0ff12763e99950909f81e5ae8b8c832375a25c88c5','wrong graph')
    require(r['comparison_only'] is True and r['full_model_inference'] is False and r['result']=='pass','reference scope')
    require([b['block'] for b in r['blocks']]==[1,2,3],'wrong prefix')
    expected={}
    for block in r['blocks']:
        n=block['block'];data=(a.reference/f'block{n}.e4m3').read_bytes()
        require(len(data)==17694720 and sha(data)==block['logical_output_sha256'],'reference tensor drift')
        require(file_sha(a.reference/'weights'/f'block{n}.bin')==block['weight_sha256'],'weight drift')
        expected[n]=data
    input_reference=checked_receipt(repo/'build/nr_swin_block/reference-v2/receipt.json')
    input_sha=file_sha(repo/'build/nr_swin_block/reference-v2/input.e4m3')
    require(input_sha==input_reference['flat_input_sha256'],'recorded Block-0 input drift')
    pids=set();runs=[];compared=0;sample_hashes={}
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            run=a.attempt/f'{poison}-{ordinal}';rr=json.loads((run/'run.json').read_text())
            require(rr['schema']=='neuralstudio.nr.swin-c32-live-chain.v1' and rr['result']=='pass','incomplete run')
            require(rr['pid'] not in pids,'reused process');pids.add(rr['pid'])
            require(rr['poison']==int(poison,16) and rr['input_sha256']==input_sha,'run ingress drift')
            require(rr['expected_intermediate_opened'] is False and rr['live_device_handoffs'] is True,'not a live prefix')
            require(rr['coverage_errors']==rr['canary_errors']==rr['immutable_errors']==0,'run guard failure')
            require([b['block'] for b in rr['blocks']]==[1,2,3],'missing block')
            for block in rr['blocks']:
                n=block['block'];data=(run/f'block{n}.e4m3').read_bytes()
                require(data==expected[n] and sha(data)==block['output_sha256'],f'RTX mismatch: {run.name} block {n}')
                compared+=len(data)
                paths=sorted((run/f'block{n}').glob('first-window-*'))
                require(len(paths)==10,'missing diagnostic samples')
                for path in paths:
                    key=f'block{n}/{path.name}';digest=file_sha(path)
                    require(key not in sample_hashes or sample_hashes[key]==digest,'intermediate poison/process dependence')
                    sample_hashes[key]=digest
            runs.append({'pid':rr['pid'],'poison':rr['poison'],'receipt_sha256':file_sha(run/'run.json'),
                         'output_sha256':{str(b['block']):b['output_sha256'] for b in rr['blocks']}})
    sources={}
    for path in sorted((a.attempt/'source').rglob('*')):
        if path.is_file():
            name=str(path.relative_to(a.attempt/'source'));digest=file_sha(path)
            require(file_sha(repo/name)==digest,'source changed since build: '+name);sources[name]=digest
    require(len(sources)==11,'incomplete frozen source closure')
    result={'schema':'neuralstudio.nr.swin-c32-live-chain-verification.v1','reference_root_sha256':r['receipt_root_sha256'],
            'input_reference_root_sha256':input_reference['receipt_root_sha256'],'input_sha256':input_sha,
            'fresh_metal_processes':len(pids),'full_grid_compared_bytes':compared,'mismatched_bytes':0,
            'write_coverage_exactly_once':True,'canaries_intact':True,'inputs_immutable':True,'runs':runs,
            'source_sha256':sources,'host_sha256':file_sha(a.attempt/'probe'),'metallib_sha256':file_sha(a.attempt/'chain.metallib'),
            'toolchain_sha256':{name:file_sha(a.attempt/name) for name in ('metal-version.txt','clang-version.txt')},
            'sample_poison_invariance_sha256':sample_hashes,
            'claims':{'complete_blocks':[1,2,3],'isolated_recorded_block0_input':True,'native_block0_producer':False,
                      'live_block1_to_block2_to_block3':True,'shifted_window_zero_padding':True,'full_model_inference':False},
            'result':'pass'}
    result['receipt_root_sha256']=sha(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256'])
    print(f'full_grid_compared_bytes={compared} mismatched_bytes=0')


if __name__=='__main__':main()
