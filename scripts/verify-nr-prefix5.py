#!/usr/bin/env python3
"""Verify a live five-block encoder prefix; never claim a complete model."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()
def require(ok,message):
    if not ok:raise ValueError(message)


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'preserve verification');repo=Path(__file__).resolve().parents[1]
    targets={};roots=[]
    for folder,fields in [('nr_swin_chain/reference-v1',None),('nr_swin_downsample/reference-v1',{'block4.e4m3':'core_logical_sha256','block4-downsample.e4m3':'downsample_logical_sha256'}),('nr_swin64/reference-v1',{'output.e4m3':'logical_output_sha256'})]:
        directory=repo/'build'/folder;r=json.loads((directory/'receipt.json').read_text());body=dict(r);root=body.pop('receipt_root_sha256')
        require(sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode())==root,'reference receipt drift');roots.append(root)
        require(r['runtime_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e','wrong runtime')
        hashes={f'block{b["block"]}.e4m3':b['logical_output_sha256'] for b in r['blocks']} if fields is None else {name:r[key] for name,key in fields.items()}
        for name,digest in hashes.items():
            data=(directory/name).read_bytes();require(sha(data)==digest,'reference tensor drift');targets['block5.e4m3' if name=='output.e4m3' else name]=data
    input_sha=sha((repo/'build/nr_swin_block/reference-v2/input.e4m3').read_bytes())
    pids=set();runs=[];compared=0;samples={}
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            run=a.attempt/f'{poison}-{ordinal}';r=json.loads((run/'run.json').read_text());d=json.loads((run/'downsample.json').read_text())
            require(r['schema']=='neuralstudio.nr.prefix5-live-chain.v1' and r['result']=='pass','run schema/result')
            require(r['pid'] not in pids and d['pid']==r['pid'],'process reuse');pids.add(r['pid'])
            require(r['input_sha256']==input_sha and r['expected_intermediate_opened'] is False and r['live_device_handoffs'] is True,'ingress violation')
            require([b['block'] for b in r['blocks']]==[1,2,3,4,5],'prefix incomplete')
            for q in (r,d):require(q['poison']==int(poison,16) and q['coverage_errors']==q['canary_errors']==q['immutable_errors']==0,'guard violation')
            hashes={f'block{b["block"]}.e4m3':b['output_sha256'] for b in r['blocks']};hashes['block4-downsample.e4m3']=d['output_sha256']
            for name,data in targets.items():require((run/name).read_bytes()==data and hashes[name]==sha(data),'full RTX mismatch '+name);compared+=len(data)
            for b in range(1,6):
                paths=sorted((run/f'block{b}').glob('first-window-*'));require(len(paths)==(11 if b==5 else 10),'missing samples')
                for path in paths:
                    key=f'block{b}/{path.name}';digest=sha(path.read_bytes());require(key not in samples or samples[key]==digest,'sample poison dependence');samples[key]=digest
            for stage in ('ffn','qkv'):
                target=(repo/f'build/nr_swin64/{stage}-analysis-v1/decoded.f16').read_bytes()
                require((run/f'block5/first-window-{stage}.f16').read_bytes()==target,'fresh RTX intermediate mismatch')
            scores=(run/'block5/first-window-scores.f16').read_bytes();gather=scores[:4096]+scores[8192:12288]
            require(gather==(repo/'build/nr_swin64/score0-analysis-v1/decoded.f16').read_bytes(),'fresh RTX score mismatch')
            runs.append({'pid':r['pid'],'poison':r['poison'],'receipt_sha256':sha((run/'run.json').read_bytes()),'output_sha256':hashes})
    sources={}
    for path in (a.attempt/'source/generated').iterdir():
        if path.name not in ('probe.mm','prefix5.metal'):
            name='src/nr_swin/'+path.name;digest=sha(path.read_bytes());require(sha((repo/name).read_bytes())==digest,'source drift');sources[name]=digest
    for path in (a.attempt/'source').glob('*.py'):
        name='scripts/'+path.name;digest=sha(path.read_bytes());require(sha((repo/name).read_bytes())==digest,'script drift');sources[name]=digest
    path=a.attempt/'source/test-nr-prefix5.sh';sources['scripts/test-nr-prefix5.sh']=sha(path.read_bytes())
    require(sources['scripts/test-nr-prefix5.sh']==sha((repo/'scripts/test-nr-prefix5.sh').read_bytes()),'launcher drift')
    r={'schema':'neuralstudio.nr.prefix5-verification.v1','reference_roots_sha256':roots,'fresh_metal_processes':len(pids),'full_grid_compared_bytes':compared,
       'mismatched_bytes':0,'write_coverage_exactly_once':True,'canaries_intact':True,'inputs_immutable':True,'runs':runs,'source_sha256':sources,
       'generated_host_sha256':sha((a.attempt/'source/generated/probe.mm').read_bytes()),'generated_shader_sha256':sha((a.attempt/'source/generated/prefix5.metal').read_bytes()),
       'host_sha256':sha((a.attempt/'probe').read_bytes()),'metallib_sha256':sha((a.attempt/'prefix.metallib').read_bytes()),
       'sample_poison_invariance_sha256':samples,'claims':{'complete_blocks':[1,2,3,4,5],'live_c32_downsample_to_c64':True,'isolated_recorded_block0_input':True,'full_model_inference':False},'result':'pass'}
    r['receipt_root_sha256']=sha(json.dumps(r,sort_keys=True,separators=(',',':')).encode());a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n')
    print('receipt_root_sha256='+r['receipt_root_sha256']);print(f'compared_bytes={compared} mismatched_bytes=0')


if __name__=='__main__':main()
