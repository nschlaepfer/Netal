#!/usr/bin/env python3
"""Verify fresh image -> Block0 -> byte layout -> full encoder and live skips."""
import argparse
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import numpy as np
from nr_candidate_primitive import validate_primitive

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def require(ok,label):
    if not ok:raise ValueError(label)
def receipt(p,root=None):
    r=json.loads(p.read_text())
    if 'receipt_root_sha256' in r:
        body=dict(r);got=body.pop('receipt_root_sha256');require(canon(body)==got,'receipt root '+str(p))
        if root:require(got==root,'parent root '+str(p))
    return r
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();a.attempt=a.attempt.resolve();repo=Path(__file__).resolve().parents[1]
    require(not a.output.exists(),'preserve verification');plan=receipt(a.attempt/'plan.json');commands=json.loads((a.attempt/'commands.json').read_text())
    require(len(commands)==15 and all(c['exit_code']==0 for c in commands),'complete build+12 native process commands')
    parent=receipt(repo/'evidence/specs/nr-prefix22-v1/verification.json','62aafb4f76a52f7bd2430ac3929f33b5c7426b8cc5038993c7eb417caa9f5ee2')
    pre_parent=receipt(repo/'evidence/specs/nr-native-block0-v1/verification.json','ddc9695ac9012f2450337637c4b0dcb1ae6b5d98b53763aff345e0ad20f53bd3')
    require(plan['parent_encoder_root']==parent['receipt_root_sha256'] and plan['parent_block0_root']==pre_parent['receipt_root_sha256'],'parent linkage')
    if 'candidate_bundle' in plan:
        cb=plan['candidate_bundle'];bundle=Path(cb['path']);require(sha(bundle/'manifest.json')==cb['manifest_sha256'],'candidate manifest identity');bm=receipt(bundle/'manifest.json',cb['root'])
        require(bm['primitive_root']==cb['primitive_root']==validate_primitive(bundle,bm),'tested candidate primitive')
        for name,h in bm['files'].items():require(sha(bundle/name)==h,'candidate artifact drift')
    for filename,h in plan['artifacts'].items():require(sha(Path(filename))==h,'material input drift')
    for key in ('source_files','layout_artifacts'):
        for filename,h in plan[key].items():require(sha(a.attempt/filename)==h,'compiled layout/source drift')
    spec=importlib.util.spec_from_file_location('pre_validate',repo/'scripts/verify-nr-pre-native.py');validator=importlib.util.module_from_spec(spec);spec.loader.exec_module(validator)
    expected=parent['runs'][0]['outputs_sha256'];old=repo/'build/nr_encoder22/macos-v2/a5-a'
    native=receipt(old/'native_exports.json');require(sha(old/'native_exports.json')==parent['runs'][0]['native_export_receipt_sha256'],'native target metadata identity')
    # Independent exhaustive address proof: every C16 source byte occurs once.
    gid=np.arange(17694720,dtype=np.uint32);row=gid//32;c=gid%32;window=row//64;token=row%64;tile=token//16;q=token%16
    x=(window%120)*8+(tile//2)*4+q%4;y=(window//120)*8+(tile%2)*4+q//4
    index=((c//16)*960*576+y*960+x)*16+c%16
    require(int(index.min())==0 and int(index.max())==17694719 and np.all(np.bincount(index,minlength=17694720)==1),'layout is not bijective')
    pids=set();runs=[];total=0;native_total=0;stage_witnesses={}
    for ordinal,name in enumerate(('a5-a','a5-b','5a-a','5a-b')):
        root=a.attempt/name;pre=root/'block0';layout=root/'entry-layout';enc=root/'encoder';poison=int(name[:2],16)
        b=receipt(pre/'run.json');validator.validate(b);l=receipt(layout/'run.json');r=receipt(enc/'run.json')
        for child in (b,l,r):require(child['pid'] not in pids and child['poison']==poison,'fresh process/poison');pids.add(child['pid'])
        require(l['schema']=='neuralstudio.nr.block0-encoder-entry-layout.v1' and l['result']=='pass','layout status')
        require(l['producer_pid']==b['pid'] and l['producer_receipt_sha256']==sha(pre/'run.json'),'layout producer receipt link')
        require(l['input_sha256']==b['encoder_sha256'] and l['output_sha256']==r['input_sha256']==parent['input_sha256'],'live entry hashes')
        require(l['input_immutable'] is True and l['expected_gpu_input'] is False and l['coverage_errors']==l['canary_errors']==0,'layout memory gates')
        for filename,key in [('skip.e4m3','skip_sha256'),('encoder.e4m3','encoder_sha256'),('features-shared.f16','feature_sha256')]:require(sha(pre/filename)==b[key],'actual native Block0 output')
        for output,n in [('skip',70778880),('encoder',17694720)]:
            cov=np.memmap(pre/(output+'.coverage.bin'),mode='r',dtype='<u4');require(cov.size==n and np.all(cov==1),'Block0 coverage')
        src=np.memmap(pre/'encoder.e4m3',mode='r',dtype=np.uint8);dst=np.memmap(layout/'encoder-window.e4m3',mode='r',dtype=np.uint8)
        require(np.array_equal(dst,src[index]) and sha(layout/'encoder-window.e4m3')==l['output_sha256'],'byte-preserving layout')
        cov=np.memmap(layout/'coverage.bin',mode='r',dtype='<u4');require(cov.size==dst.size and np.all(cov==1),'layout coverage')
        require(r['schema']=='neuralstudio.nr.prefix22-live-chain.v1' and r['result']=='pass' and [b['block'] for b in r['blocks']]==list(range(1,23)),'complete encoder')
        require(r['expected_intermediate_opened'] is False and r['live_device_handoffs'] is True and r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'encoder memory/ingress')
        outputs={f'block{b["block"]}.e4m3':b['output_sha256'] for b in r['blocks']}
        for block,filename in ((4,'downsample.json'),(8,'downsample8.json'),(14,'downsample14.json'),(22,'downsample22.json')):
            d=receipt(enc/filename);require(d['pid']==r['pid'] and d['poison']==poison and d['result']=='pass','boundary provenance')
            require(d['expected_input_opened'] is False and d['live_projection_input'] is True and d['coverage_errors']==d['canary_errors']==d['immutable_errors']==0,'boundary gates')
            require(sha(enc/f'block{block}-pooled.e4m3')==d['pooled_sha256'],'pool artifact');outputs[f'block{block}-downsample.e4m3']=d['output_sha256']
        require(outputs==expected,'encoder comparison identity')
        for filename,h in expected.items():require(sha(enc/filename)==h,'actual encoder output '+filename);total+=(enc/filename).stat().st_size
        nr=receipt(enc/'native_exports.json');require(nr['pid']==r['pid'] and nr['poison']==poison and nr['result']=='pass','native export identity')
        require(nr['expected_input_opened'] is False and nr['live_device_inputs'] is True and nr['coverage_errors']==nr['canary_errors']==nr['immutable_errors']==0,'native export gates')
        require(nr['outputs']==native['outputs'],'native output inventory')
        for filename,meta in nr['outputs'].items():require(sha(enc/filename)==meta['sha256'] and (enc/filename).stat().st_size==meta['bytes'],'actual native export');native_total+=meta['bytes']
        for block in range(1,23):
            paths=sorted((enc/f'block{block}').glob('first-window-*'));require(len(paths)==(10 if block<=4 else 11),'stage witness inventory')
            for path in paths:
                key=str(path.relative_to(enc));h=sha(path);require(key not in stage_witnesses or stage_witnesses[key]==h,'stage poison/repeat difference');stage_witnesses[key]=h
        # Command paths are part of provenance: the encoder consumes this run's
        # newly produced layout output, not an equal archived tensor.
        executed=commands[3+ordinal*3:6+ordinal*3]
        require([c['label'] for c in executed]==['block0','entry-layout','encoder'],'stage launch order')
        require(executed[1]['command'][2]==str(pre) and executed[2]['command'][2]==str(layout/'encoder-window.e4m3'),'fresh producer command paths')
        runs.append({'run':name,'block0_pid':b['pid'],'layout_pid':l['pid'],'encoder_pid':r['pid'],
                     'block0_receipt_sha256':sha(pre/'run.json'),'layout_receipt_sha256':sha(layout/'run.json'),'encoder_receipt_sha256':sha(enc/'run.json'),
                     'native_export_receipt_sha256':sha(enc/'native_exports.json'),'native_exports':nr['outputs'],'block0_skip_sha256':b['skip_sha256']})
    require(len(pids)==12 and total==668794880 and native_total==137134080,'full repeated evidence coverage')
    r={'schema':'neuralstudio.nr.live-image-encoder22-verification.v1','plan_root':plan['receipt_root_sha256'],'parent_block0_root':pre_parent['receipt_root_sha256'],
       'parent_encoder_root':parent['receipt_root_sha256'],'runs':runs,'encoder_compared_bytes':total,'native_export_compared_bytes':native_total,
       'block0_output_compared_bytes':353894400,'layout_compared_bytes':70778880,'layout_exhaustive_bijection':True,'fresh_native_processes':12,
       'source_sha256':{str(Path(__file__).relative_to(repo)):sha(Path(__file__))},'result':'pass',
       'claims':{'native_image_to_blocks0_through22':True,'recorded_block0_entry':False,'same_attempt_live_producers':True,
                 'single_resident_process':False,'full_model_inference':False,'optimized':False}}
    if 'candidate_bundle' in plan:r['candidate_bundle']=plan['candidate_bundle']
    r['receipt_root_sha256']=canon(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({'root':r['receipt_root_sha256'],'encoder_bytes':total,'native_exports':native_total,'result':'pass'},indent=2))
if __name__=='__main__':main()
