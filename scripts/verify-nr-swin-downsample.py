#!/usr/bin/env python3
"""Verify the live C32 prefix and its FP16-source downsample boundary."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(data):return hashlib.sha256(data).hexdigest()
def file_sha(path):return sha(path.read_bytes())
def require(ok,message):
    if not ok:raise ValueError(message)
def receipt(path):
    r=json.loads(path.read_text());body=dict(r);root=body.pop('receipt_root_sha256')
    require(sha(json.dumps(body,sort_keys=True,separators=(',',':')).encode())==root,'receipt drift')
    return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    require(not a.output.exists(),'preserve verification')
    repo=Path(__file__).resolve().parents[1];ref=repo/'build/nr_swin_downsample/reference-v1'
    rr=receipt(ref/'receipt.json');prior=receipt(repo/'build/nr_swin_chain/reference-v1/receipt.json')
    require(rr['runtime_sha256']==prior['runtime_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e','wrong model')
    require(rr['graph_root_sha256']==prior['graph_root_sha256']=='3de475044e840d59f0235f0ff12763e99950909f81e5ae8b8c832375a25c88c5','wrong graph')
    expected={};packed={}
    for b in prior['blocks']:
        name=f'block{b["block"]}.e4m3';expected[name]=(repo/'build/nr_swin_chain/reference-v1'/name).read_bytes()
        require(sha(expected[name])==b['logical_output_sha256'],'prefix reference drift')
        packed[name]=b['packed_output_sha256']
    for name,key in [('block4.e4m3','core_logical_sha256'),('block4-downsample.e4m3','downsample_logical_sha256')]:
        expected[name]=(ref/name).read_bytes();require(sha(expected[name])==rr[key],'Block 4 reference drift')
    packed['block4.e4m3']=rr['core_packed_sha256']
    # Invert the comparison layouts and match the original raw pool hashes.
    # This also ensures comparisons did not silently omit a channel or pixel.
    cell_indices=[(r%8)*64+((c%16)//4)*16+(r//8)*4+(c//16)*8+c%4 for r in range(16) for c in range(32)]
    require(sorted(cell_indices)==list(range(512)),'cell layout not bijective')
    for name,digest in packed.items():
        data=expected[name];raw=bytearray(len(data))
        for wy in range(72):
            for wx in range(120):
                for tile in range(4):
                    cell=((wy*2+tile%2)*240+wx*2+tile//2)*512
                    start=((wy*120+wx)*4+tile)*512
                    for offset in range(0,512,4):raw[cell+cell_indices[offset]:cell+cell_indices[offset]+4]=data[start+offset:start+offset+4]
        require(sha(raw)==digest,'inverse packed RTX layout mismatch')
    down=expected['block4-downsample.e4m3'];planar=bytearray(len(down))
    for wy in range(36):
        for wx in range(60):
            for tile in range(4):
                for row in range(16):
                    x=wx*8+(tile//2)*4+row%4;y=wy*8+(tile%2)*4+row//4
                    start=(((wy*60+wx)*4+tile)*16+row)*64
                    for plane in range(4):
                        offset=(plane*288*480+y*480+x)*16
                        planar[offset:offset+16]=down[start+plane*16:start+plane*16+16]
    require(sha(planar)==rr['downsample_planar_sha256'],'inverse downsample planar layout mismatch')
    input_ref=receipt(repo/'build/nr_swin_block/reference-v2/receipt.json')
    input_sha=file_sha(repo/'build/nr_swin_block/reference-v2/input.e4m3')
    require(input_sha==input_ref['flat_input_sha256'],'recorded diagnostic input drift')
    pids=set();runs=[];compared=0;samples={};projected=set();pooled=set()
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            run=a.attempt/f'{poison}-{ordinal}';r=json.loads((run/'run.json').read_text());d=json.loads((run/'downsample.json').read_text())
            require(r['schema']=='neuralstudio.nr.swin-c32-downsample-live-chain.v1','wrong chain')
            require(r['pid'] not in pids and d['pid']==r['pid'],'process identity');pids.add(r['pid'])
            require(r['input_sha256']==input_sha and r['expected_intermediate_opened'] is False and r['live_device_handoffs'] is True,'chain ingress')
            require(d['expected_input_opened'] is False and d['live_projection_input'] is True,'downsample ingress')
            require(d['weight_sha256']==rr['weight_sha256']==file_sha(ref/'weights/block4.bin'),'weight identity')
            require([b['block'] for b in r['blocks']]==[1,2,3,4],'incomplete prefix')
            for q in (r,d):require(q['result']=='pass' and q['poison']==int(poison,16) and q['coverage_errors']==q['canary_errors']==q['immutable_errors']==0,'run guard failure')
            recorded={f'block{b["block"]}.e4m3':b['output_sha256'] for b in r['blocks']};recorded['block4-downsample.e4m3']=d['output_sha256']
            for name,expected_bytes in expected.items():
                data=(run/name).read_bytes();require(data==expected_bytes and sha(data)==recorded[name],'RTX mismatch '+name);compared+=len(data)
            projected.add(d['projected_input_sha256']);pooled.add(d['pooled_sha256'])
            require(file_sha(run/'block4-pooled.e4m3')==d['pooled_sha256'],'pooled artifact drift')
            for b in range(1,5):
                paths=sorted((run/f'block{b}').glob('first-window-*'));require(len(paths)==10,'missing samples')
                for path in paths:
                    key=f'block{b}/{path.name}';digest=file_sha(path);require(key not in samples or samples[key]==digest,'sample process/poison dependence');samples[key]=digest
            runs.append({'pid':r['pid'],'poison':r['poison'],'receipt_sha256':file_sha(run/'run.json'),'downsample_receipt_sha256':file_sha(run/'downsample.json'),'output_sha256':recorded})
    require(len(projected)==len(pooled)==1,'projection/pooling process dependence')
    negative=a.attempt/'negative-quantized-source';nr=json.loads((negative/'run.json').read_text());nd=json.loads((negative/'downsample.json').read_text())
    require(nr['pid'] not in pids and nr['pid']==nd['pid'] and nr['result']==nd['result']=='pass','negative control execution')
    require(nd['projected_input_sha256']==next(iter(projected)),'negative control producer changed')
    for b in range(1,5):require((negative/f'block{b}.e4m3').read_bytes()==expected[f'block{b}.e4m3'],'negative control changed core')
    bad=(negative/'block4-downsample.e4m3').read_bytes();target=expected['block4-downsample.e4m3'];require(len(bad)==len(target),'negative extent')
    mismatches=sum(x!=y for x,y in zip(bad,target))
    require(mismatches>0,'quantized-source negative control unexpectedly passed')
    sources={}
    for path in sorted((a.attempt/'source').rglob('*')):
        if path.is_file() and path.name!='probe.mm':
            name=str(path.relative_to(a.attempt/'source'));digest=file_sha(path);require(file_sha(repo/name)==digest,'source drift');sources[name]=digest
    require(len(sources)==12,'source closure missing')
    result={'schema':'neuralstudio.nr.swin-c32-downsample-verification.v1','reference_root_sha256':rr['receipt_root_sha256'],
            'prefix_reference_root_sha256':prior['receipt_root_sha256'],'input_sha256':input_sha,'fresh_metal_processes':len(pids),
            'full_grid_compared_bytes':compared,'mismatched_bytes':0,'original_packed_outputs_reconstructed':True,
            'write_coverage_exactly_once':True,'canaries_intact':True,'inputs_immutable':True,'runs':runs,
            'projected_input_sha256':next(iter(projected)),'pooled_sha256':next(iter(pooled)),
            'negative_quantized_source':{'pid':nr['pid'],'output_sha256':sha(bad),'mismatched_bytes':mismatches,'same_core_output':True,'result':'rejected'},
            'source_sha256':sources,'generated_host_sha256':file_sha(a.attempt/'source/src/nr_swin/probe.mm'),
            'host_sha256':file_sha(a.attempt/'probe'),'metallib_sha256':file_sha(a.attempt/'baseline.metallib'),
            'negative_metallib_sha256':file_sha(a.attempt/'quantized-source.metallib'),
            'sample_poison_invariance_sha256':samples,
            'claims':{'complete_blocks':[1,2,3,4],'live_device_handoffs':True,'downsample_unquantized_source':True,
                      'isolated_recorded_block0_input':True,'native_block0_producer':False,'full_model_inference':False},'result':'pass'}
    result['receipt_root_sha256']=sha(json.dumps(result,sort_keys=True,separators=(',',':')).encode())
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256']);print(f'compared_bytes={compared} mismatched_bytes=0 negative_mismatched_bytes={mismatches}')


if __name__=='__main__':main()
