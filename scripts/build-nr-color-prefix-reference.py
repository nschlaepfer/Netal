#!/usr/bin/env python3
"""Build comparison-only Block0 references; never native runtime inputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
sys.dont_write_bytecode=True

INPUTS=('81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','af31801008df2fbb537f365265271e716b26339b4c5a2fa2069c01c1a949e8b8','d2db0ead99a6307a6504afed2d49867f0421e3665bb397bed8277ae4f2046d06','598bb0d29e73b30a2fb74074a8e582b7231e151920bdb52f71d436a0a3c9654e')
OUTPUTS=('a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','1391787165701ef795a2259432a8672ec3bba779b9c09c7abe602848d8677f29','4e3df0c2d6c15ba20f35bbd2f3374d98d52687bd6345e4fe1191f33aafe16379','153fa0d408035d0f88bf483bedb87ca26e250b8bf6e7927f71c2f799dc03c534')
PORTS={'skip':(388096,70778880),'encoder':(71166976,17694720)}
ENCODER_PORTS={'block0-skip':(388096,70778880),'block4-native':(141945856,17694720),'block8-native':(97709056,8847360),'block14-native':(93285376,4423680),'block22-native':(90827776,2211840),'block22-middle':(86649856,1105920)}
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def digest(data):return hashlib.sha256(data).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def canonical(x):return digest(json.dumps(x,sort_keys=True,separators=(',',':')).encode())
def module(name,file):
    spec=importlib.util.spec_from_file_location(name,Path(__file__).with_name(file));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
def main():
    p=argparse.ArgumentParser();p.add_argument('--captures',nargs='+',required=True,type=Path);p.add_argument('--case',required=True,type=int,choices=range(4));p.add_argument('--output',required=True,type=Path);p.add_argument('--diagnostic-single',action='store_true');p.add_argument('--encoder-boundary',action='store_true');a=p.parse_args()
    selected_ports=ENCODER_PORTS if a.encoder_boundary else PORTS;launches=(2,24) if a.encoder_boundary else (1,2)
    assert len(a.captures)==(1 if a.diagnostic_single else 4),'four fresh captures required unless explicitly diagnostic'
    a.output.mkdir(parents=True,exist_ok=False);parser=module('prefix_delta','build-rtx-oracle-manifest.py');graph=module('prefix_graph','summarize-nr-color-validation.py')
    records=[];ports=[];pids=set();poisons=[]
    for index,capture in enumerate(a.captures):
        r=read(capture/'capture-receipt.json');inv=read(capture/'artifact-inventory.json');files={x['path']:x for x in inv['files']}
        assert r['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and r['rtx_inference_proven'] and r['analysis_runtime'] is None
        assert sha(capture/'artifact-inventory.json')==r['artifact_inventory_sha256'] and sha(capture/'rtx-cuda-trace.jsonl')==r['cuda_trace_sha256']
        assert sha(capture/'input-color.rgba16le')==INPUTS[a.case] and sha(capture/'output.rgba16le')==r['output_sha256']==OUTPUTS[a.case]
        assert graph.original_roots(capture)==('0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61','ae6617321f5dba37b2ba46209e928c7d10ffaecfa8338a225ba450c6a6a4da9f')
        host=(capture/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
        assert sha(capture/'host-output.txt')==files['host-output.txt']['sha256'],'host identity log drift'
        if not a.diagnostic_single:
            assert r['input_arrays_immutable'] and r['snapshot_launch_limit']==max(launches) and r['counts']['snapshots']==2
            assert r.get('snapshot_launch_ids',list(launches))==list(launches)
            pidLines=[x for x in host if x.startswith('runtime.process_id=')];poisonLines=[x for x in host if x.startswith('runtime.output_poison=')]
            assert len(pidLines)==len(poisonLines)==1;pid=int(pidLines[0].split('=')[1]);assert pid not in pids;pids.add(pid);poisons.append(poisonLines[0].split('=')[1])
            for before,after in [('input-color.rgba16le','input-color-after.rgba16le'),('input-depth.r32le','input-depth-after.r32le'),('input-motion.rg16le','input-motion-after.rg16le')]:
                assert sha(capture/before)==files[before]['sha256']==files[after]['sha256']==sha(capture/after)
        state=None;snapshots=[]
        for launch in launches:
            name=f'pool-snapshots/snapshot-{launch:04d}.delta';assert sha(capture/name)==files[name]['sha256']
            state,info=parser.parse_snapshot_delta(capture/name,launch,state);assert len(state)==212519936 and info['allocation_id']=='allocation_003';snapshots.append(info)
        hashes={name:digest(memoryview(state)[start:start+size]) for name,(start,size) in selected_ports.items()};ports.append(hashes)
        if index==0:
            (a.output/'comparison-pool.bin').write_bytes(state)
            for name,(start,size) in selected_ports.items():(a.output/f'rtx-expected-{name}.e4m3').write_bytes(memoryview(state)[start:start+size])
        records.append({'capture':str(capture.resolve()),'capture_receipt_sha256':sha(capture/'capture-receipt.json'),'pool_sha256':digest(state),'ports':hashes,'snapshots':snapshots})
        del state
    equal=all(x==ports[0] for x in ports)
    if not a.diagnostic_single:assert poisons==['a5','a5','5a','5a'] and len(pids)==4
    result={'schema':'neuralstudio.nr.color-prefix-reference.v1','case':a.case,'boundary':'encoder22' if a.encoder_boundary else 'block0','snapshot_launch_ids':list(launches),'input_sha256':INPUTS[a.case],'full_rtx_output_sha256':OUTPUTS[a.case],
      'captures':records,'ports':ports[0],'comparison_pool_sha256':sha(a.output/'comparison-pool.bin'),'fresh_prefix_matrix_proven':not a.diagnostic_single and equal,
      'runtime_input_use_forbidden':True,'result':'diagnostic_reference' if a.diagnostic_single else 'pass' if equal else 'fail',
      'scope':'CPU comparison only, opened after native execution. Declared two RTX boundary snapshots reconstructed and selected output ports compared across captures. Not per-intermediate parity between snapshots. Diagnostic-single mode is never a fresh-prefix acceptance.'}
    result['receipt_root_sha256']=canonical(result);(a.output/'reference.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'result':result['result'],'root':result['receipt_root_sha256'],'ports':ports[0],'pool':result['comparison_pool_sha256']},indent=2))
    if not equal:raise RuntimeError('prefix ports differ; all evidence preserved')
if __name__=='__main__':main()
