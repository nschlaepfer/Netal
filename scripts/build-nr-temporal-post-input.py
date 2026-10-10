#!/usr/bin/env python3
"""Extract a comparison-only post155 input from a live two-frame RTX capture."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
PORTS={'block0-skip':(388096,70778880),'block0-encoder':(71166976,17694720),'block4-native':(141945856,17694720),'block8-native':(97709056,8847360),'block14-native':(93285376,4423680),'block22-native':(90827776,2211840),'block22-middle':(86649856,1105920)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--parent-reference',type=Path,required=True);p.add_argument('--parser',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();assert not a.output.exists();a.output.mkdir(parents=True)
    try:
        parent=sealed(a.parent_reference);assert parent['receipt_root_sha256']=='3d1aa05303da317a88ae46353743a6ade67017f4b9ad97d31398283d413ef6f8'
        d=a.job_root/a.label;cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');job=read(a.job_root/(a.label+'-receipt.json'));build=read(a.job_root/'build-receipt.json')
        assert build['source_root']=='1dfae0912b42f835caa09c86aea5a65a917087006e88d3b40bb78e3a02c3e6fa' and build['exit_code']==job['exit_code']==0
        assert sha(d/'capture-receipt.json')==job['capture_receipt_sha256'] and sha(d/'artifact-inventory.json')==cap['artifact_inventory_sha256'] and sha(a.job_root/'build-receipt.json')==job['build_receipt_sha256']
        assert cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None and cap['harness_sha256']==build['files']['color-cases.exe'] and cap['proxy_sha256']==build['files']['nvcuda.dll']
        assert cap['snapshot_launch_ids']==[2,24,158,180,310] and cap['counts']['launches']==312 and cap['counts']['snapshots']==5 and job['resets']==[1,0]
        files={m['path']:m for m in inv['files']};assert len(files)==len(inv['files'])
        for n,m in files.items():
            f=d/n;assert f.resolve().is_relative_to(d.resolve()) and f.stat().st_size==m['bytes'] and sha(f)==m['sha256'],n
        assert [cap['output_sha256'],cap['second_output_sha256']]==parent['outputs']==[job['first_output_sha256'],job['second_output_sha256']]
        for n,h in parent['history_sha256'].items():assert files[n]['sha256']==h,'history observation changed'
        events=[json.loads(s) for s in (d/'rtx-cuda-trace.jsonl').read_text(encoding='utf-8-sig').splitlines()];assert all(e.get('result',0)==0 for e in events)
        assert [e for e in events if e.get('call')=='cuTexObjectCreate']==parent['samplers']
        modules={e['module_id']:files['cuda-dumps/'+e['dump']]['sha256'] for e in events if e.get('call')=='cuModuleLoadData'};functions={e['function_id']:e for e in events if e.get('call')=='cuModuleGetFunction'};launches=[e for e in events if e.get('call')=='cuLaunchKernel'];assert len(launches)==312
        for group in (launches[:156],launches[156:]):
            graph=[{'module_sha256':modules[functions[e['function_id']]['module_id']],'kernel':e['name'],**{k:e[k] for k in ('grid','block','shared_bytes','argument_bytes')}} for e in group]
            assert canonical(graph)==parent['graph_roots'][0]
        spec=importlib.util.spec_from_file_location('delta',a.parser);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);state=None;snaps=[]
        for i,launch in enumerate((2,24,158,180,310)):
            state,info=module.parse_snapshot_delta(d/f'pool-snapshots/snapshot-{launch:04d}.delta',launch,state);assert len(state)==212519936 and info['allocation_id']=='allocation_003';snaps.append(info)
            if i<4:
                for n,h in parent['boundaries'][i]['ports'].items():
                    off,size=PORTS[n];assert hashlib.sha256(memoryview(state)[off:off+size]).hexdigest()==h,'prefix non-interference '+n
        pool=a.output/'post-input-pool.bin';pool.write_bytes(state)
        arguments=(d/'cuda-dumps/arguments-0311.bin').read_bytes();assert len(arguments)==184;allocs=[e for e in events if e.get('call')=='cuMemAlloc'];bindings=[]
        for offset in (0,8,24,104):
            pointer=struct.unpack_from('<Q',arguments,offset)[0];matches=[e for e in allocs if int(e['pointer'],16)<=pointer<int(e['pointer'],16)+e['bytes']];assert len(matches)==1;m=matches[0];bindings.append({'parameter_offset':offset,'allocation_id':m['allocation_id'],'allocation_offset':pointer-int(m['pointer'],16)})
        assert bindings==[{'parameter_offset':0,'allocation_id':3,'allocation_offset':124251136},{'parameter_offset':8,'allocation_id':3,'allocation_offset':388096},{'parameter_offset':24,'allocation_id':1,'allocation_offset':147429888},{'parameter_offset':104,'allocation_id':1,'allocation_offset':147429376}]
        copied={'params-311.bin':d/'cuda-dumps/arguments-0311.bin','params-312.bin':d/'cuda-dumps/arguments-0312.bin','input-color.rgba16le':d/'input-color.rgba16le','motion.rg16le':d/'input-motion.rg16le','history.rgba16le':d/'frame-1-history-before.rgba16le','expected-output.rgba16le':d/'frame-1-output.rgba16le'}
        for n,src in copied.items():shutil.copyfile(src,a.output/n)
        for n,m in files.items():assert sha(d/n)==m['sha256'],'source drift'
        r={'schema':'neuralstudio.nr.temporal-post-input.v1','result':'pass_comparison_reference','parent_reference_root':parent['receipt_root_sha256'],'source_root':build['source_root'],'capture_receipt_sha256':sha(d/'capture-receipt.json'),'inventory_sha256':sha(d/'artifact-inventory.json'),'snapshots':snaps,'post_parameter_bindings':bindings,'outputs':parent['outputs'],'files':{p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in a.output.iterdir() if p.is_file()},'gpu_executed_by_extractor':False,'live_runtime_input_forbidden':True,'scope':'Actual snapshot310 after the second-frame decoder and before post155. Diagnostic component replay/comparison only; never ingest this archived pool into a live whole-model acceptance path.','extractor_sha256':sha(Path(__file__)),'parser_sha256':sha(a.parser)}
    except Exception as e:r={'schema':'neuralstudio.nr.temporal-post-input.v1','result':'fail','error':str(e),'live_runtime_input_forbidden':True}
    r['receipt_root_sha256']=canonical(r);(a.output/'reference.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='snapshots'},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
