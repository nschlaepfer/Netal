#!/usr/bin/env python3
"""CPU-only audit/reconstruction of bounded same-instance history captures."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
PORTS={'block0-skip':(388096,70778880),'block4-native':(141945856,17694720),'block8-native':(97709056,8847360),'block14-native':(93285376,4423680),'block22-native':(90827776,2211840),'block22-middle':(86649856,1105920)}
def build(job,label,parser_path):
    d=job/label;cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');run=read(job/(label+'-receipt.json'));compiled=read(job/'build-receipt.json')
    assert compiled['source_root']=='80560afd03db3771d9fe61bbc58a349426e5a3ab41f6fb7d3f91a05c5b71f63c' and compiled['exit_code']==0
    assert sha(job/'build-receipt.json')==run['build_receipt_sha256'] and sha(d/'capture-receipt.json')==run['capture_receipt_sha256'] and sha(d/'artifact-inventory.json')==cap['artifact_inventory_sha256']
    assert cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None and cap['input_arrays_immutable'] and cap['rtx_inference_proven']
    assert cap['harness_sha256']==compiled['files']['color-cases.exe'] and cap['proxy_sha256']==compiled['files']['nvcuda.dll']
    assert cap['counts']['launches']==468 and cap['counts']['snapshots']==6 and cap['snapshot_launch_ids']==[2,24,158,180,314,336]
    files={f['path']:f for f in inv['files']};assert len(files)==len(inv['files'])
    # Verify all selected module, weight, parameter and I/O payloads on the rig.
    for n,m in files.items():
        path=d/n;assert path.resolve().is_relative_to(d.resolve());assert path.stat().st_size==m['bytes'] and sha(path)==m['sha256'],n
    host=(d/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
    def values(name):return [x.split('=',1)[1] for x in host if x.startswith('runtime.'+name+'=')]
    assert len(values('process_id'))==1 and values('sequence_frames')==['3'] and values('evaluate_result')==['0x1','0x1','0x1'] and values('output_poison')==[run['poison']]
    assert values('sequence_frame')==['0 reset=1','1 reset='+str(run['resets'][1]),'2 reset='+str(run['resets'][2])]
    for prefix in ('','frame-1-','frame-2-'):
        for name in ('color.rgba16le','depth.r32le','motion.rg16le'):
            stem,ext=name.split('.',1);before=(prefix if stem=='color' else '')+'input-'+name;assert files[before]['sha256']==files[prefix+'input-'+stem+'-after.'+ext]['sha256']
    outputs=[files[p+'output.rgba16le']['sha256'] for p in ('','frame-1-','frame-2-')]
    case=run['color_case'];assert case in (0,1,2)
    assert values('color_case')==[str(case)]
    input_pins=('81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','af31801008df2fbb537f365265271e716b26339b4c5a2fa2069c01c1a949e8b8','d2db0ead99a6307a6504afed2d49867f0421e3665bb397bed8277ae4f2046d06')
    first_pins=('a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','1391787165701ef795a2259432a8672ec3bba779b9c09c7abe602848d8677f29','4e3df0c2d6c15ba20f35bbd2f3374d98d52687bd6345e4fe1191f33aafe16379')
    assert files['input-color.rgba16le']['sha256']==input_pins[case] and outputs[0]==first_pins[case]
    sequence=run['image_sequence'];assert sequence in ('static','mixed')
    cases=[case]*3 if sequence=='static' else [0,2,1];assert sequence!='mixed' or case==0
    assert values('sequence_color_case')==[str(i)+':'+str(c) for i,c in enumerate(cases)]
    inputs=[files[p+'input-color.rgba16le']['sha256'] for p in ('','frame-1-','frame-2-')]
    assert inputs==[input_pins[c] for c in cases]
    for i in (1,2):
        if run['resets'][i]:assert outputs[i]==first_pins[cases[i]],'per-image reset control changed'
    if sequence=='static' and case==0 and not run['resets'][1]:assert outputs[1]=='d48bf48553e61c9e76d23b008dc1192862a11894e1465a6366bf1476473c9ac0','static control changed'
    history={n:files[n]['sha256'] for n in ('history-before.rgba16le','history-after.rgba16le','frame-1-history-before.rgba16le','frame-1-history-after.rgba16le','frame-2-history-before.rgba16le','frame-2-history-after.rgba16le')}
    assert history['history-after.rgba16le']==history['frame-1-history-before.rgba16le']==outputs[0]
    assert history['frame-1-history-after.rgba16le']==history['frame-2-history-before.rgba16le']==outputs[1]
    assert history['frame-2-history-after.rgba16le']==outputs[2]

    events=[json.loads(s) for s in (d/'rtx-cuda-trace.jsonl').read_text(encoding='utf-8-sig').splitlines()];assert all(e.get('result',0)==0 for e in events)
    hs=[e for e in events if e.get('call')=='history_readback'];assert [(e['frame'],e['after'],e['array_id'],e['bytes']) for e in hs]==[(f,a,1,16588800) for f in (0,1,2) for a in (0,1)]
    samplers=[e for e in events if e.get('call')=='cuTexObjectCreate'];assert len(samplers)==6
    assert all(s['address_modes']==[1,1,1] and s['flags']==2 and s['filter_mode']==(0 if s['object']==2 else 1) for s in samplers)
    modules={e['module_id']:files['cuda-dumps/'+e['dump']]['sha256'] for e in events if e.get('call')=='cuModuleLoadData'};functions={e['function_id']:e for e in events if e.get('call')=='cuModuleGetFunction'}
    launches=[e for e in events if e.get('call')=='cuLaunchKernel'];graphs=[]
    for group in (launches[:156],launches[156:312],launches[312:]):
        g=[]
        for e in group:
            f=functions[e['function_id']];assert f['name']==e['name'];g.append({'module_sha256':modules[f['module_id']],'kernel':e['name'],**{k:e[k] for k in ('grid','block','shared_bytes','argument_bytes')}})
        graphs.append(canonical(g))
    assert graphs==['0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61']*3
    weights=[(n,m['bytes'],m['sha256']) for n,m in sorted(files.items()) if n.startswith('cuda-dumps/transfer-')];assert len(weights)==153 and canonical(weights)=='ae6617321f5dba37b2ba46209e928c7d10ffaecfa8338a225ba450c6a6a4da9f'
    spec=importlib.util.spec_from_file_location('delta',parser_path);parser=importlib.util.module_from_spec(spec);spec.loader.exec_module(parser)
    state=None;boundaries=[]
    for launch in (2,24,158,180,314,336):
        state,info=parser.parse_snapshot_delta(d/f'pool-snapshots/snapshot-{launch:04d}.delta',launch,state);assert len(state)==212519936 and info['allocation_id']=='allocation_003'
        ports=PORTS if launch in (24,180,336) else {'block0-skip':PORTS['block0-skip'],'block0-encoder':(71166976,17694720)}
        hashes={n:hashlib.sha256(memoryview(state)[o:o+b]).hexdigest() for n,(o,b) in ports.items()}
        boundaries.append({'launch':launch,'frame':(launch-1)//156,'snapshot':info,'ports':hashes})
    for n,m in files.items():assert sha(d/n)==m['sha256'],'artifact changed during CPU audit'
    return {'schema':'neuralstudio.nr.image-sequence-reference.v1','result':'pass_bounded_history_observation','label':label,'color_case':case,'image_sequence':sequence,'color_cases':cases,'input_sha256':input_pins[case],'input_hashes':inputs,'pid':int(values('process_id')[0]),'resets':run['resets'],'poison':run['poison'],'source_root':compiled['source_root'],'capture_receipt_sha256':sha(d/'capture-receipt.json'),'inventory_sha256':sha(d/'artifact-inventory.json'),'outputs':outputs,'history_sha256':history,'samplers':samplers,'boundaries':boundaries,'graph_roots':graphs,'all_inventory_payloads_rehashed_on_rig':True,'gpu_executed_by_this_auditor':False,'runtime_input_use_forbidden':True,'mac_temporal_parity':False,'limits':['Single three-call observation; repeated-process/poison promotion is separate.','Raw history/pool payloads remain on the rig until requested.','Comparison-only boundary hashes; never supply expected tensors to live inference.'],'auditor_sha256':sha(Path(__file__)),'parser_sha256':sha(parser_path)}
def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);p.add_argument('--label',required=True);p.add_argument('--parser',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    assert not a.output.exists() and re.fullmatch('[a-z0-9-]+',a.label)
    try:r=build(a.job_root.resolve(),a.label,a.parser.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.image-sequence-reference.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('boundaries','samplers')},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
