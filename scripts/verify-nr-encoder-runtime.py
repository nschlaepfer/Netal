#!/usr/bin/env python3
"""Independent read-only check of the two-image, fresh-process runtime matrix."""
import argparse
import hashlib
import json
from pathlib import Path

def require(ok,why):
    if not ok:raise ValueError(why)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p));return r
def audit(attempt):
    repo=Path(__file__).resolve().parents[1];attempt=attempt.resolve();plan=sealed(attempt/'plan.json');matrix=sealed(attempt/'verification.json');build_dir=Path(plan['build_dir']);build=sealed(build_dir/'build.json');commands=read(attempt/'commands.json')
    require(matrix['result']=='pass' and matrix['plan_root']==plan['receipt_root_sha256'] and build['receipt_root_sha256']==plan['build_root'] and build['compile_exit']==0,'build/plan/result linkage')
    require(sha(build_dir/'nr_encoder_runtime.o')==build['object_sha256'],'object identity')
    for name,h in build['source_files'].items():require(sha(build_dir/'source'/name)==h,'compiled source '+name)
    for name,h in build['assets'].items():require(sha(build_dir/'assets'/name)==h,'runtime constant '+name)
    for path,h in plan['artifacts'].items():require(sha(Path(path))==h,'pinned test input '+path)
    original=sealed(repo/'build/nr_encoder_fscale/live-v1/verification.json');dense=sealed(repo/'build/nr_color_encoder_reference/reference-3-v1/reference.json')
    require(original['receipt_root_sha256']==plan['original_producer_root'],'original comparison parent')
    old=original['runs'][0];targets0={Path(n).stem:m['sha256'] for n,m in old['native_exports'].items()};targets0['block0-skip']=old['block0_skip_sha256']
    targets={'81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342':targets0,dense['input_sha256']:dense['ports']}
    sizes={'block0-skip':70778880,'block22-middle':1105920,'block22-native':2211840,'block14-native':4423680,'block8-native':8847360,'block4-native':17694720}
    keys=('a5-forward','a5-reverse','5a-forward','5a-reverse');require([c['key'] for c in commands]==list(keys) and all(c['exit_code']==0 for c in commands),'fresh native command matrix')
    require([r['key'] for r in matrix['runs']]==list(keys),'matrix inventory')
    pids=set();seen={};rows=[];compared=0
    for ordinal,key in enumerate(keys):
        c=commands[ordinal];r=read(attempt/key/'run.json');claimed=matrix['runs'][ordinal]
        require(r['schema']=='neuralstudio.nr.encoder-runtime-characterization.v1' and r['result']=='pass' and r['context_reused'] is True and r['activation_file_ingress'] is False,'native API result')
        require(r['pid']==claimed['pid'] and r['pid'] not in pids and r['pid']>0,'fresh process');pids.add(r['pid'])
        require(sha(attempt/key/'run.json')==claimed['receipt_sha256'] and r['frames']==claimed['frames'],'matrix receipt binding')
        require(len(r['negative_controls'])==6 and [v['test'] for v in r['negative_controls']]==list(range(6)) and all(v['code']==2 and v['error'] for v in r['negative_controls']),'fail-closed input tests')
        require(len(r['frames'])==2 and len({v['input_sha256'] for v in r['frames']})==2,'two different images per context')
        require(Path(c['command'][0]).resolve()==build_dir/'probe' and Path(c['command'][1]).resolve()==build_dir/'assets' and c['command'][6]==key[:2] and Path(c['command'][7]).resolve()==attempt/key,'executed API harness')
        for i,frame in enumerate(r['frames']):
            image=Path(c['command'][2+i*2]);reference=Path(c['command'][3+i*2]);image_sha=sha(image)
            require(frame['frame']==i and frame['input_sha256']==image_sha and image_sha in targets,'actual image identity')
            poison=int(key[:2],16)^(255 if i else 0);require(frame['poison']==poison,'alternate output poison')
            require(frame['mismatching_bytes']==0 and frame['expected_gpu_input'] is False and frame['input_immutable'] is True and frame['gpu_complete'] is True and frame['prior_outputs_still_intact'] is True,'runtime correctness/lifetime gates')
            require(frame['bundle_root']==build['parent_root'],'selected parent model')
            require(frame['guarded_outputs_checked']==1039 and frame['coverage_values_checked']==5900107776,'complete checked dispatch count')
            require(set(frame['ports'])==set(sizes),'six output ports')
            for name,n in sizes.items():
                m=frame['ports'][name];actual=attempt/key/f'frame{i}'/(name+'.e4m3');ref=reference/('block0/skip.e4m3' if name=='block0-skip' else 'encoder/'+name+'.e4m3')
                require(m['bytes']==actual.stat().st_size==ref.stat().st_size==n and m['offset']==256 and m['mismatching_bytes']==0,'output extent/region')
                h=sha(actual);require(h==m['sha256']==sha(ref)==targets[image_sha][name],'actual output vs accepted RTX-linked target')
                compared+=n
            seen.setdefault(image_sha,[]).append((r['pid'],poison))
        rows.append({'key':key,'pid':r['pid'],'receipt_sha256':sha(attempt/key/'run.json'),'distinct_image_inputs':[v['input_sha256'] for v in r['frames']]})
    require(len(pids)==4 and all(len(v)==4 and sorted(p for _,p in v)==[90,90,165,165] for v in seen.values()),'each image four-process dual-poison matrix')
    source=(build_dir/'source/generated/pre.inc').read_text()+(build_dir/'source/generated/encoder.inc').read_text()
    require('Read(' not in source and 'argv' not in source,'no activation filesystem entry in extracted stages')
    return {'schema':'neuralstudio.nr.encoder-runtime-independent-audit.v1','result':'pass','build_root':build['receipt_root_sha256'],'matrix_root':matrix['receipt_root_sha256'],'runs':rows,'actual_boundary_bytes_checked':compared,'fresh_processes':4,'live_image_frames':8,'claims':{'same_process_block0_layout_encoder22':True,'two_distinct_images_per_context':True,'retained_output_lifetime_checked':True,'raw_coverage_retained':False,'native_coverage_checks':True,'fully_connected_model':False,'temporal_history':False,'optimized':False},'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve previous audit')
    try:r=audit(a.attempt)
    except Exception as e:r={'schema':'neuralstudio.nr.encoder-runtime-independent-audit.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']!='pass':raise SystemExit(3)
if __name__=='__main__':main()
