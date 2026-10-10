#!/usr/bin/env python3
"""Independent per-case three-frame RTX matrix audit; no native inference."""
import argparse
import hashlib
import json
from pathlib import Path

SOURCE='80560afd03db3771d9fe61bbc58a349426e5a3ab41f6fb7d3f91a05c5b71f63c'
GRAPH='0e7a4f6ba72bde69ec42eea4968ebee4d359f551d2961dd03945f7a7c25a2d61'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def require(v,label):
    if not v:raise ValueError(label)
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'receipt seal');return r
def audit(directory):
    repo=Path(__file__).resolve().parents[1];case=0;cases=[0,2,1]
    original=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json')
    black=sealed(repo/'build/nr_color_encoder_reference/reference-1-v1/reference.json')
    checker=sealed(repo/'build/nr_color_encoder_reference/reference-2-retry-v2/reference.json')
    baseline={'input_sha256':'81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','ports':original['boundary_ports'][1],'full_rtx_output_sha256':original['outputs'][0],'receipt_root_sha256':original['receipt_root_sha256']}
    bases=[baseline,checker,black];inputs=[b['input_sha256'] for b in bases]
    compiled=read(directory/'build-receipt.json');require(compiled['source_root']==SOURCE and compiled['exit_code']==0,'compiled source')
    static=sealed(directory/'static-control100-a5-reference.json');require(static['image_sequence']=='static' and static['color_cases']==[0,0,0] and static['outputs']==original['outputs'] and [b['ports'] for b in static['boundaries']]==original['boundary_ports'],'static control')
    control=sealed(directory/'mixed-control111-a5-reference.json')
    require(control['color_cases']==cases and control['image_sequence']=='mixed' and control['resets']==[1,1,1] and control['outputs']==[b['full_rtx_output_sha256'] for b in bases],'per-image reset controls')
    require(control['all_inventory_payloads_rehashed_on_rig'] and control['source_root']==SOURCE,'control source/payload audit')
    for i,b in enumerate(bases):require(control['boundaries'][2*i+1]['ports']==b['ports'],'per-image reset encoder control')
    rows=[];pids=set();outputs=None;ports=None;samplers=None
    for suffix in ('a5-a','a5-b','5a-a','5a-b'):
        label=f'mixed-reset100-{suffix}';d=directory/label;r=sealed(d/(label+'-reference.json'));job=read(d/(label+'-receipt.json'));cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');files={f['path']:f for f in inv['files']}
        require(len(files)==len(inv['files']),'unique inventory')
        require(r['schema']=='neuralstudio.nr.image-sequence-reference.v1' and r['result']=='pass_bounded_history_observation' and r['color_case']==case and r['label']==label,'case/schema/status')
        require(r['source_root']==SOURCE and r['auditor_sha256']==sha(repo/'scripts/build-nr-image-sequence-reference.py') and r['parser_sha256']==sha(repo/'scripts/build-rtx-oracle-manifest.py'),'auditor provenance')
        require(r['all_inventory_payloads_rehashed_on_rig'] and not r['gpu_executed_by_this_auditor'] and r['runtime_input_use_forbidden'],'observation-only provenance')
        require(job['image_sequence']==r['image_sequence']=='mixed' and r['color_cases']==cases and job['exit_code']==0 and job['color_case']==case and job['resets']==r['resets']==[1,0,0] and job['poison']==r['poison']==suffix[:2],'job settings')
        require(job['build_receipt_sha256']==sha(directory/'build-receipt.json') and job['capture_receipt_sha256']==r['capture_receipt_sha256']==sha(d/'capture-receipt.json'),'build/capture binding')
        require(cap['harness_sha256']==compiled['files']['color-cases.exe'] and cap['proxy_sha256']==compiled['files']['nvcuda.dll'],'actual executables')
        require(cap['artifact_inventory_sha256']==r['inventory_sha256']==sha(d/'artifact-inventory.json'),'inventory binding')
        require(cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e' and cap['analysis_runtime'] is None and cap['rtx_inference_proven'] and cap['input_arrays_immutable'],'unchanged NVIDIA execution')
        require(cap['counts']['launches']==468 and cap['counts']['snapshots']==6 and cap['snapshot_launch_ids']==[2,24,158,180,314,336] and r['graph_roots']==[GRAPH]*3,'same selected graph')
        require(sha(d/'host-output.txt')==files['host-output.txt']['sha256'],'actual host log')
        lines=(d/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
        def values(n):return [s.split('=',1)[1] for s in lines if s.startswith('runtime.'+n+'=')]
        require(values('process_id')==[str(r['pid'])] and r['pid']>0 and r['pid'] not in pids and r['pid']!=control['pid'],'fresh process');pids.add(r['pid'])
        require(values('color_case')==[str(case)] and values('output_poison')==[suffix[:2]] and values('sequence_frames')==['3'] and values('sequence_frame')==['0 reset=1','1 reset=0','2 reset=0'] and values('evaluate_result')==['0x1']*3,'live invocation')
        require(r['input_sha256']==files['input-color.rgba16le']['sha256']==baseline['input_sha256'] and r['input_hashes']==inputs and values('sequence_color_case')==['0:0','1:2','2:1'],'known per-frame input recipes/order')
        for i,prefix in enumerate(('','frame-1-','frame-2-')):
            for before,after in [('input-color.rgba16le','input-color-after.rgba16le'),('input-depth.r32le','input-depth-after.r32le'),('input-motion.rg16le','input-motion-after.rg16le')]:
                source_name=(prefix if before=='input-color.rgba16le' else '')+before
                require(files[source_name]['sha256']==files[prefix+after]['sha256'],'per-frame immutable input')
            require(files[prefix+'input-color.rgba16le']['sha256']==inputs[i],'changed image before evaluation')
            require(files[prefix+'output.rgba16le']['bytes']==16588800 and files[prefix+'output.rgba16le']['sha256']==r['outputs'][i],'output inventory')
            require(r['history_sha256'][prefix+'history-after.rgba16le']==r['outputs'][i],'history equals output')
            if i:require(r['history_sha256'][prefix+'history-before.rgba16le']==r['outputs'][i-1],'history continuity')
        for n,h in r['history_sha256'].items():require(files[n]['bytes']==16588800 and files[n]['sha256']==h,'history readback identity')
        require(r['outputs'][0]==baseline['full_rtx_output_sha256'],'accepted reset output retained')
        current=[]
        require([b['launch'] for b in r['boundaries']]==[2,24,158,180,314,336] and [b['frame'] for b in r['boundaries']]==[0,0,1,1,2,2],'six boundaries')
        for b in r['boundaries']:
            require(b['snapshot']['delta_sha256']==files[f'pool-snapshots/snapshot-{b["launch"]:04d}.delta']['sha256'],'snapshot identity');current.append(b['ports'])
        require(current[0]==control['boundaries'][0]['ports'] and current[1]==baseline['ports'],'frame0 boundaries retained')
        for i in (0,2,4):require(current[i]['block0-skip']==current[i+1]['block0-skip'],'frame boundary consistency')
        if outputs is not None:require(r['outputs']==outputs and current==ports and r['samplers']==samplers,'cross-process/two-poison invariance')
        outputs,ports,samplers=r['outputs'],current,r['samplers'];rows.append({'label':label,'pid':r['pid'],'poison':r['poison'],'reference_root':r['receipt_root_sha256'],'native_capture_sha256':r['capture_receipt_sha256']})
    return {'schema':'neuralstudio.nr.image-sequence-matrix.v1','result':'pass_bounded_RTX_reference','color_cases':cases,'input_hashes':inputs,'source_root':SOURCE,'prior_reset_reference_root':baseline['receipt_root_sha256'],'reset_control_root':control['receipt_root_sha256'],'input_sha256':baseline['input_sha256'],'outputs':outputs,'boundary_ports':ports,'samplers':samplers,'runs':rows,'fresh_processes':4,'evaluations':12,'history_continuity':True,'output_poison_invariance':True,'runtime_input_use_forbidden':True,'mac_parity':False,'full_goal_complete':False,'limits':['Original->checker->black across three calls, zero motion, fixed1920x1080 descriptor.','Raw snapshots/history rehashed and reconstructed on rig; local audit binds evidence inventories.','No native Mac parity, nonzero-motion combination, longer-sequence or performance claim.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--receipts',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve prior audit')
    r=audit(a.receipts.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('boundary_ports','samplers')},indent=2))
if __name__=='__main__':main()
