#!/usr/bin/env python3
"""Supplement retained mixed-image Mac frames with refreshed RTX input evidence.

Read-only model/artifact audit. Does not replace the separate runtime audit or
claim temporal behavior, new middle checkpoints, or performance acceptance.
"""
import argparse
import hashlib
import json
from pathlib import Path

def require(ok,label):
    if not ok:raise ValueError(label)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'receipt seal '+str(p));return r
PORTS={'entry':('block22-middle',1105920),'block0_skip':('block0-skip',70778880),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720)}
def reference(directory,case):
    ref=sealed(directory/'reference.json');retrieval=sealed(directory/'retrieval.json');identity=sealed(directory/'identities/verification.json')
    require(ref['case']==retrieval['case']==case and ref['result']==retrieval['result']==identity['result']=='pass','reference case/result')
    require(ref['receipt_root_sha256']==retrieval['reference_root']==identity['reference_root'] and retrieval['identity_audit_root']==identity['receipt_root_sha256'],'reference identity linkage')
    require(ref['snapshot_launch_ids']==[2,24] and ref['fresh_prefix_matrix_proven'] and ref['boundary']=='encoder22','RTX prefix boundary')
    require(len(ref['captures'])==len(identity['runs'])==4,'four RTX runs')
    pids=set();rows=[]
    for i,key in enumerate(('a5-a','a5-b','5a-a','5a-b')):
        d=directory/'identities'/key;cap=read(d/'capture-receipt.json');inv=read(d/'artifact-inventory.json');files={x['path']:x for x in inv['files']};host=(d/'host-output.txt').read_text(encoding='utf-8-sig').splitlines()
        require(len(files)==len(inv['files']),'unique inventory paths')
        require(sha(d/'capture-receipt.json')==ref['captures'][i]['capture_receipt_sha256']==identity['runs'][i]['capture_receipt_sha256'],'capture receipt binding')
        require(sha(d/'artifact-inventory.json')==cap['artifact_inventory_sha256'] and sha(d/'host-output.txt')==files['host-output.txt']['sha256']==identity['runs'][i]['host_log_sha256'],'inventory/host linkage')
        def value(name):
            values=[x.split('=',1)[1] for x in host if x.startswith('runtime.'+name+'=')];require(len(values)==1,'unique host field '+name);return values[0]
        pid=int(value('process_id'));require(pid>0 and pid not in pids and pid==identity['runs'][i]['pid'],'fresh RTX process');pids.add(pid)
        require(int(value('color_case'))==case and value('output_poison')==key[:2],'RTX case/poison')
        require(cap['rtx_inference_proven'] is True and cap['input_arrays_immutable'] is True and cap['analysis_runtime'] is None,'actual immutable RTX execution')
        require(cap['runtime_file_sha256']=='e16bcf15e16e13f527491cdf7845b2fe6521a738d8f7c9c721866a8496e1fc8e','pinned NR DLL')
        for name,n in {'launches':156,'modules':9,'selected_kernels':96,'weight_transfers':153,'snapshots':2}.items():require(cap['counts'][name]==n,'selected graph '+name)
        for before,after in [('input-color.rgba16le','input-color-after.rgba16le'),('input-depth.r32le','input-depth-after.r32le'),('input-motion.rg16le','input-motion-after.rg16le')]:
            require(files[before]['sha256']==files[after]['sha256'],'input before/after inventory '+before)
        require(files['input-color.rgba16le']['sha256']==ref['input_sha256'],'input hash')
        require(files['output.rgba16le']['sha256']==cap['output_sha256']==ref['full_rtx_output_sha256'],'RTX output identity')
        require(ref['captures'][i]['ports']==ref['ports'],'RTX port repeat')
        for launch,snap in zip((2,24),ref['captures'][i]['snapshots']):require(snap['delta_sha256']==files[f'pool-snapshots/snapshot-{launch:04d}.delta']['sha256'],'snapshot seal')
        rows.append({'pid':pid,'poison':key[:2],'capture_sha256':sha(d/'capture-receipt.json')})
    return ref,rows
def audit(matrix,black,checker):
    summary=sealed(matrix/'summary.json');require(summary['schema']=='neuralstudio.nr.full-frame-inmemory-bwc.v1' and summary['result']=='pass','retained v4 BWC matrix')
    refs={};rtx=[]
    for case,d in ((1,black),(2,checker)):
        r,rows=reference(d,case);refs[r['input_sha256']]=r;rtx.extend(rows)
    require(len({r['pid'] for r in rtx})==8,'eight distinct RTX processes')
    require(len(summary['processes'])==4,'four Mac processes');pids=set();rows=[];seen={};observed={};compared=0
    for process in summary['processes']:
        directory=matrix/process['process'];r=read(directory/'receipt.json');observed[str(directory/'receipt.json')]=sha(directory/'receipt.json')
        require(r['result']=='pass' and r['pid']==process['pid'] and r['pid'] not in pids,'Mac process identity');pids.add(r['pid'])
        require(r['frames_in_process']==len(r['frames'])==len(process['frames'])==4,'four mixed frames per process')
        for f,claimed in zip(r['frames'],process['frames']):
            image=f['input_sha256']
            if image not in refs:continue
            ref=refs[image];e=f['encoder'];b=f['body'];require(sha(Path(f['input_file']))==image==e['input_sha256']==claimed['input_sha256'],'original live image')
            require(f['poison'] in (90,165) and f['poison']==claimed['poison'],'Mac poison')
            require(f['result']=='pass' and e['code']==b['code']==0 and e['input_immutable'] and e['gpu_complete'],'native execution')
            require(e['guarded_outputs_checked']==1039 and e['coverage_values_checked']==5900107776,'encoder coverage')
            require(b['checks_enforced']==241 and not b['check_failures'] and b['constants_unchanged_after_frame'] and b['constants_rehashed_after_frame']==153,'body checks')
            require(b['input_texture_sha256_before']==b['input_texture_sha256_after']==image and b['input_texture_immutable'],'body input immutable')
            require(f['producer_regions_unchanged_by_body'] and set(f['regions'])==set(PORTS),'producer region inventory')
            for role,(name,n) in PORTS.items():
                region=f['regions'][role];require(region['offset']==256 and region['bytes']==n and region['sha256']==ref['ports'][name],'RTX live boundary '+role)
            actual=directory/f'final-{f["frame"]}.rgba16le';expected=Path(f['reference']['file']);require(actual.stat().st_size==expected.stat().st_size==16588800,'frame extent')
            require(sha(actual)==sha(expected)==ref['full_rtx_output_sha256']==b['final_surface_sha256']==f['surface_sha256_caller_readback']==claimed['final_sha256'],'retained final RTX equality')
            require(f['reference']['cpu_comparison_only'] and f['reference']['mismatching_values']==0,'CPU comparison role')
            require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'final write coverage')
            seen.setdefault(image,[]).append(f['poison']);compared+=actual.stat().st_size
            rows.append({'mac_pid':r['pid'],'frame':f['frame'],'case':ref['case'],'poison':f['poison'],'reference_root':ref['receipt_root_sha256'],'final_sha256':sha(actual),'matched_boundary_ports':6})
    require(len(rows)==8 and all(sorted(v)==[90,90,165,165] for v in seen.values()),'both cases four-process dual-poison matrix')
    for p,h in observed.items():require(sha(Path(p))==h,'receipt changed during audit')
    return {'schema':'neuralstudio.nr.reset-reference-refresh-audit.v1','result':'pass_boundary_and_final_supplement','matrix_root':summary['receipt_root_sha256'],'rtx_runs':rtx,'mac_frames':rows,'retained_final_bytes_hashed':compared,'boundary_hashes_matched':48,'observed_mac_receipt_hashes':observed,'new_gpu_execution':False,'full_goal_complete':False,'limits':['RTX raw pool/input-after payloads remain on the rig; verified inventory and reconstruction receipts were retrieved.','Live Mac region hashes are recorded native evidence, not newly retained raw region files.','This supplements the pending black/checker RTX input-after reference gate; it does not replace executable/runtime provenance audit.','Reset-only fixed descriptors; not temporal, new middle-stage parity, or performance acceptance.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--matrix',required=True,type=Path);p.add_argument('--black',required=True,type=Path);p.add_argument('--checker',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve previous audit')
    try:r=audit(a.matrix.resolve(),a.black.resolve(),a.checker.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.reset-reference-refresh-audit.v1','result':'fail','error':str(e),'full_goal_complete':False}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
