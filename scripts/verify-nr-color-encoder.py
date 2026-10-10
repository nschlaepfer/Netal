#!/usr/bin/env python3
"""Read-only audit of fresh color encoder evidence, independent of its driver.

All artifacts are comparison inputs to this CPU verifier, never GPU inputs.
Interior encoder hashes are checked for integrity/repeat, not new-case RTX parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from nr_candidate_primitive import validate_primitive

KEYS=('a5-a','a5-b','5a-a','5a-b')
SIZES={'block0-skip':70778880,'block4-native':17694720,'block8-native':8847360,
       'block14-native':4423680,'block22-native':2211840,'block22-middle':1105920}
def require(ok,label):
    if not ok:raise ValueError(label)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p));return r
def memory(r,poison,pid=None):
    require(r['result']=='pass' and r['poison']==poison,'status/poison')
    require(isinstance(r['pid'],int) and r['pid']>0 and (pid is None or r['pid']==pid),'PID')
    for k in ('coverage_errors','canary_errors','immutable_errors'):require(r[k]==0,k)
def coverage(path,n):
    require(path.stat().st_size==4*n,'coverage size')
    a=np.memmap(path,mode='r',dtype='<u4');require(np.all(a==1),'raw coverage holes or multiple writes')
def verify(attempt):
    attempt=attempt.resolve();repo=Path(__file__).resolve().parents[1]
    plan=sealed(attempt/'plan.json');claimed=sealed(attempt/'verification.json');commands=read(attempt/'commands.json')
    require(plan['schema']=='neuralstudio.nr.color-encoder-plan.v1','plan schema')
    require(claimed['schema']=='neuralstudio.nr.color-encoder-boundary.v1' and claimed['result']=='pass' and not claimed['failures'],'driver status')
    require(claimed['plan_root']==plan['receipt_root_sha256'] and claimed['case']==plan['case'],'plan link')
    require(len(commands)==13 and all(c['exit_code']==0 for c in commands),'compile and twelve successful commands')
    require([r['run'] for r in claimed['runs']]==list(KEYS),'run inventory')
    pre=Path(plan['pre_bundle']);enc=Path(plan['encoder_bundle']);ph=sealed(pre/'case-host.json');em=sealed(enc/'manifest.json')
    require(ph['receipt_root_sha256']==plan['pre_case_host_root'] and em['receipt_root_sha256']==plan['encoder_bundle_root']==ph['parent_bundle_root'],'bundle linkage')
    validate_primitive(enc,em)
    for name,h in em['files'].items():require(sha(enc/name)==h,'encoder source/binary drift')
    require(ph['reference_gpu_input'] is False and ph['shader_changes'] is False,'case host ingress')
    require(sha(pre/'source/source-manifest.json')==ph['source_manifest_sha256'],'pre source manifest')
    for name,h in read(pre/'source/source-manifest.json')['files'].items():require(sha(pre/'source'/name)==h,'pre source drift')
    for path,h in plan['artifacts'].items():require(sha(Path(path))==h,'material artifact drift '+path)
    for name,key in [('entry_probe','layout_probe_sha256'),('entry.metallib','layout_metallib_sha256'),('source/nr_pre/entry_layout_probe.mm','layout_source_sha256')]:require(sha(attempt/name)==plan[key],'layout identity')
    require(sha(repo/'scripts/test-nr-color-encoder.py')==plan['driver_sha256'],'executed driver identity')
    b0=sealed(Path(plan['rtx_block0_reference'])/'reference.json');refdir=Path(plan['rtx_boundary_reference']);rtx=sealed(refdir/'reference.json');rr=sealed(refdir/'retrieval.json');identity=sealed(refdir/'identities/verification.json')
    require(b0['receipt_root_sha256']==plan['rtx_block0_root'] and rtx['receipt_root_sha256']==plan['rtx_encoder_boundary_root']==rr['reference_root']==identity['reference_root'],'RTX reference links')
    require(rr['result']==identity['result']=='pass' and rr['identity_audit_root']==identity['receipt_root_sha256'],'RTX identity gate')
    require(rtx['boundary']=='encoder22' and rtx['fresh_prefix_matrix_proven'] and rtx['case']==b0['case']==plan['case'],'RTX scope')
    require(rtx['input_sha256']==b0['input_sha256']==ph['input_sha256']==plan['input_sha256']==sha(Path(ph['image'])),'case original image identity')
    require(set(rtx['ports'])==set(SIZES) and rtx['ports']['block0-skip']==b0['ports']['skip'],'RTX port inventory')
    pids=set();rows=[];repeat={};boundary_bytes=0;interior_bytes=0
    for ordinal,key in enumerate(KEYS):
        root=attempt/key;poison=int(key[:2],16);b=read(root/'block0/run.json');l=read(root/'entry-layout/run.json');e=read(root/'encoder/run.json')
        memory(b,poison);memory(e,poison)
        require(l['result']=='pass' and l['poison']==poison and l['coverage_errors']==l['canary_errors']==0 and l['input_immutable'] is True,'layout memory')
        for stage,r in [('block0',b),('entry-layout',l),('encoder',e)]:
            require(r['pid'] not in pids,'reused process');pids.add(r['pid'])
            require(sha(root/stage/'run.json')==claimed['runs'][ordinal]['receipt_sha256'][stage],'driver receipt linkage')
        require(b['input_sha256']==ph['input_sha256'] and b['expected_gpu_input'] is False and b['live_device_handoffs'] is True and b['instruction_corrections'] is True,'Block0 live ingress')
        require(b['skip_mismatching_bytes']==b['encoder_mismatching_bytes']==0,'Block0 reference compare')
        for name,field in [('skip.e4m3','skip_sha256'),('encoder.e4m3','encoder_sha256'),('features-shared.f16','feature_sha256')]:require(sha(root/'block0'/name)==b[field],'Block0 actual output')
        require(b['skip_sha256']==b0['ports']['skip'] and b['encoder_sha256']==b0['ports']['encoder'],'Block0 reference hashes')
        coverage(root/'block0/skip.coverage.bin',70778880);coverage(root/'block0/encoder.coverage.bin',17694720);coverage(root/'entry-layout/coverage.bin',17694720)
        require(l['expected_gpu_input'] is False and l['producer_pid']==b['pid'] and l['producer_receipt_sha256']==sha(root/'block0/run.json'),'layout producer')
        require(l['input_sha256']==b['encoder_sha256'] and l['output_sha256']==e['input_sha256']==plan['layout_expected_sha256']==sha(root/'entry-layout/encoder-window.e4m3'),'entry linkage')
        # Independent window-coordinate comparison, not the driver's gathered index array.
        src=np.memmap(root/'block0/encoder.e4m3',mode='r',dtype='u1',shape=(2,576,960,16))
        dst=np.memmap(root/'entry-layout/encoder-window.e4m3',mode='r',dtype='u1',shape=(72,120,64,32))
        for token in range(64):
            tile,q=divmod(token,16);dx=(tile//2)*4+q%4;dy=(tile%2)*4+q//4
            for group in range(2):require(np.array_equal(dst[:,:,token,group*16:(group+1)*16],src[group,dy::8,dx::8,:]),'live layout permutation')
        require(e['schema']=='neuralstudio.nr.prefix22-live-chain.v1' and [v['block'] for v in e['blocks']]==list(range(1,23)),'all encoder blocks')
        require(e['expected_intermediate_opened'] is False and e['live_device_handoffs'] is True,'encoder live ingress')
        tracked={f'block{v["block"]}.e4m3':v['output_sha256'] for v in e['blocks']}
        for block,suffix in [(4,''),(8,'8'),(14,'14'),(22,'22')]:
            d=read(root/f'encoder/downsample{suffix}.json');memory(d,poison,e['pid'])
            require(d['expected_input_opened'] is False and d['live_projection_input'] is True,'live downsample')
            tracked[f'block{block}-pooled.e4m3']=d['pooled_sha256'];tracked[f'block{block}-downsample.e4m3']=d['output_sha256']
        for filename,h in tracked.items():
            path=root/'encoder'/filename;require(sha(path)==h,'encoder stored output integrity')
            require(filename not in repeat or repeat[filename]==h,'encoder poison/repeat invariance');repeat[filename]=h;interior_bytes+=path.stat().st_size
        nr=read(root/'encoder/native_exports.json');memory(nr,poison,e['pid'])
        require(nr['expected_input_opened'] is False and nr['live_device_inputs'] is True,'export ingress')
        require(set(nr['outputs'])=={n+'.e4m3' for n in SIZES if n!='block0-skip'},'export inventory')
        for name,size in SIZES.items():
            path=root/('block0/skip.e4m3' if name=='block0-skip' else 'encoder/'+name+'.e4m3');h=sha(path)
            require(path.stat().st_size==size and h==rtx['ports'][name]==claimed['runs'][ordinal]['ports'][name],'actual RTX boundary '+name)
            if name!='block0-skip':require(nr['outputs'][path.name]=={'bytes':size,'sha256':h},'export metadata')
            boundary_bytes+=size
        executed=commands[1+ordinal*3:4+ordinal*3]
        expected=[['block0',pre/'probe',pre/'pre.metallib',Path(ph['image'])],['layout',attempt/'entry_probe',attempt/'entry.metallib',root/'block0'],['encoder',enc/'encoder/probe',enc/'encoder/prefix.metallib',root/'entry-layout/encoder-window.e4m3']]
        for c,(stage,exe,lib,ingress) in zip(executed,expected):
            require(c['label']==key+'-'+stage,'command order')
            require([Path(v).resolve() for v in c['command'][:3]]==[exe.resolve(),lib.resolve(),ingress.resolve()],'same-attempt live command paths')
        rows.append({'key':key,'producer_directory':str(root),'block0_pid':b['pid'],'layout_pid':l['pid'],'encoder_pid':e['pid'],'native_exports_sha256':sha(root/'encoder/native_exports.json')})
    require(len(pids)==claimed['fresh_processes']==12,'four fresh three-process chains')
    return {'schema':'neuralstudio.nr.color-encoder-independent-audit.v1','result':'pass','attempt':str(attempt),'case':plan['case'],'plan_root':plan['receipt_root_sha256'],'execution_root':claimed['receipt_root_sha256'],'rtx_boundary_root':rtx['receipt_root_sha256'],'boundary_compared_bytes':boundary_bytes,'interior_integrity_repeat_checked_bytes':interior_bytes,'fresh_native_processes':12,'runs':rows,'verifier_sha256':sha(Path(__file__)),'claims':{'fresh_image_to_encoder22':True,'six_boundary_ports_match_rtx':True,'raw_block0_layout_coverage':True,'encoder_coverage_from_native_counters':True,'rtx_interior_encoder_parity':False,'full_new_frame_parity':False,'single_resident_process':False,'local_rtx_snapshot_payload_reaudit':False}}
def main():
    p=argparse.ArgumentParser();p.add_argument('attempt',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    require(not a.output.exists(),'preserve previous audit')
    try:r=verify(a.attempt)
    except Exception as e:r={'schema':'neuralstudio.nr.color-encoder-independent-audit.v1','result':'fail','error':str(e),'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']!='pass':raise SystemExit(3)
if __name__=='__main__':main()
