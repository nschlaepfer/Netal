#!/usr/bin/env python3
"""Independent retained-frame audit, explicitly not strict model promotion.

Does not import Fable's acceptance decision or execute model code. In particular,
byte parity cannot supply missing execution-time binary/parameter provenance.
"""
import argparse
import hashlib
import json
from pathlib import Path

KEYS=('a5-a','a5-b','5a-a','5a-b')
PORTS=[('encoder/block22-native.e4m3',90827776,2211840),('encoder/block14-native.e4m3',93285376,4423680),('encoder/block8-native.e4m3',97709056,8847360),('encoder/block4-native.e4m3',141945856,17694720),('block0/skip.e4m3',388096,70778880)]
def check(ok,why):
    if not ok:raise ValueError(why)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');check(canonical(b)==h,'seal mismatch: '+str(p));return r
def same_bytes(a,b):
    check(a.stat().st_size==b.stat().st_size==16588800,'frame extent')
    with a.open('rb') as x,b.open('rb') as y:
        while True:
            data=x.read(1048576)
            if not data:break
            check(data==y.read(len(data)),'actual frame bytes differ')
def audit(body,producer,case):
    body=body.resolve();producer=producer.resolve();case=case.resolve()
    summary=sealed(body/'summary.json');native=sealed(producer/'verification.json');pa=sealed(producer/'independent-audit-v1.json');plan=sealed(producer/'plan.json')
    check(summary['producer_execution_root']==native['receipt_root_sha256']==pa['execution_root'],'producer execution binding')
    check(summary['producer_independent_audit_root']==pa['receipt_root_sha256'] and pa['result']=='pass','producer audit binding')
    check(native['plan_root']==plan['receipt_root_sha256'] and native['result']=='pass','native plan')
    rtx=sealed(Path(plan['rtx_boundary_reference'])/'reference.json');input_sha=sha(case/'input-color.rgba16le')
    check(input_sha==rtx['input_sha256']==plan['input_sha256'],'original image')
    check(summary['case3_rtx_target']==rtx['full_rtx_output_sha256'],'RTX target')
    check([r['key'] for r in summary['runs']]==list(KEYS),'summary key inventory')
    pids=set();rows=[]
    for i,key in enumerate(KEYS):
        d=body/key;p=producer/key;gate=read(d/'case-validate.json');raw=read(d/'run/chain-run.json');chain=read(d/'chain.json');resident=sealed(d/'resident.json')
        check(sha(d/'case-validate.json')==summary['runs'][i]['case_validate_sha256'],'gate receipt binding')
        check(resident['receipt_root_sha256']==summary['runs'][i]['receipt_root'],'resident seal binding')
        check(raw['pid'] not in pids and isinstance(raw['pid'],int) and raw['pid']>0,'fresh native process');pids.add(raw['pid'])
        check(raw['pid'] not in {v[k] for v in native['runs'] for k in ('block0_pid','layout_pid','encoder_pid')},'body/producer process collision')
        check(raw['poison']==int(key[:2],16),'output poison')
        check(gate['key']==key and Path(gate['encoder_run']).resolve()==p and Path(gate['case_dir']).resolve()==case,'case and producer paths')
        check(Path(chain['entry_file']).resolve()==p/'encoder/block22-middle.e4m3','fresh encoder entry path')
        check(Path(chain['decoder']['tex_in']).resolve()==case/'input-color.rgba16le','original texture path')
        check(raw['decoder']['texture_in_sha256']==input_sha,'actual uploaded texture')
        patches=chain['decoder']['patches'];check(len(patches)==5,'skip inventory')
        for patch,(rel,off,size) in zip(patches,PORTS):
            check(Path(patch['file']).resolve()==p/rel and patch['offset']==off and patch['bytes']==size,'skip path/layout')
            name='block0-skip' if rel.startswith('block0/') else Path(rel).stem
            check((p/rel).stat().st_size==size and sha(p/rel)==native['runs'][i]['ports'][name],'producer skip bytes')
        check(sha(p/'encoder/block22-middle.e4m3')==native['runs'][i]['ports']['block22-middle'],'producer entry bytes')
        launches=raw['launches'];dec=[v for v in launches if v['kind']=='ptx']
        check([v['launch'] for v in dec]==list(range(133,157)),'complete decoder/output')
        for v in dec:check(v['weights_immutable'] is True and v['params_immutable'] is True,'native constant immutability')
        for v in launches:
            if 'coverage' in v:
                for role,c in v['coverage'].items():check(len(c)==3 and c[0]>0 and c[1]==0 and (c[2]==0 or role=='out_vit'),'native middle coverage')
            if 'coverage_dwords' in v:check(v['coverage_dwords'][0]>0 and v['coverage_dwords'][1:]==[0,0],'repack coverage')
        post=next(v for v in dec if v['launch']==155)['texture'];fin=dec[-1]['texture']
        check(post['input_after_sha256']==input_sha,'post input texture immutability')
        for tex in (post,fin):check(tex['pixels']==tex['pixels_once']==2073600 and tex['pixels_multi']==0,'native texture write coverage')
        actual=d/'run/tex_out_156.bin';expected=case/key/'output.rgba16le';same_bytes(actual,expected);h=sha(actual)
        check(h==sha(expected)==rtx['full_rtx_output_sha256']==fin['output_sha256']==gate['final_surface_sha256'],'actual final output identity')
        check(gate['result']=='pass' and not gate['gate_reasons'] and not gate['body_failures_retained'],'peer scoped gate status')
        rows.append({'key':key,'body_pid':raw['pid'],'poison':raw['poison'],'final_sha256':h,'actual_compared_bytes':actual.stat().st_size,'raw_receipt_sha256':sha(d/'run/chain-run.json'),'chain_sha256':sha(d/'chain.json'),'peer_gate_sha256':sha(d/'case-validate.json'),'peer_runner_returncode':gate['runner_returncode'],'peer_exempt_comparisons':len(gate['body_failures_exempt_old_frame'])})
    return {'schema':'neuralstudio.nr.color-body-observation-audit.v1','result':'observed_final_parity','case':plan['case'],'producer_audit_root':pa['receipt_root_sha256'],'body_summary_root':summary['receipt_root_sha256'],'runs':rows,'actual_frame_bytes_compared':4*16588800,'fresh_body_processes':len(pids),'strict_promotion':False,'full_model_goal_complete':False,'limitations':['Production decoder omits per-launch pool write scans; texture coverage is native counter evidence, not retained raw texture counters.','This audit does not establish pre/post execution-time body executable, selected library and model-parameter manifest binding.','Fresh files connect producer and body; not a single resident full model.','New-case middle/decoder intermediates, temporal sequences, other inputs/descriptors, integration and performance remain separate gates.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--body',required=True,type=Path);p.add_argument('--producer',required=True,type=Path);p.add_argument('--case-dir',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();check(not a.output.exists(),'preserve audit')
    try:r=audit(a.body,a.producer,a.case_dir)
    except Exception as e:r={'schema':'neuralstudio.nr.color-body-observation-audit.v1','result':'fail','error':str(e),'strict_promotion':False}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
