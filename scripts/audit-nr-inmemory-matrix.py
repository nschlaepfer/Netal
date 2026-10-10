#!/usr/bin/env python3
"""Audit retained full-frame parity without promoting unproven runtime checks."""
import argparse
import hashlib
import json
from pathlib import Path

def require(ok,label):
    if not ok:raise ValueError(label)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p));return r
def identical(a,b):
    require(a.stat().st_size==b.stat().st_size==16588800,'frame extent')
    with a.open('rb') as x,b.open('rb') as y:
        while True:
            v=x.read(1048576)
            if not v:break
            require(v==y.read(len(v)),'actual frame bytes differ')
def audit(peer,matrix):
    peer=peer.resolve();matrix=matrix.resolve();repo=Path(__file__).resolve().parents[1]
    summary=sealed(matrix/'summary.json');require(summary['schema']=='neuralstudio.nr.full-frame-inmemory-matrix.v1' and summary['result']=='pass','matrix schema/status')
    require(summary['harness_build']=='build/nr_full/v1b','this observation audit is scoped to frozen v1b; review new runtime versions separately')
    build=peer/summary['harness_build'];pins={}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,name=line.split(None,1);p=peer/name.strip();require(sha(p)==h,'frozen harness artifact '+name);pins[str(p)]=h
    producer=repo/'build/nr_encoder_runtime/build-v1';pb=sealed(producer/'build.json')
    require(sha(build/'source/encoder/nr_encoder_runtime.o')==pb['object_sha256']==summary['encoder_object_sha256'],'linked tested producer object')
    require(sha(build/'source/encoder/nr_encoder_runtime.h')==sha(producer/'source/nr_encoder_runtime.h'),'linked producer ABI')
    old=sealed(repo/'build/nr_encoder_fscale/live-v1/verification.json')['runs'][0];new=sealed(repo/'build/nr_color_encoder_reference/reference-3-v1/reference.json')
    orig={Path(n).stem:v['sha256'] for n,v in old['native_exports'].items()};orig['block0-skip']=old['block0_skip_sha256']
    data={'original':{'input':'81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','target':'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','ports':orig,'case':repo/'build/nr_color_cases/rtx-v1/case0'},
          'dense':{'input':new['input_sha256'],'target':new['full_rtx_output_sha256'],'ports':new['ports'],'case':repo/'build/nr_color_cases/rtx-v1/case3-retry-v1'}}
    keys={'dense-a5-1','dense-a5-2','dense-5a-1','dense-5a-2','original-a5-1','original-5a-1'}
    require(len(summary['runs'])==6 and {r['run'] for r in summary['runs']}==keys,'six-run inventory')
    roles={'entry':('block22-middle',1105920),'block0_skip':('block0-skip',70778880),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720)}
    pids=set();rows=[]
    for row in summary['runs']:
        key=row['run'];case,poison,_=key.split('-');target=data[case];r=read(matrix/key/'receipt.json');b=r['body'];e=r['encoder']
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v1' and r['result']=='pass','native harness result')
        require(r['pid']==row['pid'] and r['pid'] not in pids and r['pid']>0,'six fresh PIDs');pids.add(r['pid'])
        require(r['poison']==row['poison']==int(poison,16),'poison binding')
        require(r['input_sha256']==e['input_sha256']==b['input_texture_sha256_before']==b['input_texture_sha256_after']==target['input']==sha(target['case']/'input-color.rgba16le'),'input identity/immutability')
        require(e['code']==b['code']==0 and e['input_immutable'] is True and e['gpu_complete'] is True and b['input_texture_immutable'] is True,'runtime status')
        require(e['bundle_root']==pb['parent_root'] and e['coverage_values_checked']==5900107776 and e['guarded_outputs_checked']==1039,'complete producer checks')
        require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'final texture coverage report')
        require(r['producer_regions_unchanged_by_body'] is True and set(r['regions'])==set(roles),'live input region integrity')
        for role,(name,n) in roles.items():
            reg=r['regions'][role];require(reg['bytes']==n and reg['offset']==256 and reg['sha256']==target['ports'][name]==row['regions'][role],'same live region contract')
        final=matrix/key/'final.rgba16le';expected=target['case']/'a5-a/output.rgba16le';identical(final,expected);h=sha(final)
        require(h==target['target']==r['surface_sha256_caller_readback']==b['final_surface_sha256']==r['reference']['sha256']==row['final_sha256'],'actual retained final identity')
        require(r['reference']['mismatching_values']==0 and r['reference']['cpu_comparison_only'] is True,'RTX comparison report')
        rows.append({'key':key,'pid':r['pid'],'poison':r['poison'],'frame_sha256':h,'receipt_sha256':sha(matrix/key/'receipt.json'),'actual_compared_bytes':final.stat().st_size})
    return {'schema':'neuralstudio.nr.inmemory-matrix-observation-audit.v1','result':'observed_final_parity','matrix_root':summary['receipt_root_sha256'],'producer_build_root':pb['receipt_root_sha256'],'observed_harness_artifact_hashes':pins,'runs':rows,'actual_frame_bytes_compared':6*16588800,'fresh_processes':6,'strict_promotion':False,'goal_complete':False,'open_gates':['Frozen body v2 computes middle/repack/ViT coverage and decoder parameter/weight immutability but does not enforce those results in its code0 return when diagnostics are off.','Per-run pre/post executable, library and model-parameter identity receipts are absent in this matrix; build hashes are observed separately.','Production decoder pool coverage scans and intermediate RTX comparisons are omitted.','This matrix uses a fresh process per frame: combined-context reuse and temporal histories are separate gates.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',required=True,type=Path);p.add_argument('--matrix',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer_root,a.matrix)
    except Exception as e:r={'schema':'neuralstudio.nr.inmemory-matrix-observation-audit.v1','result':'fail','error':str(e),'strict_promotion':False}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
