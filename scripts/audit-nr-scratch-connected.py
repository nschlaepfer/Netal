#!/usr/bin/env python3
"""Audit the host-only scratch change's connected regression and file provenance."""
import argparse
import hashlib
import json
from pathlib import Path

def require(ok,label):
    if not ok:raise ValueError(label)
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'seal '+str(p));return r
def audit(peer):
    repo=Path(__file__).resolve().parents[1];matrix=peer/'build/nr_full/bwc-scratch-v1';baseline=peer/'build/nr_full/bwc-v1';build=peer/'build/nr_full/v5-scratch';candidate=repo/'build/nr_encoder_runtime/scratch-candidate-v1'
    summary=sealed(matrix/'summary.json');old=sealed(baseline/'summary.json');cb=sealed(candidate/'build.json');checks=sealed(repo/'build/nr_encoder_runtime/scratch-matrix-v1/independent-audit-v1.json');refresh=sealed(repo/'build/nr_full_inmemory_audit/black-checker-refresh-v1.json')
    require(summary['result']==old['result']=='pass' and summary['harness_build']=='build/nr_full/v5-scratch','connected matrix selection')
    require(checks['result']=='pass' and checks['build_root']==cb['receipt_root_sha256'] and refresh['result']=='pass_boundary_and_final_supplement' and refresh['matrix_root']==old['receipt_root_sha256'],'audited encoder and refreshed reference parents')
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);require(sha(peer/p.strip())==h,'retained combined build identity')
    require(sha(build/'source/encoder/nr_encoder_runtime.o')==cb['object_sha256']==sha(candidate/'nr_encoder_runtime.o'),'linked frozen encoder')
    require(sha(build/'source/src/nr_body/nr_body_runtime_v4.mm')=='ab53f3c85c6cec84f8a4129f4ed4ac40d7be51df23aae3413ea4d3730a1d996f','unchanged reviewed body implementation')
    require(sha(build/'source/encoder/nr_encoder_runtime.h')==sha(candidate/'source/nr_encoder_runtime.h'),'producer ABI')
    for n,h in cb['assets'].items():require(sha(candidate/'assets'/n)==h,'pinned producer asset')
    def path(s):
        p=Path(s);return p.resolve() if p.is_absolute() else (peer/p).resolve()
    pids=set();rows=[];material={};seen={};oldrows={p['process']:p for p in old['processes']};require(len(summary['processes'])==4,'four fresh processes')
    for process in summary['processes']:
        key=process['process'];r=read(matrix/key/'receipt.json');prior=read(baseline/key/'receipt.json')
        require(r['pid']==process['pid'] and r['pid'] not in pids and r['pid'] not in {p['pid'] for p in oldrows.values()},'fresh native process');pids.add(r['pid'])
        require(r['result']=='pass' and r['files_unchanged_after_run'] and r['frames_in_process']==len(r['frames'])==4,'connected run gate')
        require(path(r['command'][0])==(build/'nr_full_frame').resolve() and r['executable_sha256']==sha(build/'nr_full_frame'),'executed combined binary')
        require(path(r['command'][1])==peer/'build/nr_body/desc-v17/production-a5-a/chain.json' and sha(path(r['command'][1]))==r['body_descriptor_sha256']=='c70638f54a1858ba43375845b3670a081394b8b458fce214b8d53a2cedb1160c','v17 descriptor')
        require(path(r['command'][2])==(candidate/'assets').resolve()==path(r['encoder_asset_root']),'candidate assets selected')
        for f,oldf,claimed in zip(r['frames'],prior['frames'],process['frames']):
            require(f['frame']==oldf['frame']==claimed['frame'] and f['poison']==oldf['poison']==claimed['poison'],'same input order and poison')
            require(f['input_sha256']==oldf['input_sha256']==sha(path(f['input_file'])) and f['regions']==oldf['regions'],'all six live boundaries unchanged')
            e,b=f['encoder'],f['body'];require(e['code']==b['code']==0 and e['input_immutable'] and e['gpu_complete'] and f['producer_regions_unchanged_by_body'],'live API and lifetime gates')
            require(e['guarded_outputs_checked']==1039 and e['coverage_values_checked']==5900107776 and e['bundle_root']==cb['parent_root'],'encoder checks')
            require(b['checks_enforced']==241 and not b['check_failures'] and b['constants_unchanged_after_frame'] and b['constants_rehashed_after_frame']==153,'enforced body checks')
            require(b['input_texture_immutable'] and b['input_texture_sha256_before']==b['input_texture_sha256_after']==f['input_sha256'],'live texture immutable')
            require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'surface coverage')
            loaded=b['loaded'];require(len(loaded)==160 and sum(k.startswith('lib:') for k in loaded)==7 and sum(k.startswith('file:') for k in loaded)==153,'loaded model inventory')
            for n,h in loaded.items():
                file=path(n[4:] if n.startswith('lib:') else n[5:].rsplit(':',1)[0]);require(file.is_relative_to(peer/'build'),'model path scope')
                if str(file) not in material:material[str(file)]=sha(file)
                require(material[str(file)]==h,'loaded model identity')
            actual=matrix/key/f'final-{f["frame"]}.rgba16le';expected=path(f['reference']['file']);require(actual.stat().st_size==expected.stat().st_size==16588800,'final byte extent');h=sha(actual)
            require(h==sha(expected)==sha(baseline/key/actual.name)==f['surface_sha256_caller_readback']==b['final_surface_sha256']==claimed['final_sha256'],'actual final reference equality')
            require(f['reference']['cpu_comparison_only'] and f['reference']['mismatching_values']==0,'comparison only')
            rows.append({'pid':r['pid'],'frame':f['frame'],'input_sha256':f['input_sha256'],'poison':f['poison'],'final_sha256':h});seen.setdefault(f['input_sha256'],[]).append(f['poison'])
        post=r['post_run_file_identity'];required={str((build/'nr_full_frame').resolve())}|{str(candidate/'assets'/n) for n in cb['assets'] if not n.startswith('weights/')}|{k[4:] for k in r['frames'][0]['body']['loaded'] if k.startswith('lib:')}
        require(set(post)==required,'complete executed-file post identity inventory')
        for n,meta in post.items():require(meta['unchanged'] and meta['post']==meta.get('pre',meta.get('load'))==sha(Path(n)),'pre/load/post file integrity')
    require(len(rows)==16 and len(seen)==4 and all(sorted(v)==[90,90,165,165] for v in seen.values()),'complete four-case dual-poison matrix')
    for p,h in material.items():require(sha(Path(p))==h,'model drift during audit')
    return {'schema':'neuralstudio.nr.scratch-connected-independent-audit.v1','result':'pass_tested_reset_frames','candidate_root':cb['receipt_root_sha256'],'matrix_root':summary['receipt_root_sha256'],'reference_refresh_root':refresh['receipt_root_sha256'],'runs':rows,'final_bytes_checked':16*16588800,'live_boundary_hashes_compared':96,'retained_model_files_checked':len(material),'correctness_promotion_supported':True,'performance_promotion_supported':False,'full_goal_complete':False,'limits':['Reset-only fixed 1920x1080 descriptors and four synthetic inputs.','In-memory producer regions are recorded hashes, not retained intermediate payloads.','Production decoder per-launch pool coverage scans remain omitted; strict original-frame evidence is separate.','This is an incremental host-change regression audit, not universal model or temporal equivalence.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer_root.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.scratch-connected-independent-audit.v1','result':'fail','error':str(e)}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='runs'},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
