#!/usr/bin/env python3
"""Audit v3 frame bytes, live API checks and recorded loaded-model provenance."""
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
def same(a,b):
    require(a.stat().st_size==b.stat().st_size==16588800,'frame size')
    with a.open('rb') as x,b.open('rb') as y:
        while True:
            v=x.read(1048576)
            if not v:break
            require(v==y.read(len(v)),'actual output byte mismatch')
def audit(peer,matrix):
    repo=Path(__file__).resolve().parents[1];peer=peer.resolve();matrix=matrix.resolve();summary=sealed(matrix/'summary.json')
    require(summary['schema']=='neuralstudio.nr.full-frame-inmemory-matrix.v2' and summary['harness_build']=='build/nr_full/v3' and summary['result']=='pass','reviewed v3 matrix')
    build=peer/'build/nr_full/v3';pins={}
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);path=peer/p.strip();require(sha(path)==h,'frozen build identity');pins[str(path)]=h
    require(sha(build/'source/src/nr_body/nr_body_runtime_v3.mm')=='e75e21cb6f7d3a000f1cbe8a111b294da0f96c1d1cd31180769cef8b9633d660','reviewed enforcement implementation')
    producer=repo/'build/nr_encoder_runtime/build-v1';pb=sealed(producer/'build.json')
    require(sha(build/'source/encoder/nr_encoder_runtime.o')==pb['object_sha256'],'accepted producer object')
    require(sha(build/'source/encoder/nr_encoder_runtime.h')==sha(producer/'source/nr_encoder_runtime.h'),'producer ABI')
    for n,h in pb['assets'].items():require(sha(producer/'assets'/n)==h,'pinned encoder asset')
    old=sealed(repo/'build/nr_encoder_fscale/live-v1/verification.json')['runs'][0];new=sealed(repo/'build/nr_color_encoder_reference/reference-3-v1/reference.json')
    oldports={Path(n).stem:m['sha256'] for n,m in old['native_exports'].items()};oldports['block0-skip']=old['block0_skip_sha256']
    cases={'original':(repo/'build/nr_color_cases/rtx-v1/case0',oldports),'dense':(repo/'build/nr_color_cases/rtx-v1/case3-retry-v1',new['ports'])}
    targets={'original':'a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','dense':new['full_rtx_output_sha256']}
    roles={'entry':('block22-middle',1105920),'block0_skip':('block0-skip',70778880),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720)}
    keys={'dense-a5-1','dense-a5-2','dense-5a-1','dense-5a-2','original-a5-1','original-5a-1'};require(len(summary['runs'])==6 and {r['run'] for r in summary['runs']}==keys,'matrix inventory')
    pids=set();rows=[];loaded_parent=None;cache={}
    def digest(path):
        if path not in cache:cache[path]=sha(path)
        return cache[path]
    for row in summary['runs']:
        key=row['run'];case,poison,_=key.split('-');case_dir,ports=cases[case];r=read(matrix/key/'receipt.json')
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v2' and r['result']=='pass' and r['frames_in_process']==1 and len(r['frames'])==1,'native one-frame record')
        require(r['pid']==row['pid'] and r['pid'] not in pids and r['pid']>0,'fresh process');pids.add(r['pid'])
        cmd=r['command'];exe=Path(cmd[0]);exe=exe if exe.is_absolute() else peer/exe
        require(exe.resolve()==(build/'nr_full_frame').resolve() and r['executable_sha256']==sha(exe),'execution-time executable identity')
        desc=Path(cmd[1]);desc=desc if desc.is_absolute() else peer/desc;require(sha(desc)==r['body_descriptor_sha256'],'execution descriptor identity')
        require(Path(cmd[2]).resolve()==(producer/'assets').resolve()==Path(r['encoder_asset_root']).resolve(),'pinned encoder assets')
        f=r['frames'][0];e=f['encoder'];b=f['body'];image_sha=sha(case_dir/'input-color.rgba16le')
        input_path=Path(f['input_file']).resolve();command_input=Path(cmd[3]);command_input=command_input if command_input.is_absolute() else peer/command_input
        allowed_images={case_dir/'input-color.rgba16le'}
        if case=='original':allowed_images.add(repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le')
        require(input_path in allowed_images and input_path==command_input.resolve() and sha(input_path)==f['input_sha256']==e['input_sha256']==image_sha,'original-image ingress')
        require(f['poison']==r['poison']==int(poison,16),'poison')
        require(e['code']==b['code']==0 and not b['check_failures'] and b['checks_enforced']==241,'body enforced check gate')
        require(e['input_immutable'] is True and e['gpu_complete'] is True and e['guarded_outputs_checked']==1039 and e['coverage_values_checked']==5900107776 and e['bundle_root']==pb['parent_root'],'producer checks')
        require(b['input_texture_immutable'] is True and b['input_texture_sha256_before']==b['input_texture_sha256_after']==image_sha,'body texture immutability')
        require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'surface write coverage')
        require(f['producer_regions_unchanged_by_body'] is True and set(f['regions'])==set(roles),'live regions retained')
        for role,(name,n) in roles.items():
            v=f['regions'][role];require(v['offset']==256 and v['bytes']==n and v['sha256']==ports[name],'live encoder boundary contract')
        loaded=b['loaded'];require(len(loaded)==160 and b['constants_rehashed_after_frame']==153 and b['constants_unchanged_after_frame'] is True,'loaded constant evidence')
        require(loaded_parent is None or loaded==loaded_parent,'loaded model differs across fresh processes');loaded_parent=loaded
        files=libs=0
        for name,h in loaded.items():
            if name.startswith('file:'):
                path,pad=name[5:].rsplit(':',1);require(int(pad) in (16,64,4096),'model buffer padding');files+=1
            elif name.startswith('lib:'):path=name[4:];libs+=1
            else:raise ValueError('unrecognized loaded resource')
            path=Path(path).resolve();require(path.is_relative_to(peer/'build') and path.suffix in ('.bin','.metallib'),'model resource outside reviewed tree/type')
            require(digest(path)==h,'recorded loaded model hash differs from retained artifact')
        require(files==153 and libs==7,'model resource inventory')
        actual=matrix/key/'final.rgba16le';expected=case_dir/'a5-a/output.rgba16le';same(actual,expected);h=sha(actual)
        require(h==targets[case]==f['surface_sha256_caller_readback']==b['final_surface_sha256']==f['reference']['sha256'] and f['reference']['mismatching_values']==0 and f['reference']['cpu_comparison_only'] is True,'actual RTX frame parity')
        rows.append({'key':key,'pid':r['pid'],'final_sha256':h,'receipt_sha256':sha(matrix/key/'receipt.json'),'checks_enforced':b['checks_enforced'],'constants_rehashed':files,'loaded_fingerprint':canonical(loaded)})
    # Recheck files after the audit, not only once while building its inventory.
    for path,h in cache.items():require(sha(path)==h,'asset drift during audit')
    return {'schema':'neuralstudio.nr.inmemory-v3-independent-audit.v1','result':'pass_tested_reset_frames','matrix_root':summary['receipt_root_sha256'],'producer_build_root':pb['receipt_root_sha256'],'observed_build_hashes':pins,'runs':rows,'actual_frame_bytes_compared':6*16588800,'loaded_model_resources':len(cache),'loaded_model_fingerprint':canonical(loaded_parent),'fully_inmemory_frames_verified':6,'full_goal_complete':False,'scope':'Six fresh reset-only in-memory frames on original/dense inputs, exact retained RTX bytes, same live producer regions, enforced native checks, execution-time executable/descriptor and loaded-model hashes, post-frame GPU model-constant rehash. Not a general temporal/descriptor/resolution/performance claim.','limits':['Decoder production-mode per-launch pool write scans and new-input intermediate RTX comparisons remain omitted.','Frozen v3 still performs the entry blit before later texture/skip validation; valid tested frames are unaffected, but no-GPU-work-on-invalid-input is not established.','Loaded library hashes and executable hash are bound at execution/load and rehashed now, not a separately recorded immediate post-run file manifest.','Mixed-image v3 sequence evidence requires separate audit.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',required=True,type=Path);p.add_argument('--matrix',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();require(not a.output.exists(),'preserve previous audit')
    try:r=audit(a.peer_root,a.matrix)
    except Exception as e:r={'schema':'neuralstudio.nr.inmemory-v3-independent-audit.v1','result':'fail','error':str(e),'full_goal_complete':False}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
