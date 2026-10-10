#!/usr/bin/env python3
"""Independent prior-mode regression audit of the isolated half-FMA encoder candidates."""
import argparse
import hashlib
import json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def require(ok,label):
    if not ok:raise ValueError(label)
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');require(canonical(b)==h,'receipt seal');return r
ORIGINAL='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342'
ZERO_MOTION='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49'
REFERENCES={
 'black3':('build/nr_varied_temporal/case1-matrix-v1.json','65bbd1ba8734473c1d82c0e76933449724282e9ff2e8c05681cf3cee15677ffc'),
 'checker3':('build/nr_varied_temporal/case2-matrix-v1.json','be99d6c68dabeac39fd05105b4e84c268ddb87cbe53079dfe633208c8173264a'),
 'images3':('build/nr_image_sequence/matrix-audit-v1.json','8d02cace7fa03262b679b73e91b6af1cef2e17d834d60830e38169114e9f0c38'),
 'midreset8':('build/nr_eight_frame/matrix-audit-v1.json','f2fe8f4e02f4c40972b41643d482b37ef8e4ab87c8301593b6f9d9e8bbdd84c2'),
 'uniform3':('build/nr_motion_reference/matrix-audit-v1.json','da60f9d5cfc806e5b3dbba313f4a6adbe1e2478af829e63065966b68bd87150b'),
 'spatial3':('build/nr_spatial_motion/matrix-audit-v1.json','c055846cdf9534bc91782fedf6a8b05b3c4aab2163da20e4dd88d5efba0e445e'),
 'dense':('evidence/specs/nr-color-encoder-v1/rtx-reference.json','703c034ea02c686d18414ce7f463284152e056dccd5b2166987e5f5710a1abdd'),
}
def configuration(repo,name):
    require(name in REFERENCES,'known regression contract')
    path,root=REFERENCES[name];rtx=sealed(repo/path)
    require(rtx['receipt_root_sha256']==root,'pinned repeated RTX reference')
    if name=='dense':
        require(len(rtx['captures'])==4 and all(x['ports']==rtx['captures'][0]['ports'] for x in rtx['captures']),'repeated dense RTX boundaries')
        rtx={**rtx,'outputs':[rtx['full_rtx_output_sha256']],'boundary_ports':[{},rtx['captures'][0]['ports']]}
    n=len(rtx['outputs']);require(n==(8 if name=='midreset8' else 1 if name=='dense' else 3),'reference frame extent')
    resets=[1,0,0,0,1,0,0,0] if name=='midreset8' else [1]+[0]*(n-1)
    images=rtx.get('input_hashes',[rtx.get('input_sha256',ORIGINAL)]*n)
    motion=rtx.get('motion_sha256',ZERO_MOTION)
    fractional=name in ('uniform3','spatial3')
    return rtx,n,resets,images,motion,fractional

def audit(peer,matrix,name):
    repo=Path(__file__).resolve().parents[1]
    require(matrix.is_relative_to(peer/'build/nr_full'),'scoped peer matrix')
    rtx,n,resets,images,motion_hash,fractional=configuration(repo,name)
    build=peer/('build/nr_full/v20t' if fractional else 'build/nr_full/v19t')
    enc=repo/('build/nr_encoder_temporal/half-fma-spatial-v1' if fractional else 'build/nr_encoder_temporal/half-fma-v1')
    eb=sealed(enc/'build.json')
    require(eb['receipt_root_sha256']==('9289bb7b21bd36571fffbdf057757fc6397baea3d9377c813965558e0430bfec' if fractional else '0a57b3bac8616c7103ae30a2df5a91296f3bda2ab9f84960725bb943df053b2e'),'isolated encoder candidate')
    pins={}
    def observe(path):
        p=Path(path).resolve();h=sha(p)
        if str(p) in pins:require(pins[str(p)]==h,'artifact drift '+str(p))
        pins[str(p)]=h;return h
    def resolve(p):
        p=Path(p);return p.resolve() if p.is_absolute() else (peer/p).resolve()
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);require(observe(resolve(p.strip()))==h,'combined build')
    require(observe(build/'source/encoder/nr_encoder_runtime.o')==eb['object_sha256'],'linked isolated encoder candidate')
    require(observe(build/'source/src/nr_body/nr_body_runtime_v11.mm')=='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9','reviewed body')
    require(observe(build/'source/src/nr_body/nr_full_frame.mm')=='ade17848864e8fbf10c3240eb79423b39c784f45b8f262621816b2c3e4840338','reviewed harness')
    require(observe(build/'source/src/nr_temporal/nr_temporal.h')==observe(enc/'source/nr_temporal.h')=='d44c36cff1f40ea7574619a87a8d743048ca4f30792a7a771c6b2dd6011cb28d','pinned shared temporal ABI')
    require((build/'build-defines.txt').read_text().strip()=='defines: none','no test hooks')
    for asset_name,h in eb['assets'].items():require(observe(enc/'assets'/asset_name)==h,'encoder asset')
    regions={'entry':('block22-middle',1105920),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720),'block0_skip':('block0-skip',70778880)}
    rows=[];pids=set();stage_by_frame={};padding={}
    configs=[('pilot',([165,90]*n)[:n]),('replica-2',[165]*n),('replica-3',[90]*n),('replica-4',([90,165]*n)[:n])]
    for label,poisons in configs:
        d=matrix/label;r=read(d/'receipt.json');rh=observe(d/'receipt.json')
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v8' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==n,'all declared native calls')
        require(r['pid']>0 and r['pid'] not in pids,'fresh native process');pids.add(r['pid'])
        require(r['context_preparations']==1 and r['contexts_prepared_once'] and r['files_unchanged_after_run'],'one persistent context')
        cmd=r['command'];require(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==observe(build/'nr_full_frame'),'executed binary')
        require(not any(x.startswith('--encoder-diagnostics') for x in cmd),'encoder diagnostics disabled')
        require(resolve(cmd[1])==peer/('build/nr_body/desc-v24t/chain.json' if fractional else 'build/nr_body/desc-v23t/chain.json') and observe(resolve(cmd[1]))==r['body_descriptor_sha256'],'descriptor')
        require(resolve(cmd[2])==enc/'assets' and cmd[cmd.index('--reset-sequence')+1]==','.join(map(str,resets)),'assets and reset mask')
        motion=resolve(cmd[cmd.index('--motion')+1]);require(motion.stat().st_size==8294400 and observe(motion)==motion_hash,'pinned motion input')
        state=0
        for i,f in enumerate(r['frames']):
            before=state;reset=bool(resets[i]);step=0 if reset else state;state=step+1;e,b,t=f['encoder'],f['body'],f['temporal'];expected=rtx['outputs'][i]
            require(f['frame']==i and f['poison']==poisons[i] and f['result']=='pass' and not f['plumbing_only_no_parity_meaning'],'frame/poison')
            require(e['code']==b['code']==0 and not e['error'] and not b['error'] and e['gpu_complete'] and e['input_immutable'],'live APIs')
            require(e['diagnostics_dir']=='','no archived encoder diagnostics ingress')
            require(t['enabled'],'shared temporal state enabled')
            require(e['reset']==t['body_reset']==t['encoder_reset']==reset and e['frame_index']==step and e['entry']==('nr_encoder_run_frame' if reset else 'nr_encoder_run_temporal_frame'),'reset and live producer selection')
            require(t['frames_since_reset_before']==before and b['frames_since_reset']==step+1 and f['frames_since_reset_after']==step+1,'counter reset/progression')
            require(f['temporal_state_advanced'] and f['ex2_half_faults']=={'argument':0,'domain':0} and f['history_copy_value_differences']==0 and f['test_fault']=='','clean state commit')
            require(e['guarded_outputs_checked']==(1039 if reset else 1040) and e['coverage_values_checked']==(5900107776 if reset else 5935497216),'encoder coverage')
            ih=observe(resolve(f['input_file']));require(ih==images[i] and ih==e['input_sha256']==f['input_sha256']==b['input_texture_sha256_before']==b['input_texture_sha256_after'],'exact declared immutable image')
            require(b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'] and not b['check_failures'] and b['constants_unchanged_after_frame'],'immutability/checks')
            require(b['checks_enforced']==(242 if reset else 246) and b['constants_rehashed_after_frame']==(153 if i==0 else 154),'check counts and warm constants')
            require(b['checks_skipped']==['decoder per-launch pool hash','decoder per-launch pool coverage scan'],'only declared strict-checkpoint omissions')
            require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'final coverage')
            require(b['history_sha256_after']==f['surface_sha256_caller_readback']==b['final_surface_sha256']==expected,'output/history identities')
            if not reset:require(b['history_sha256_before']==rtx['outputs'][i-1] and e['sampler_variant']==(3 if fractional else 0),'live preceding history')
            require(set(f['regions'])==set(regions),'all six producer regions')
            for k,(target,size) in regions.items():
                v=f['regions'][k];require(v['offset']==256 and v['bytes']==size and v['sha256']==rtx['boundary_ports'][2*i+1][target],'RTX encoder boundary '+k)
            for label,h in b['loaded'].items():
                if label.startswith('lib:'):path=resolve(label[4:])
                elif label.startswith('file:'):path=resolve(label[5:].rsplit(':',1)[0])
                elif label.startswith('ex2_half_table:'):path=resolve(label.split(':',1)[1]);require(h=='f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf','pinned EX2 table')
                else:raise ValueError('unknown resource type')
                require(path.is_relative_to(peer/'build') and observe(path)==h,'loaded model resource')
            require(len(b['loaded'])==(160 if i==0 else (162 if reset else 163)),'complete warm resource inventory')
            for path in (d/f'final-{i}.rgba16le',d/f'diag/frame-{i}/tex_out_155.bin',d/f'diag/frame-{i}/tex_out_156.bin'):
                require(path.stat().st_size==16588800 and observe(path)==expected,'actual final/post/history bytes')
            require(f['reference']['cpu_comparison_only'] and not f['no_reference_no_parity_meaning'] and f['reference']['mismatching_values']==0 and f['reference']['sha256']==expected,'actual reference comparison required')
            reference=resolve(f['reference']['file']);require(reference.stat().st_size==16588800 and observe(reference)==expected,'actual RTX reference bytes')
            cp=d/f'diag/frame-{i}/chain-run.json';chain=read(cp);observe(cp);require(chain['pid']==r['pid'] and chain['poison']==poisons[i] and len(chain['launches'])==100,'same-process chain')
            stages=[];count=0
            for launch in chain['launches']:
                if launch['kind']=='ptx':
                    require(launch['weights_immutable'] and launch['params_immutable'],'decoder constants')
                    require(launch['pool_post_sha256']=='skipped' and launch['words_written_once'] is None and launch['words_written_more_than_once'] is None,'explicit production-style pool coverage limits')
                    if 'texture' in launch:require(launch['texture']['pixels_once']==2073600 and launch['texture']['pixels_multi']==0 and launch['texture']['output_sha256']==expected,'post/copy coverage')
                    continue
                count+=len(launch['outputs'])
                require(launch.get('coverage') or 'coverage_dwords' in launch,'middle write coverage present')
                if launch['launch']==57 and launch['kind']=='final_head':
                    require(launch['coverage']['out_vit']==[552960,0,102400],'known padding contract')
                    key=(i,poisons[i]);h=launch['outputs']['out_vit']
                    if key in padding:require(padding[key]==h,'same-poison padding hash')
                    padding[key]=h
                else:
                    stages.append((launch['launch'],launch['kind'],launch['outputs']))
                    for v in launch.get('coverage',{}).values():require(v[0]>0 and v[1:]==[0,0],'middle coverage')
                if 'coverage_dwords' in launch:require(launch['coverage_dwords']==[163840,0,0],'repack coverage')
            require(count==125,'all middle checkpoints')
            if i in stage_by_frame:require(stage_by_frame[i]==stages,'middle checkpoint repeats')
            stage_by_frame[i]=stages
            rows.append({'pid':r['pid'],'frame':i,'poison':poisons[i],'reset':reset,'output_and_history_sha256':expected,'native_receipt_sha256':rh})
        for p,m in r['post_run_file_identity'].items():require(m['unchanged'] and m['post']==m.get('pre',m.get('load'))==observe(p),'post-run file identity')
    for p,h in pins.items():require(sha(p)==h,'audit-time drift')
    return {'schema':'neuralstudio.nr.half-fma-regression-audit.v1','result':'pass_bounded_prior_mode_regression','configuration':name,'matrix':str(matrix),'RTX_matrix_root':rtx['receipt_root_sha256'],'encoder_root':eb['receipt_root_sha256'],'runs':rows,'actual_final_post_history_bytes_checked':4*n*3*16588800,'encoder_boundary_hashes_checked':4*n*6,'middle_checkpoint_hashes_repeated':4*n*124,'artifacts':pins,'full_goal_complete':False,'production_promotion':False,'limits':['Only this explicitly declared prior-mode case; other configurations and general input domains remain separate.','Decoder per-launch pool hashes/coverage are omitted in production-style runs; strict original-frame evidence remains separate.','Fresh RTX capture and actual state-input provenance remain pinned to the earlier independently audited reference; not a new capture.','No performance claim.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--configuration',choices=REFERENCES,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer_root.resolve(),a.matrix.resolve(),a.configuration)
    except Exception as e:r={'schema':'neuralstudio.nr.half-fma-regression-audit.v1','result':'fail','configuration':a.configuration,'error':str(e),'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','artifacts')},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
