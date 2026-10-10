#!/usr/bin/env python3
"""Independent retained-byte audit of eight native frames and mid-sequence reset."""
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
def audit(peer):
    repo=Path(__file__).resolve().parents[1];rtx=sealed(repo/'build/nr_eight_frame/matrix-audit-v1.json');packet=sealed(repo/'build/nr_eight_frame/packet-audit-v1.json')
    require(rtx['receipt_root_sha256']==packet['matrix_root']=='f2fe8f4e02f4c40972b41643d482b37ef8e4ab87c8301593b6f9d9e8bbdd84c2','repeated RTX references')
    build=peer/'build/nr_full/v16t';enc=repo/'build/nr_encoder_temporal/build-v2-sass';eb=sealed(enc/'build.json');require(eb['receipt_root_sha256']=='b2628242a1e425ec460cd2cb5e445da64283d736a3c5b1789bcecfa127c84c9f','accepted encoder')
    pins={}
    def observe(path):
        p=Path(path).resolve();h=sha(p)
        if str(p) in pins:require(pins[str(p)]==h,'artifact drift '+str(p))
        pins[str(p)]=h;return h
    def resolve(p):
        p=Path(p);return p.resolve() if p.is_absolute() else (peer/p).resolve()
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);require(observe(resolve(p.strip()))==h,'combined build')
    require(observe(build/'source/encoder/nr_encoder_runtime.o')==eb['object_sha256'],'linked accepted encoder')
    require(observe(build/'source/src/nr_body/nr_body_runtime_v11.mm')=='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9','reviewed body')
    require(observe(build/'source/src/nr_body/nr_full_frame.mm')=='264005ff6d684cfebbfedeb789ff0f9494995ce45989a57c554e3d61234969b2','reviewed harness')
    require((build/'build-defines.txt').read_text().strip()=='defines: none','no test hooks')
    for n,h in eb['assets'].items():require(observe(enc/'assets'/n)==h,'encoder asset')
    regions={'entry':('block22-middle',1105920),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720),'block0_skip':('block0-skip',70778880)}
    rows=[];pids=set();stage_by_frame={};padding={}
    configs=[('matrix-a5',[165]*8),('matrix-5a',[90]*8),('matrix-a5-5a',[165,90]*4),('matrix-5a-a5',[90,165]*4)]
    for name,poisons in configs:
        d=peer/'build/nr_full/temporal-eight-v1'/name;r=read(d/'receipt.json');rh=observe(d/'receipt.json')
        require(r['schema']=='neuralstudio.nr.full-frame-inmemory.v7' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==8,'eight successful native calls')
        require(r['pid']>0 and r['pid'] not in pids,'fresh native process');pids.add(r['pid'])
        require(r['context_preparations']==1 and r['contexts_prepared_once'] and r['files_unchanged_after_run'],'one persistent context')
        cmd=r['command'];require(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==observe(build/'nr_full_frame'),'executed binary')
        require(resolve(cmd[1])==peer/'build/nr_body/desc-v23t/chain.json' and observe(resolve(cmd[1]))==r['body_descriptor_sha256'],'descriptor')
        require(resolve(cmd[2])==enc/'assets' and cmd[cmd.index('--reset-sequence')+1]=='1,0,0,0,1,0,0,0','assets and reset mask')
        motion=resolve(cmd[cmd.index('--motion')+1]);require(motion.stat().st_size==8294400 and observe(motion)=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','zero motion')
        for i,f in enumerate(r['frames']):
            step=i%4;reset=step==0;e,b,t=f['encoder'],f['body'],f['temporal'];expected=rtx['outputs'][i]
            require(f['frame']==i and f['poison']==poisons[i] and f['result']=='pass' and not f['plumbing_only_no_parity_meaning'],'frame/poison')
            require(e['code']==b['code']==0 and not e['error'] and not b['error'] and e['gpu_complete'] and e['input_immutable'],'live APIs')
            require(e['reset']==t['body_reset']==t['encoder_reset']==reset and e['frame_index']==step and e['entry']==('nr_encoder_run_frame' if reset else 'nr_encoder_run_temporal_frame'),'reset and live producer selection')
            require(t['frames_since_reset_before']==(0 if i==0 else (i-1)%4+1) and b['frames_since_reset']==step+1 and f['frames_since_reset_after']==step+1,'counter reset/progression')
            require(f['temporal_state_advanced'] and f['ex2_half_faults']=={'argument':0,'domain':0} and f['history_copy_value_differences']==0 and f['test_fault']=='','clean state commit')
            require(e['guarded_outputs_checked']==(1039 if reset else 1040) and e['coverage_values_checked']==(5900107776 if reset else 5935497216),'encoder coverage')
            ih=observe(resolve(f['input_file']));require(ih=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342' and ih==e['input_sha256']==f['input_sha256']==b['input_texture_sha256_before']==b['input_texture_sha256_after'],'original immutable input')
            require(b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'] and not b['check_failures'] and b['constants_unchanged_after_frame'],'immutability/checks')
            require(b['checks_enforced']==(242 if reset else 246) and b['constants_rehashed_after_frame']==(153 if i==0 else 154),'check counts and warm constants')
            require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'final coverage')
            require(b['history_sha256_after']==f['surface_sha256_caller_readback']==b['final_surface_sha256']==expected,'output/history identities')
            if not reset:require(b['history_sha256_before']==rtx['outputs'][i-1] and e['sampler_variant']==0,'live preceding history')
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
            reference=Path(packet['outputs'][i]['file']);require(observe(reference)==expected,'actual RTX comparison file')
            for path in (d/f'final-{i}.rgba16le',d/f'diag/frame-{i}/tex_out_155.bin',d/f'diag/frame-{i}/tex_out_156.bin'):
                require(path.stat().st_size==16588800 and observe(path)==expected,'actual final/post/history bytes')
            if i<3:require(f['reference']['mismatching_values']==0 and f['reference']['sha256']==expected,'original reference comparisons')
            else:require(f['no_reference_no_parity_meaning'] and f['reference']['file']=='','later comparison does not rewrite original unreferenced receipts')
            cp=d/f'diag/frame-{i}/chain-run.json';chain=read(cp);observe(cp);require(chain['pid']==r['pid'] and chain['poison']==poisons[i] and len(chain['launches'])==100,'same-process chain')
            stages=[];count=0
            for launch in chain['launches']:
                if launch['kind']=='ptx':
                    require(launch['weights_immutable'] and launch['params_immutable'],'decoder constants')
                    if 'texture' in launch:require(launch['texture']['pixels_once']==2073600 and launch['texture']['pixels_multi']==0 and launch['texture']['output_sha256']==expected,'post/copy coverage')
                    continue
                count+=len(launch['outputs'])
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
    return {'schema':'neuralstudio.nr.eight-native-audit.v1','result':'pass_eight_frame_mid_reset_parity','RTX_matrix_root':rtx['receipt_root_sha256'],'packet_root':packet['receipt_root_sha256'],'runs':rows,'actual_final_post_history_bytes_checked':32*3*16588800,'encoder_boundary_hashes_checked':192,'middle_checkpoint_hashes_repeated':32*124,'artifacts':pins,'full_goal_complete':False,'production_promotion':False,'limits':['Original image/zero motion; eight calls consist of two four-frame cycles separated by reset, not eight uninterrupted history frames.','Decoder per-launch pool hashes/coverage are omitted in these production-style runs; strict original-frame evidence remains separate.','Frames3-7 were originally unreferenced; this independent comparison supplements rather than rewrites those receipts.','No general sequence-length or performance claim.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    r=audit(a.peer_root.resolve());r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','artifacts')},indent=2))
if __name__=='__main__':main()
