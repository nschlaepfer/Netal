#!/usr/bin/env python3
"""Independent audit of the pinned v9t/v10t reset[1,0] full-output matrices.

This observes bounded parity, not general temporal acceptance or production
promotion. Known negative-control and strict-checkpoint gaps are explicit.
"""
import argparse
import hashlib
import json
from pathlib import Path

def require(ok, label):
    if not ok: raise ValueError(label)
def sha(p):
    with Path(p).open('rb') as f: return hashlib.file_digest(f, 'sha256').hexdigest()
def read(p): return json.loads(Path(p).read_text())
def canonical(r): return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=read(p); b=dict(r); h=b.pop('receipt_root_sha256'); require(canonical(b)==h, 'seal '+str(p)); return r

def audit(peer, version=3, case=0, fractional=False, changing=False, spatial=False):
    repo=Path(__file__).resolve().parents[1]
    if spatial:require(not changing and case==0 and version==6,'reviewed spatial configuration');fractional=True
    require(version in (2,3,4,6),'bounded matrix version')
    matrix=peer/f'build/nr_full/temporal-seq-v{version}'; build=peer/f'build/nr_full/v{version+7}t'; frames=3 if version>=4 else 2
    require(case in (0,1,2) and (case==0 or version==6),'reviewed varied-case build')
    if case: matrix=peer/f'build/nr_full/temporal-varied-v1/case{case}'
    enc=repo/'build/nr_encoder_temporal/build-v2-sass'; eb=sealed(enc/'build.json')
    er=sealed(repo/'build/nr_encoder_temporal/matrix-audit-v1.json')
    rtx=sealed(repo/'build/nr_reset_history/matrix-audit-v1.json')
    rtx3=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json') if version==6 else None
    if rtx3: require(rtx3['receipt_root_sha256']=='e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657','third-frame RTX parent')
    ex2=sealed(repo/'build/nr_ex2_half/table-v1/verification.json')
    require(eb['receipt_root_sha256']=='b2628242a1e425ec460cd2cb5e445da64283d736a3c5b1789bcecfa127c84c9f','accepted encoder build')
    require(er['rtx_reference_root']==rtx['receipt_root_sha256']=='b2f44873eebc444a53784b2a2eaf2d4b5f221174436cb1c14fe2f450c1bf2ddf','accepted temporal RTX parent')
    if case:
        varied=sealed(repo/f'build/nr_varied_temporal/case{case}-matrix-v1.json')
        require(varied['case']==case and varied['receipt_root_sha256']=={1:'65bbd1ba8734473c1d82c0e76933449724282e9ff2e8c05681cf3cee15677ffc',2:'be99d6c68dabeac39fd05105b4e84c268ddb87cbe53079dfe633208c8173264a'}[case],'audited varied RTX reference')
        rtx=rtx3=varied
    if fractional:
        require(case==0 and version==6,'reviewed fractional-motion configuration')
        matrix=peer/'build/nr_full/temporal-motion-v2';build=peer/'build/nr_full/v14t'
        enc=repo/'build/nr_encoder_temporal/motion-v1';eb=sealed(enc/'build.json')
        er=sealed(repo/'build/nr_encoder_temporal/motion-matrix-audit-v1.json')
        rtx=rtx3=sealed(repo/'build/nr_motion_reference/matrix-audit-v1.json')
        require(eb['receipt_root_sha256']=='1b42bb34e487a0817c7ee7491565e88f3780b0afad724f906c452377759a6cd3' and er['receipt_root_sha256']=='c06252aabe4dc31ff20f69eefb552f305c67f0ebbb0e4405f6e0308fa31c1344','accepted motion encoder')
        require(er['rtx_reference_root']==rtx['receipt_root_sha256']=='da60f9d5cfc806e5b3dbba313f4a6adbe1e2478af829e63065966b68bd87150b','accepted motion RTX matrix')
    if changing:
        require(case==0 and version==6 and not fractional,'reviewed changing-image configuration')
        matrix=peer/'build/nr_full/temporal-images-v1';build=peer/'build/nr_full/v16t'
        rtx=rtx3=sealed(repo/'build/nr_image_sequence/matrix-audit-v1.json')
        require(rtx['receipt_root_sha256']=='8d02cace7fa03262b679b73e91b6af1cef2e17d834d60830e38169114e9f0c38' and rtx['color_cases']==[0,2,1],'accepted changing-image RTX matrix')
    if spatial:
        matrix=peer/'build/nr_full/temporal-spatial-v1';build=peer/'build/nr_full/v17t'
        enc=repo/'build/nr_encoder_temporal/spatial-v1';eb=sealed(enc/'build.json')
        er=sealed(repo/'build/nr_encoder_temporal/spatial-matrix-audit-v1.json');rtx=rtx3=sealed(repo/'build/nr_spatial_motion/matrix-audit-v1.json')
        require(eb['receipt_root_sha256']=='7c5531e9d15b68b4252060bc2e5daeaa55aa62fe155bcb3ec59fe89441ca5844' and er['receipt_root_sha256']=='2cf5c749862773214bbe4ef957e1399d78bd3d4ffac3359f9ef5760e022dd6a2','accepted spatial encoder')
        require(er['rtx_reference_root']==rtx['receipt_root_sha256']=='c055846cdf9534bc91782fedf6a8b05b3c4aab2163da20e4dd88d5efba0e445e','accepted spatial RTX matrix')
    table_sha=ex2['table']['sha256']; require(table_sha=='f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf','audited primitive table')
    material={}
    def observe(p):
        p=Path(p).resolve(); h=sha(p)
        if str(p) in material: require(material[str(p)]==h,'artifact changed during audit '+str(p))
        material[str(p)]=h; return h
    def resolve(p):
        p=Path(p); return p.resolve() if p.is_absolute() else (peer/p).resolve()
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1); require(observe(resolve(p.strip()))==h,'combined build identity')
    require(observe(build/'source/encoder/nr_encoder_runtime.o')==eb['object_sha256'],'frozen encoder object')
    body_source=f'nr_body_runtime_v{version+4}.mm'; body_sha={2:'33219cfaa31a3bff0753d6b16b42c09aeb6da924908cc9e2de56534ed4f85277',3:'c462475389acbf017f88cc68520d6c4317900dd5cbcf36be0e2609566bd6640e',4:'3976e39bedde2790858a55bc587267005ecde708025c236d9bc0bfa117eaf2e2',6:'8f02851c1ec1ffdcf7f1ff499e33bd2ebedd436f343ddcd0d5aff9caa7ddb038'}[version]
    if changing or spatial:body_source='nr_body_runtime_v11.mm';body_sha='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9'
    require(observe(build/'source/src/nr_body'/body_source)==body_sha,'reviewed body source')
    harness_sha={4:'d2a8e22da68c6ddb50abe9e186ee8886bb4c5091bd3fcfc571adebf1b4cdf7f2',6:'df6ddd0cc76cfa6c3d88fb2325a5d63c05adfc3a5136203cc317f9c96a01449d'}.get(version,'a60c166940e688b210e37e3056167e6a62df49534541094a81aa866c8505a07c')
    if fractional:harness_sha='49d763639b30fae7f625ce960334cb12efa993b8de5208d90530914dd2fcc378'
    if changing or spatial:harness_sha='264005ff6d684cfebbfedeb789ff0f9494995ce45989a57c554e3d61234969b2'
    require(observe(build/'source/src/nr_body/nr_full_frame.mm')==harness_sha,'reviewed live-history harness source')
    if version==6: require((build/'build-defines.txt').read_text().strip()=='defines: none','production build has no fault hooks')
    require(observe(build/'source/src/nr_temporal/nr_temporal.h')==observe(enc/'source/nr_temporal.h'),'shared history ABI')
    for n,h in eb['assets'].items(): require(observe(enc/'assets'/n)==h,'frozen producer asset')
    sizes={'entry':1105920,'skip_c256':2211840,'skip_c128':4423680,'skip_c64':8847360,'skip_c32':17694720,'block0_skip':70778880}
    ports={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    pids=set(); rows=[]; stages={}; loaded_by_frame={}; records=[]; regions_by_frame={}; outputs_by_frame={}; padding_hashes={}
    for label,poisons in [('a5-a5',(165,165)),('5a-5a',(90,90)),('a5-5a',(165,90)),('5a-a5',(90,165))]:
        if frames==3: poisons=(*poisons,poisons[0])
        if case or changing: label='matrix-'+'-'.join(f'{p:02x}' for p in poisons)
        if fractional:label='-'.join(f'{p:02x}' for p in poisons)
        if spatial:label='matrix-'+'-'.join(f'{p:02x}' for p in poisons)
        folder=matrix/label; receipt=folder/'receipt.json'; records.append(observe(receipt)); r=read(receipt)
        require(r['schema']==('neuralstudio.nr.full-frame-inmemory.v7' if changing or spatial else ('neuralstudio.nr.full-frame-inmemory.v5' if fractional else 'neuralstudio.nr.full-frame-inmemory.v4')) and r['result']=='pass','native matrix result')
        require(r['pid']>0 and r['pid'] not in pids,'fresh process'); pids.add(r['pid'])
        require(r['frames_in_process']==len(r['frames'])==frames and r['contexts_prepared_once'] and r['files_unchanged_after_run'],'persistent contexts/identity')
        cmd=r['command']; require(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==observe(build/'nr_full_frame'),'executed binary')
        descriptor_path='build/nr_body/desc-v24t/chain.json' if fractional else ('build/nr_body/desc-v23t/chain.json' if version>=4 else 'build/nr_body/desc-v22t/chain.json')
        require(resolve(cmd[1])==peer/descriptor_path and r['body_descriptor_sha256']==observe(resolve(cmd[1])),'descriptor identity')
        descriptor=read(resolve(cmd[1])); padding=descriptor['vit']['padding_tokens']; require(len(padding)==len(set(padding))==100 and all(0<=x<640 for x in padding),'original final-head padding contract')
        require(resolve(cmd[2])==resolve(r['encoder_asset_root'])==enc/'assets','producer asset root')
        require(cmd[cmd.index('--reset-sequence')+1]==('1,0,0' if frames==3 else '1,0'),'reset sequence')
        motion=resolve(cmd[cmd.index('--motion')+1]);mv=motion.read_bytes();require(len(mv)==1920*1080*4,'motion extent');motion_sha=observe(motion)
        if fractional:require(motion_sha==rtx['motion_sha256'] and cmd[cmd.index('--encoder-sampler-variant')+1]=='3','admitted fractional motion and sampler')
        else:require(not any(mv),'zero-motion admitted domain')
        for i,f in enumerate(r['frames']):
            require(f['frame']==i and f['poison']==poisons[i] and f['result']=='pass' and not f['plumbing_only_no_parity_meaning'],'real frame and poison')
            e,b,t=f['encoder'],f['body'],f['temporal']; h=rtx['outputs'][i] if i<2 else observe(folder/f'final-{i}.rgba16le')
            if rtx3: require(h==rtx3['outputs'][i],'all three RTX output hashes')
            if i in outputs_by_frame: require(outputs_by_frame[i]==h,'repeated output bytes')
            outputs_by_frame[i]=h
            require(e['code']==b['code']==0 and not e['error'] and not b['error'] and e['gpu_complete'] and e['input_immutable'],'live APIs')
            require(e['entry']==('nr_encoder_run_frame' if i==0 else 'nr_encoder_run_temporal_frame') and e['reset']==(i==0) and e['frame_index']==i,'live producer selection')
            require(e['guarded_outputs_checked']==1039+min(i,1) and e['coverage_values_checked']==(5900107776 if i==0 else 5935497216),'encoder coverage')
            if i==1 and case==0 and not changing: require(e['feature_sha256']==er['feature_sha256'],'verified second-frame features')
            if i: require(e['sampler_variant']==(3 if fractional else 0),'admitted sampler')
            require(t['enabled'] and t['body_reset']==t['encoder_reset']==(i==0) and t['frames_since_reset_before']==i,'shared state continuity')
            require(b['frames_since_reset']==i+1 and b['history_sha256_after']==h,'live history advances')
            if i: require(b['history_sha256_before']==outputs_by_frame[i-1] and b['motion_sha256']==motion_sha,'own preceding output used as history')
            if version>=4: require(f['history_copy_value_differences']==0,'explicit history bit comparison')
            if version==6: require(f['temporal_state_advanced'] and f['ex2_half_faults']=={'argument':0,'domain':0} and f['test_fault']=='','clean commit and explicitly serialized fault counts')
            ih=observe(resolve(f['input_file'])); require(ih==e['input_sha256']==f['input_sha256']==b['input_texture_sha256_before']==b['input_texture_sha256_after'],'input identity')
            if changing:require(ih==rtx['input_hashes'][i],'per-frame changed image matches RTX')
            require(b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'],'input/producer immutability')
            require(b['checks_enforced']==(242 if i==0 else 246) and not b['check_failures'] and b['constants_unchanged_after_frame'] and b['constants_rehashed_after_frame']==153+min(i,1),'enforced body checks')
            require(b['checks_skipped']==['decoder per-launch pool hash','decoder per-launch pool coverage scan'],'known strict-checkpoint limits')
            require(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'output coverage')
            require(set(f['regions'])==set(sizes),'all producer regions')
            for name,size in sizes.items():
                meta=f['regions'][name]; require(meta['bytes']==size and meta['offset']==256,'producer region extent '+name)
                if i<2: require(meta['sha256']==rtx['boundary_ports'][1+2*i][ports[name]],'RTX producer boundary '+name)
                elif rtx3: require(meta['sha256']==rtx3['boundary_ports'][5][ports[name]],'third-frame RTX producer boundary '+name)
            if i in regions_by_frame: require(regions_by_frame[i]==f['regions'],'live producer boundary repeat')
            regions_by_frame[i]=f['regions']
            loaded=b['loaded']; require(len(loaded)==(160 if i==0 else 163),'complete loaded inventory')
            if i in loaded_by_frame: require(loaded_by_frame[i]==loaded,'loaded inventory repeat')
            loaded_by_frame[i]=loaded
            for n,value in loaded.items():
                if n.startswith('lib:'): p=resolve(n[4:])
                elif n.startswith('file:'): p=resolve(n[5:].rsplit(':',1)[0])
                elif n.startswith('ex2_half_table:'): p=resolve(n[len('ex2_half_table:'):]); require(value==table_sha,'promoted EX2 table')
                else: raise ValueError('unknown loaded resource')
                require(p.is_relative_to(peer/'build') and observe(p)==value,'actual loaded resource identity')
            for p in [folder/f'final-{i}.rgba16le']+([resolve(f['reference']['file'])] if i<2 or rtx3 else []):
                require(p.stat().st_size==16588800 and observe(p)==h,'actual final bytes versus RTX')
            require(f['surface_sha256_caller_readback']==b['final_surface_sha256']==h and f['reference']['cpu_comparison_only'],'independent caller readback')
            if i<2 or rtx3: require(f['reference']['sha256']==h and f['reference']['mismatching_values']==0,'RTX reference equality')
            else: require(f['reference']['file']==f['reference']['sha256']=='' and f['reference']['mismatching_values']==-1 and f['no_reference_no_parity_meaning'],'unreferenced third frame honestly labelled')
            diag=folder/f'diag/frame-{i}'; chain=read(diag/'chain-run.json'); observe(diag/'chain-run.json')
            require(chain['pid']==r['pid'] and chain['poison']==poisons[i] and len(chain['launches'])==100,'same-process body diagnostics')
            digest=[]; out_count=0
            for launch in chain['launches']:
                if launch['kind']=='ptx':
                    require(launch['weights_immutable'] and launch['params_immutable'],'PTX constants')
                    require(launch['pool_post_sha256']=='skipped' and launch['words_written_once'] is None and launch['words_written_more_than_once'] is None,'explicit omitted pool scans')
                    if 'texture' in launch:
                        tx=launch['texture']; require(tx['pixels_once']==tx['pixels']==2073600 and tx['pixels_multi']==0 and tx['output_sha256']==h,'post/copy coverage and equality')
                        raw=diag/f'tex_out_{launch["launch"]}.bin'; actual_history=observe(raw)
                        require(actual_history==h, f'retained post/copy bytes: {raw}; actual={actual_history}; required={h}')
                else:
                    outputs=launch['outputs']; out_count+=len(outputs)
                    if launch['launch']==57 and launch['kind']=='final_head':
                        require(set(outputs)=={'out_vit'},'only known padding-dependent checkpoint')
                        key=(i,poisons[i]); value=outputs['out_vit']
                        if key in padding_hashes: require(padding_hashes[key]==value,'padding checkpoint repeat within same poison')
                        padding_hashes[key]=value
                    else: digest.append((launch['launch'],launch['kind'],outputs))
                    coverage=launch.get('coverage',{})
                    require(coverage or 'coverage_dwords' in launch,'middle coverage recorded')
                    for name,c in coverage.items():
                        if launch['launch']==57 and launch['kind']=='final_head' and name=='out_vit':
                            require(c==[552960,0,len(padding)*1024],'final-head padding write contract')
                        else: require(len(c)==3 and c[0]>0 and c[1:]==[0,0],'middle exactly-once coverage')
                    if 'coverage_dwords' in launch: require(launch['coverage_dwords']==[163840,0,0],'repack write contract')
            require(out_count==125,'all recorded middle checkpoints')
            if i in stages: require(stages[i]==digest,'middle checkpoint repeat')
            stages[i]=digest
            rows.append({'pid':r['pid'],'frame':i,'poison':poisons[i],'final_sha256':h,'native_receipt_sha256':records[-1]})
        for n,meta in r['post_run_file_identity'].items(): require(meta['unchanged'] and meta['post']==meta.get('pre',meta.get('load'))==observe(Path(n)),'pre/load/post identity')
    if version>=3:
        neg=peer/f'build/nr_temporal/negctl-v{11 if changing or spatial else version+4}/summary.txt'; text=neg.read_text(); observe(neg)
        require(('negative controls passed 20/20' if changing or spatial else 'negative controls passed 17/17') in text and 'FAIL ' not in text,'corrected temporal negative controls')
        require(any(line.startswith('PASS temporal-ex2-table-wrong-sha:') and 'ex2 half-domain table: size/sha mismatch' in line for line in text.splitlines()),'table rejection actually reached')
    for p,h in material.items(): require(sha(p)==h,'audit-time artifact drift')
    return {'schema':'neuralstudio.nr.connected-temporal-independent-audit.v1','result':'pass_bounded_three_frame_output_history_parity' if version==6 else ('pass_bounded_two_frame_parity_and_third_call_continuity' if version==4 else 'pass_bounded_live_two_frame_output_parity'),'matrix_version':version,'case':case,'fractional_motion':fractional,'changing_images':changing,'spatial_motion':spatial,'frames_per_process':frames,'final_hashes':outputs_by_frame,'runs':rows,'fresh_processes':4,'actual_final_bytes_checked':4*frames*16588800,'encoder_RTX_boundary_hashes_checked':72 if version==6 else 48,'middle_checkpoint_hashes_repeated':4*frames*124,'padding_checkpoint_policy':'Launch57 out_vit retains exactly102400 unwritten poison-padding bytes, verified by its enforced byte-coverage contract; hash compared within each same-poison class, not across poisons. Driver zeroing followed by repack58 and all124 remaining middle hashes are poison-invariant.','model_artifacts_and_receipts_checked':len(material),'artifact_hashes':material,'RTX_reference_root':(rtx3 or rtx)['receipt_root_sha256'],'encoder_reference_root':er['receipt_root_sha256'],'EX2_reference_root':ex2['receipt_root_sha256'],'production_promotion':False,'full_goal_complete':False,'limits':[((('Original image/signed spatial motion[-8,8] at1/16 steps; all three outputs and histories compared to RTX.' if spatial else 'Original image/uniform fractional motion(0.25,-0.5); all three outputs and histories compared to RTX.') if fractional else ('Original->checker->black, zero motion; all three outputs and histories compared to RTX.' if changing else 'One static input recipe/zero motion; all three outputs and histories compared to its RTX reference.')) if version==6 else 'One original 1920x1080 image, zero motion. First two outputs compared to RTX; third call, if present, has continuity/repeat evidence only, pending RTX reference.'),'Encoder boundaries are recorded live hashes; separate accepted producer audit retained actual boundary payloads.','Decoder per-launch pool hashes/coverage omitted in this production-style matrix; strict original-frame evidence remains separate.','Selected negative-control suite passes; injected GPU failure/recovery tests are separately audited, not proved by clean runs.',('EX2 fault and state-advance counters explicitly zero/true in every clean frame.' if version==6 else 'EX2 fault counts are enforced in pinned runtime source but not explicitly serialized in these receipts.'),'No real-time or net-FPS/performance claim.'],'verifier_sha256':sha(Path(__file__))}

def main():
    p=argparse.ArgumentParser(); p.add_argument('--peer-root',type=Path,required=True); p.add_argument('--matrix-version',type=int,choices=(2,3,4,6),default=3); p.add_argument('--case',type=int,choices=(0,1,2),default=0);p.add_argument('--fractional',action='store_true');p.add_argument('--changing',action='store_true');p.add_argument('--spatial',action='store_true');p.add_argument('--output',type=Path,required=True); a=p.parse_args(); require(not a.output.exists(),'preserve prior audit')
    try: r=audit(a.peer_root.resolve(),a.matrix_version,a.case,a.fractional,a.changing,a.spatial)
    except Exception as e: r={'schema':'neuralstudio.nr.connected-temporal-independent-audit.v1','result':'fail','error':str(e),'matrix_version':a.matrix_version,'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r); a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k not in ('artifact_hashes','runs')},indent=2))
    if r['result']=='fail': raise SystemExit(3)
if __name__=='__main__': main()
