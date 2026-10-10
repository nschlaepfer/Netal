#!/usr/bin/env python3
"""Independent pinned resident output/port/history audit; checked-pair acceptance is separate."""
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
 'combined3':('build/nr_combined_sequence/packet-audit-v1.json','8fc7129e7dfa5b2457707796319bbc57c0706cf3ddb32f99fe7ae2a544fa4a06'),
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
    fractional=name in ('uniform3','spatial3','combined3')
    return rtx,n,resets,images,motion,fractional

def backend_contract(version,fractional):
    if version==13:
        return ('v27t-sp' if fractional else 'v27t',
                '72f3e9b8872ba191b4ea970498cc1e359139805491724ff671803d7acc646188',
                '44f427ceea9d63c1ab49d1ecc6fd9a7ba4ed2e19f129fb17bea21e1301233091',10)
    require(version==17,'unreviewed resident build/variant')
    return ('v31t-sp' if fractional else 'v31t','64bbe143ffc11373ddc0d796a86d6adf45104113be52022bdc50e9e647dccc1a',
            '2d98d752b33fb505b53c91aaea34122539207c1ab3ffa5c07bca37e4bad6c65f',12)

def audit(peer,matrix,name,version=13):
    repo=Path(__file__).resolve().parents[1]
    require(matrix.is_relative_to(peer/'build/nr_full'),'scoped peer matrix')
    rtx,n,resets,images,motion_hash,fractional=configuration(repo,name)
    build_name,body_sha,harness_sha,harness_schema=backend_contract(version,fractional)
    build=peer/'build/nr_full'/build_name
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
    require(observe(build/f'source/src/nr_body/nr_body_runtime_v{version}.mm')==body_sha,'reviewed body')
    require(observe(build/'source/src/nr_body/nr_full_frame.mm')==harness_sha,'reviewed harness')
    if version==17:require(observe(build/'source/src/nr_body/nr_body_runtime.h')=='89948e693cff987eb22a883d26368e0bb738ba8c41378e8e399909e2c0d2f917','reviewed unique-context header')
    require(observe(build/'source/src/nr_temporal/nr_temporal.h')==observe(enc/'source/nr_temporal.h')=='d44c36cff1f40ea7574619a87a8d743048ca4f30792a7a771c6b2dd6011cb28d','pinned shared temporal ABI')
    require((build/'build-defines.txt').read_text().strip()=='defines: none','no test hooks')
    for asset_name,h in eb['assets'].items():require(observe(enc/'assets'/asset_name)==h,'encoder asset')
    regions={'entry':('block22-middle',1105920),'skip_c256':('block22-native',2211840),'skip_c128':('block14-native',4423680),'skip_c64':('block8-native',8847360),'skip_c32':('block4-native',17694720),'block0_skip':('block0-skip',70778880)}
    rows=[];pids=set();stage_by_frame={};padding={}
    configs=[('replica-1-pilot',([165,90]*n)[:n]),('replica-2',[165]*n),('replica-3',[90]*n),('replica-4',([90,165]*n)[:n])]
    if name=='combined3':configs[0]=('pilot',configs[0][1])
    for label,poisons in configs:
        d=matrix/label;r=read(d/'receipt.json');rh=observe(d/'receipt.json')
        require(r['schema']==f'neuralstudio.nr.full-frame-inmemory.v{harness_schema}' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==n,'all declared native calls')
        require(r['pid']>0 and r['pid'] not in pids,'fresh native process');pids.add(r['pid'])
        require(r['context_preparations']==1 and r['contexts_prepared_once'] and r['files_unchanged_after_run'],'one persistent context')
        cmd=r['command'];require(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==observe(build/'nr_full_frame'),'executed binary')
        require(not any(x.startswith('--encoder-diagnostics') for x in cmd),'encoder diagnostics disabled')
        require(resolve(cmd[1])==peer/('build/nr_body/desc-v28t/chain.json' if fractional else 'build/nr_body/desc-v27t/chain.json') and observe(resolve(cmd[1]))==r['body_descriptor_sha256'],'descriptor')
        descriptor=read(resolve(cmd[1]));original_descriptor=read(peer/('build/nr_body/desc-v24t/chain.json' if fractional else 'build/nr_body/desc-v23t/chain.json'))
        model_descriptor={k:v for k,v in descriptor.items() if k not in ('resident_lib','resident_lib_sha256','note_v13')}
        require(model_descriptor==original_descriptor,'unchanged model descriptor')
        require(observe(resolve(descriptor['resident_lib']))==descriptor['resident_lib_sha256']=='f8dc47fc056254505ccf4a6783ae0bccca41c9a75ec078e02be1840649961351','resident validation library pin')
        require(resolve(cmd[2])==enc/'assets' and cmd[cmd.index('--reset-sequence')+1]==','.join(map(str,resets)),'assets and reset mask')
        motion=resolve(cmd[cmd.index('--motion')+1]);require(motion.stat().st_size==8294400 and observe(motion)==motion_hash,'pinned motion input')
        state=0
        for i,f in enumerate(r['frames']):
            before=state;reset=bool(resets[i]);step=0 if reset else state;state=step+1;e,b,t=f['encoder'],f['body'],f['temporal'];expected=rtx['outputs'][i]
            require(f['frame']==i and f['poison']==poisons[i] and f['result']=='pass' and not f['plumbing_only_no_parity_meaning'],'frame/poison')
            if version==17:require(f['body_mode']=='production','declared resident production mode')
            require(e['code']==b['code']==0 and not e['error'] and not b['error'] and e['gpu_complete'] and e['input_immutable'],'live APIs')
            require(e['diagnostics_dir']=='','no archived encoder diagnostics ingress')
            require(t['enabled'],'shared temporal state enabled')
            require(e['reset']==t['body_reset']==t['encoder_reset']==reset and e['frame_index']==step and e['entry']==('nr_encoder_run_frame' if reset else 'nr_encoder_run_temporal_frame'),'reset and live producer selection')
            require(t['frames_since_reset_before']==before and b['frames_since_reset']==step+1 and f['frames_since_reset_after']==step+1,'counter reset/progression')
            require(f['temporal_state_advanced'] and f['ex2_half_faults']=={'argument':0,'domain':0} and f['history_copy_value_differences']==0 and f['test_fault']=='','clean state commit')
            require(e['guarded_outputs_checked']==(1039 if reset else 1040) and e['coverage_values_checked']==(5900107776 if reset else 5935497216),'encoder coverage')
            ih=observe(resolve(f['input_file']));require(ih==images[i] and ih==e['input_sha256']==f['input_sha256']==b['input_texture_sha256_before']==b['input_texture_sha256_after'],'exact declared immutable image')
            require(b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'] and not b['check_failures'] and b['constants_unchanged_after_frame'],'immutability/checks')
            require(b['checks_enforced']==(416 if reset else 426) and b['constants_rehashed_after_frame']==(153 if i==0 else 154),'check counts and warm constants')
            require(b['checks_skipped']==['decoder per-launch pool digest','decoder per-launch pool coverage validator','intermediate SHA-256 snapshots (checked mode only; GPU digests + boundary SHA-256 in production)','decoder pre-image actual GPU snapshot SHA-256 (production reports the CPU reconstruction with the GPU digest cross-check)'],'only declared strict-checkpoint omissions')
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
            require(len(b['loaded'])==(161 if i==0 else (163 if reset else 164)),'complete warm resource inventory')
            for path in (d/f'final-{i}.rgba16le',d/f'diag/frame-{i}/tex_out_155.bin',d/f'diag/frame-{i}/tex_out_156.bin'):
                require(path.stat().st_size==16588800 and observe(path)==expected,'actual final/post/history bytes')
            require(f['reference']['cpu_comparison_only'] and not f['no_reference_no_parity_meaning'] and f['reference']['mismatching_values']==0 and f['reference']['sha256']==expected,'actual reference comparison required')
            reference=resolve(f['reference']['file']);require(reference.stat().st_size==16588800 and observe(reference)==expected,'actual RTX reference bytes')
            cp=d/f'diag/frame-{i}/chain-run.json';chain=read(cp);observe(cp);require(chain['pid']==r['pid'] and chain['poison']==poisons[i] and len(chain['launches'])==105,'same-process chain')
            evidence=f['resident_evidence'];inventory=f['inventory']
            require(inventory['command_buffers']==inventory['waits']==1,'single committed command buffer')
            require(evidence['validators_expected']==evidence['validators_run']==126 and evidence['coverage_faults']==0,'resident coverage validation')
            require(evidence['digests_expected']==evidence['digests_complete']==(131 if reset else 135) and evidence['digest_cpu_mismatches']==0,'supplemental digest completion')
            require(evidence['decoder_pre_image_source']=='cpu-reconstruction (+gpu-digest-crosscheck)','production preimage scope explicit')
            require(chain['schema']==f'neuralstudio.nr.body-runtime.v{version}-resident-experimental' and chain['command_buffers']==1,'resident chain identity')
            require(chain['validators']=={'expected':126,'run':126,'faults':0},'chain coverage result')
            validators=[v for l in chain['launches'] for v in l['validators']]
            require(len(validators)==126 and sorted(v['slot'] for v in validators)==list(range(126)),'unique complete validator inventory')
            require(all(v['elements']>0 and v['done'] and v['faults']==0 for v in validators),'validator completion records')
            require(len(chain['digest_table'])==(131 if reset else 135) and all(d['done'] for d in chain['digest_table']),'supplemental digest inventory')
            require(f['status']=='comparison_passed' and not f['encoder_telemetry_enabled'] and f['encoder_telemetry'] is None,'ordinary correctness run, no telemetry substitution')
            # 64-bit production digests do not replace old strong intermediate
            # identities; corresponding checked live pairs remain a separate gate.
            rows.append({'pid':r['pid'],'frame':i,'poison':poisons[i],'reset':reset,'output_and_history_sha256':expected,'native_receipt_sha256':rh})
        for p,m in r['post_run_file_identity'].items():require(m['unchanged'] and m['post']==m.get('pre',m.get('load'))==observe(p),'post-run file identity')
    for p,h in pins.items():require(sha(p)==h,'audit-time drift')
    return {'schema':f'neuralstudio.nr.resident-v{version}-live-audit.v1','body_version':version,'result':'pass_live_outputs_ports_history_with_explicit_omissions','configuration':name,'matrix':str(matrix),'reference_root':rtx['receipt_root_sha256'],'encoder_root':eb['receipt_root_sha256'],'runs':rows,'actual_final_post_history_bytes_checked':4*n*3*16588800,'encoder_boundary_hashes_checked':4*n*6,'artifacts':pins,'strong_live_intermediate_acceptance':False,'full_goal_complete':False,'production_promotion':False,'limits':['This production-style mode omits strong intermediate and decoder-pool snapshots; corresponding live checked pairs remain required.','The preimage SHA is explicitly expected-side reconstruction, not an actual GPU snapshot in this mode.','64-bit digests are supplemental only; no cross-poison equality is inferred for poison-dependent padding/pools.','No speedup or end-to-end performance claim.'],'verifier_sha256':sha(Path(__file__))}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer-root',type=Path,required=True);p.add_argument('--matrix',type=Path,required=True);p.add_argument('--configuration',choices=REFERENCES,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--body-version',type=int,choices=(13,17),default=13);a=p.parse_args();require(not a.output.exists(),'preserve audit')
    try:r=audit(a.peer_root.resolve(),a.matrix.resolve(),a.configuration,a.body_version)
    except Exception as e:r={'result':'fail','configuration':a.configuration,'error':str(e),'verifier_sha256':sha(Path(__file__))}
    r['schema']=f'neuralstudio.nr.resident-v{a.body_version}-live-audit.v1'
    r['body_version']=a.body_version
    r['receipt_root_sha256']=canonical(r);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','artifacts')},indent=2))
    if r['result']=='fail':raise SystemExit(3)
if __name__=='__main__':main()
