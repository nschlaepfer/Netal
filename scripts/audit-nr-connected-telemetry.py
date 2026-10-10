#!/usr/bin/env python3
"""Independent connected telemetry on/off parity audit, not a latency benchmark."""
import argparse,hashlib,json,math
from pathlib import Path
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def need(ok,why):
    if not ok:raise ValueError(why)
def sealed(p,expected):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');need(canonical(b)==h==expected,'sealed parent');return r
def validate_telemetry(t,frame,pid,index):
    need(isinstance(t,dict) and t['schema']=='neuralstudio.nr.encoder-timing.v2','telemetry schema')
    need(t['pid']==pid and t['generation']==index+1 and t['frame_index']==index and t['reset']==(index==0),'context-local telemetry identity')
    need(t['input_sha256']==frame['input_sha256'],'telemetry belongs to current frame')
    need(t['serialization_in_copy_accessor'] and t['guards_unchanged'] and t['input_blits_included'] and not t['shader_changes'] and not t['tensor_diagnostics_enabled'],'telemetry invariants')
    commands=[x for x in t['records'] if x['command']]
    need(len(commands)==(139 if index==0 else 140) and sum(x['label']=='encoder.input_texture_blit' for x in commands)==2,'complete encoder command inventory including input blits')
    end=0
    for c in commands:
        need(c['gpu_timestamp_valid'] and all(math.isfinite(c[k]) for k in ('gpu_start_s','gpu_end_s','gpu_ms','submit_wait_ms')),'valid finite GPU timestamps')
        need(c['gpu_start_s']>0 and c['gpu_start_s']>=end-1e-6 and c['gpu_end_s']>=c['gpu_start_s'],'ordered GPU intervals')
        need(math.isclose(c['gpu_ms'],1000*(c['gpu_end_s']-c['gpu_start_s']),rel_tol=1e-8,abs_tol=1e-6),'GPU unit conversion')
        end=c['gpu_end_s']
    need(t['phases'] and all(math.isfinite(x['wall_ms']) and x['wall_ms']>=0 for x in t['phases']),'phase records')
    return len(commands)
def audit(peer):
    repo=Path(__file__).resolve().parents[1];matrix=peer/'build/nr_full/telemetry-parity-v1';build=peer/'build/nr_full/v29t-telemetry-v11';enc=repo/'build/nr_encoder_telemetry/zero-v3'
    er=sealed(enc/'build.json','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
    ref=sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json','e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657')
    pins={}
    def observe(p):
        p=Path(p).resolve();h=sha(p);need(str(p) not in pins or pins[str(p)]==h,'artifact drift');pins[str(p)]=h;return h
    def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (peer/p).resolve()
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,p=line.split(None,1);need(observe(resolve(p.strip()))==h,'linked build identity')
    need(observe(build/'source/encoder/nr_encoder_runtime.o')==er['object_sha256']=='942541e17fa3c733605a65813a35a84853ab781ac4501b8695c4f71c35f953de','linked reviewed telemetry object')
    need(observe(build/'source/src/nr_body/nr_full_frame.mm')=='645552aab97f0c69ab60ab053f11f34fad904a83a64d09228bd71329e8c08c0c','reviewed harness')
    need(observe(build/'source/src/nr_body/nr_body_runtime_v11.mm')=='4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9','same exact body')
    for name,h in er['assets'].items():need(observe(enc/'assets'/name)==h,'encoder assets')
    regions={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    pids=set();frames={};commands=0
    for side in ('off','on'):
        for rep in (1,2):
            d=matrix/side/f'{side}-{rep}';r=read(d/'receipt.json');observe(d/'receipt.json')
            need(r['schema']=='neuralstudio.nr.full-frame-inmemory.v11' and r['result']=='pass' and r['frames_in_process']==len(r['frames'])==3,'complete connected sequence')
            need(r['pid'] not in pids and r['contexts_prepared_once'] and r['context_preparations']==1 and r['files_unchanged_after_run'],'fresh persistent context');pids.add(r['pid'])
            cmd=r['command'];need(resolve(cmd[0])==build/'nr_full_frame' and observe(build/'nr_full_frame')==r['executable_sha256']=='31d011ebdc566ebf839570f4a8ef09ead2ea6d784953626dc4f0eaee267e428b','executed binary')
            need(('--encoder-telemetry' in cmd)==(side=='on') and '--body-checked' not in cmd and '--encoder-diagnostics' not in cmd,'requested instrumentation mode')
            need(resolve(cmd[2])==enc/'assets' and r['encoder_asset_root']==str(enc/'assets'),'actual encoder assets')
            need(observe(resolve(cmd[1]))==r['body_descriptor_sha256']=='98654e30f40bd13dbe92e28f8c9827189614e7b5662195504e2b1f45e14c95bb','descriptor pin')
            need(cmd[cmd.index('--reset-sequence')+1]=='1,0,0' and observe(resolve(cmd[cmd.index('--motion')+1]))=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','history sequence/motion')
            for i,f in enumerate(r['frames']):
                e,b=f['encoder'],f['body'];pz=165 if (i+rep)%2 else 90;wanted=ref['outputs'][i]
                need(f['frame']==i and f['poison']==pz and f['status']=='comparison_passed' and f['result']=='pass' and e['code']==b['code']==0,'frame status/poison')
                need(f['input_sha256']==e['input_sha256']==observe(resolve(f['input_file']))=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','immutable input identity')
                need(e['reset']==(i==0) and e['frame_index']==i and e['gpu_complete'] and e['input_immutable'],'encoder execution')
                need(e['guarded_outputs_checked']==(1039 if i==0 else 1040) and e['coverage_values_checked']==(5900107776 if i==0 else 5935497216),'encoder coverage')
                need(f['body_mode']=='production' and b['checks_enforced']==(242 if i==0 else 246) and not b['check_failures'],'body check scope')
                need(b['checks_skipped']==['decoder per-launch pool hash','decoder per-launch pool coverage scan'],'explicit production omissions')
                need(b['input_texture_immutable'] and b['constants_unchanged_after_frame'] and f['producer_regions_unchanged_by_body'],'input/constant/producer integrity')
                need(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'full final coverage')
                need(f['temporal_state_advanced'] and f['frames_since_reset_after']==i+1 and f['history_copy_value_differences']==0,'live history advance')
                need(f['ex2_half_faults']=={'argument':0,'domain':0} and f['test_fault']=='','no hidden faults')
                need(b['history_sha256_after']==b['final_surface_sha256']==f['surface_sha256_caller_readback']==wanted,'frame/history reference')
                if i:need(b['history_sha256_before']==ref['outputs'][i-1],'live previous history')
                for key,target in regions.items():need(f['regions'][key]['sha256']==ref['boundary_ports'][2*i+1][target],'RTX producer port')
                need(f['reference']['cpu_comparison_only'] and f['reference']['mismatching_values']==0,'CPU-only reference comparison')
                for p in (d/f'final-{i}.rgba16le',d/f'diag/frame-{i}/tex_out_155.bin',d/f'diag/frame-{i}/tex_out_156.bin',resolve(f['reference']['file'])):need(p.stat().st_size==16588800 and observe(p)==wanted,'actual frame/post/history/reference bytes')
                need(f['encoder_telemetry_enabled']==(side=='on') and f['encoder_telemetry_error']=='','telemetry request/result')
                if side=='on':commands+=validate_telemetry(f['encoder_telemetry'],f,r['pid'],i)
                else:need(f['encoder_telemetry'] is None,'off mode publishes no stale snapshot')
                frames[side,rep,i]=f
            for path,meta in r['post_run_file_identity'].items():need(meta['unchanged'] and observe(resolve(path))==meta['post']==meta.get('pre',meta.get('load')),'post-run provenance')
    for rep in (1,2):
        for i in range(3):
            a,b=frames['off',rep,i],frames['on',rep,i]
            need(a['regions']==b['regions'] and a['temporal']==b['temporal'],'on/off ports and temporal semantics')
            for field,drop in [('encoder',{'prepare_ms','frame_ms'}),('body',{'wall_ms','middle_gpu_ms','decoder_gpu_ms'})]:
                need({k:v for k,v in a[field].items() if k not in drop}=={k:v for k,v in b[field].items() if k not in drop},'on/off semantic receipt equality '+field)
    for path,h in pins.items():need(sha(path)==h,'audit-time drift')
    return {'schema':'neuralstudio.nr.connected-telemetry-independent.v1','result':'pass_on_off_parity','encoder_root':er['receipt_root_sha256'],'rtx_reference_root':ref['receipt_root_sha256'],'fresh_processes':4,'frames':12,'paired_frames':6,'actual_final_post_history_bytes_checked':597196800,'RTX_port_hashes_checked':72,'telemetry_command_records_checked':commands,'pins':pins,'quiet_measurement_accepted':False,'speedup_accepted':False,'limits':['Only original-image, zero-motion reset[1,0,0] case; body v11 with explicit production-mode omissions.','Old frame wall_ms_total is cumulative process elapsed, not per-frame latency; do not benchmark it or difference it.','Body inventory zeros in this build mean uninstrumented, not zero commands/cost.','Telemetry publication parity is accepted; a corrected frame-local clock and agreed quiet window remain required for baseline-0.']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve audit')
    try:r=audit(a.peer.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.connected-telemetry-independent.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(__file__);r['receipt_root_sha256']=canonical(r);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k!='pins'},indent=2));return int(r['result']=='fail')
if __name__=='__main__':raise SystemExit(main())
