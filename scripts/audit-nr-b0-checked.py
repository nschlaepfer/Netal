#!/usr/bin/env python3
"""Audit balanced checked B0 outputs before calculating a scoped timing result."""
import argparse,hashlib,importlib.util,json,math,re,statistics
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
def audit(peer,root):
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('telemetry_audit',repo/'scripts/audit-nr-connected-telemetry.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    reference=helper.sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json','e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657')
    er=helper.sealed(repo/'build/nr_encoder_telemetry/zero-v3/build.json','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
    pins={};records={};pids=set();rows=[];raw_bytes=0
    def observe(p):
        p=Path(p).resolve();h=sha(p);need(str(p) not in pins or pins[str(p)]==h,'artifact drift');pins[str(p)]=h;return h
    def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (peer/p).resolve()
    order=[('v11',1),('v17',1),('v17',2),('v11',2),('v11',3),('v17',3),('v17',4),('v11',4)]
    admission=(root/'admission.log').read_text();observed_order=[(arm,int(pair)) for arm,pair in re.findall(r'admission (v11|v17) p(\d)',admission)]
    need(observed_order==order,'balanced complete eight-process order');need(all(int(v)>=13 for v in re.findall(r'free (\d+) GiB',admission)),'storage admission floor');observe(root/'admission.log');observe(root/'confounds.log')
    patterns=[['a5','5a','a5'],['5a','a5','5a'],['a5']*3,['5a']*3]
    regions={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    for arm,buildname,exe,bodyhash,descname in [('v11','v32t-telemetry-v11','951f18b589d8e0a46f07da1e0e87bf60a978338648d0eae1428cce45a35458fa','4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9','desc-v27t'),('v17','v32t-telemetry','08aa848099d3ee8690b356507a1fb31220d239f350b79525fb5039b9032a4754','64bbe143ffc11373ddc0d796a86d6adf45104113be52022bdc50e9e647dccc1a','desc-v30t')]:
        build=peer/'build/nr_full'/buildname
        for line in (build/'build-sha256.txt').read_text().splitlines():
            h,path=line.split(None,1);need(observe(resolve(path.strip()))==h,'pinned build artifact')
        need(observe(build/'nr_full_frame')==exe and observe(build/f'source/src/nr_body/nr_body_runtime_{arm}.mm')==bodyhash,'reviewed body/executable')
        need(observe(build/'source/src/nr_body/nr_full_frame.mm')=='463e16b1492a398096f2636924823d6b4d86afbb4ecc4cfef354800b8c54ff60','reviewed frame-local timer harness')
        need(observe(build/'source/encoder/nr_encoder_runtime.o')==er['object_sha256']=='942541e17fa3c733605a65813a35a84853ab781ac4501b8695c4f71c35f953de','same encoder object')
        need((build/'build-defines.txt').read_text().strip()=='defines: none','no fault hooks')
        for pair in range(1,5):
            runroot=root/f'{arm}-p{pair}';directory=runroot/f'process-{pair}';s=read(runroot/'run-summary.json');observe(runroot/'run-summary.json')
            recipe=read(runroot/'recipe-as-run.json');need(observe(runroot/'recipe-as-run.json')==s['recipe_sha256'],'as-run recipe identity')
            need(s['all_replicas_passed'] and s['replicas_requested']==s['replicas_passed']==len(s['runs'])==1,'complete admitted run')
            need(s['runs'][0]['exit_code']==0 and s['runs'][0]['passed'] and s['runs'][0]['frame_status']==['comparison_passed']*3,'all three frames passed')
            need(recipe['harness_executable_sha256']==exe and recipe['harness_extra_args']==['--encoder-telemetry','--body-checked'] and recipe['replicas']==[{'tag':f'process-{pair}','poisons':patterns[pair-1]}],'recipe arm/flags/poisons')
            r=read(directory/'receipt.json');need(observe(directory/'receipt.json')==s['runs'][0]['receipt_sha256'],'native receipt hash')
            need(r['schema']=='neuralstudio.nr.full-frame-inmemory.v13' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==3,'native frame inventory')
            need(r['pid'] not in pids and r['pid']>0 and r['contexts_prepared_once'] and r['context_preparations']==1,'fresh process and persistent context');pids.add(r['pid'])
            cmd=r['command'];need(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==exe,'executed arm')
            desc=resolve(cmd[1]);need(desc==peer/'build/nr_body'/descname/'chain.json' and observe(desc)==recipe['descriptor_sha256']==r['body_descriptor_sha256'],'descriptor identity')
            descriptor=read(desc);canonical_desc=read(peer/'build/nr_body/desc-v27t/chain.json')
            if arm=='v17':need(descriptor.pop('note_b0',None) is not None and descriptor['decoder'].pop('write_pool_post',None)=='true','equal diagnostic pool retention')
            need(descriptor==canonical_desc,'only reviewed diagnostic descriptor difference')
            need(cmd[cmd.index('--reset-sequence')+1]=='1,0,0' and cmd[cmd.index('--encoder-sampler-variant')+1]=='0','same reset/motion sampler')
            need(observe(resolve(cmd[cmd.index('--motion')+1]))=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','zero motion bytes')
            need('--body-checked' in cmd and '--encoder-telemetry' in cmd and '--encoder-diagnostics' not in cmd,'actual checked instrumentation')
            previous_elapsed=0
            for i,f in enumerate(r['frames']):
                e,b=f['encoder'],f['body'];want=reference['outputs'][i]
                need(f['frame']==i and f['poison']==int(patterns[pair-1][i],16) and f['body_mode']=='checked' and f['status']=='comparison_passed' and f['result']=='pass','actual frame configuration/status')
                need(e['code']==b['code']==0 and e['gpu_complete'] and e['input_immutable'] and e['reset']==(i==0) and e['frame_index']==i,'encoder/body execution')
                need(e['guarded_outputs_checked']==(1039 if i==0 else 1040) and e['coverage_values_checked']==(5900107776 if i==0 else 5935497216),'unchanged encoder coverage')
                need(b['checks_skipped']==[] and not b['check_failures'] and b['constants_unchanged_after_frame'] and b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'],'all required checks active')
                need(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'full output coverage')
                need(f['temporal_state_advanced'] and f['frames_since_reset_after']==i+1 and f['history_copy_value_differences']==0 and f['test_fault']=='','history state progression')
                need(f['ex2_half_faults']=={'argument':0,'domain':0},'EX2 guards')
                need(f['input_sha256']==e['input_sha256']==observe(resolve(f['input_file']))=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','same original input')
                need(f['surface_sha256_caller_readback']==b['final_surface_sha256']==b['history_sha256_after']==want,'final/history RTX identity')
                if i:need(b['history_sha256_before']==reference['outputs'][i-1],'own previous history')
                for key,target in regions.items():need(f['regions'][key]['sha256']==reference['boundary_ports'][2*i+1][target],'RTX encoder port')
                need(f['reference']['cpu_comparison_only'] and f['reference']['mismatching_values']==0 and observe(resolve(f['reference']['file']))==want,'reference only used for comparison')
                for path in (directory/f'final-{i}.rgba16le',directory/f'diag/frame-{i}/tex_out_155.bin',directory/f'diag/frame-{i}/tex_out_156.bin'):
                    need(path.stat().st_size==16588800 and observe(path)==want,'actual output/post/history bytes');raw_bytes+=path.stat().st_size
                pool=directory/f'diag/frame-{i}/pool_post.bin';need(pool.stat().st_size==canonical_desc['pool_bytes'] and observe(pool)==b['pool_final_sha256'],'same retained full pool');raw_bytes+=pool.stat().st_size
                cp=directory/f'diag/frame-{i}/chain-run.json';c=read(cp);observe(cp);need(c['pid']==r['pid'] and c['poison']==f['poison'],'same-process checkpoints')
                need(f['encoder_telemetry_enabled'] and f['encoder_telemetry_error']=='','telemetry available');helper.validate_telemetry(f['encoder_telemetry'],f,r['pid'],i)
                wall=f['frame_wall_ms'];elapsed=f['process_elapsed_ms']
                need('wall_ms_total' not in f and math.isfinite(wall) and wall>0 and math.isfinite(elapsed) and elapsed>=wall and elapsed>=previous_elapsed,'frame-local and cumulative clock separation');previous_elapsed=elapsed
                need(wall+1>=e['frame_ms']+b['wall_ms'] and 'excludes telemetry copy' in f['frame_wall_boundary'],'timing envelope')
                records[arm,pair,i]=(f,c)
            need(r['files_unchanged_after_run'],'run source/assets unchanged')
            for path,h in r['post_run_file_identity'].items():need(h['unchanged'] and observe(resolve(path))==h['post']==h.get('pre',h.get('load')),'post-run file identities')
    checkpoints=0
    for pair in range(1,5):
        for i in range(3):
            a,c0=records['v11',pair,i];b,c1=records['v17',pair,i]
            need(a['regions']==b['regions'],'same returned region semantics')
            old={(x['launch'],k):v for x in c0['launches'] if x['kind']!='ptx' for k,v in x['outputs'].items()};need(len(old)==125,'all baseline middle hashes')
            for x in c0['launches']:
                if x['kind']=='ptx':old[x['launch'],'pool']=x['pool_post_sha256']
            new={(x['launch'],k[:-7]):v for x in c1['launches'] for k,v in x['digests'].items() if k.endswith('_sha256')}
            need(len(old)==149 and old.keys()<=new.keys() and all(old[k]==new[k] for k in old),'all paired middle/pool checkpoints')
            need(c1['decoder_pre_image_source']=='gpu-snapshot' and c1['decoder_pre_image_sha256']==c1['decoder_pre_image_expected_sha256']==c0['decoder']['pool_pre_sha256'],'actual pre-image equality')
            need(c1['pool_final_sha256']==c0['decoder']['pool_final_sha256'] and c1['tex155_output_sha256']==next(x for x in c0['launches'] if x['launch']==155)['texture']['output_sha256'],'remaining boundary identities')
            need(c1['validators']=={'expected':150,'run':150,'faults':0} and b['inventory']['command_buffers']==b['inventory']['waits']==1,'resident coverage and inventory')
            checkpoints+=152
            rows.append({'pair':pair,'frame':i,'class':'reset-first' if i==0 else 'temporal-short','baseline_pid':a['encoder_telemetry']['pid'],'resident_pid':b['encoder_telemetry']['pid'],'baseline_frame_ms':a['frame_wall_ms'],'resident_frame_ms':b['frame_wall_ms'],'baseline_encoder_ms':a['encoder']['frame_ms'],'resident_encoder_ms':b['encoder']['frame_ms'],'baseline_body_ms':a['body']['wall_ms'],'resident_body_ms':b['body']['wall_ms']})
    need(len(pids)==8 and checkpoints==1824 and len(rows)==12,'complete paired study')
    # Independent process-pair units: average the two temporal steps WITHIN
    # each process before computing cross-process summaries; no pseudo-n=8.
    stats={}
    for kind in ('reset-first','temporal-short'):
        pairs=[]
        for pair in range(1,5):
            rs=[x for x in rows if x['pair']==pair and x['class']==kind];a=statistics.mean(x['baseline_frame_ms'] for x in rs);b=statistics.mean(x['resident_frame_ms'] for x in rs)
            pairs.append({'pair':pair,'baseline_ms':a,'resident_ms':b,'reduction_percent':100*(1-b/a)})
        stats[kind]={'pairs':pairs,'baseline_median_ms':statistics.median(x['baseline_ms'] for x in pairs),'resident_median_ms':statistics.median(x['resident_ms'] for x in pairs),'paired_median_reduction_percent':statistics.median(x['reduction_percent'] for x in pairs),'pairs_resident_faster':sum(x['resident_ms']<x['baseline_ms'] for x in pairs)}
    for p,h in pins.items():need(sha(p)==h,'audit-time drift')
    return {'schema':'neuralstudio.nr.b0-checked-independent.v1','result':'pass_checked_correctness_with_scoped_timing','fresh_processes':8,'frames':24,'paired_checkpoint_identities':checkpoints,'raw_surface_history_pool_bytes_checked':raw_bytes,'rows':rows,'timing':stats,'pins':pins,'default_promoted':False,'full_goal_complete':False,'limits':['One original-image zero-motion reset[1,0,0] workload, four process pairs; no tail/long-run/game-FPS inference.','Frame-local envelope includes input/readback/hash/reference comparison, final saves and body diagnostics, excluding telemetry copy and final process receipt output.','Encoder prepare happens before do_frame; first reset-frame timing is not complete process-startup latency.','Body v11 per-launch GPU sums and v17 whole-buffer GPU time cover different populations; not compared as equivalent.','Agent quiet window coordinated; confounds log has load/thermal/memory snapshots, but first-sample top reports zeros and does not prove machine-wide idleness.','No population confidence interval or performance promotion from this small pilot.']}
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve audit')
    try:r=audit(a.peer.resolve(),a.root.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.b0-checked-independent.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(__file__);r['receipt_root_sha256']=canonical(r);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('pins','rows')},indent=2));return int(r['result']=='fail')
if __name__=='__main__':raise SystemExit(main())
