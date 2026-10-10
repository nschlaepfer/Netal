#!/usr/bin/env python3
"""Audit matched production outputs and telemetry BEFORE summarizing paired timings."""
import argparse,hashlib,importlib.util,json,math,re,statistics
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
RECIPE_PINS={
'v11':['a86abc738169c29f6c4a81b4f71515518f1e58e5df41635e8e5376740d87d85b','4fb536106fd086f8a5d7d5b3f9e501fec12f78599ab3c99b7844d5a9ca49e077','5a0b3f94da3961547c0b0aed9459967796b0be5bb4c422d19ad727d405931726','be9b0358d7c689e62d179a43a19f1a30f41d76ca65e07ffbb7fa778d5f074485'],
'v17':['df4bf85d482ba0a52cf4f2d098975dae621bdb712988ea6b376b8adbec990bce','5749b6911d8dbb8949a13e408428d0edfe6aaf06887d554f0677d361811cd2f2','81007d6cf12f77c2e2651dc24c5d4516297b9df493ac48cf0cb22cd7a7458395','fcea065a6018025fdb7456235857ba3f6fc327bf756ecc9a0cba2864b53fb3ac']}
OMISSIONS={
'v11':['decoder per-launch pool hash','decoder per-launch pool coverage scan'],
'v17':['decoder per-launch pool digest','decoder per-launch pool coverage validator','intermediate SHA-256 snapshots (checked mode only; GPU digests + boundary SHA-256 in production)','decoder pre-image actual GPU snapshot SHA-256 (production reports the CPU reconstruction with the GPU digest cross-check)']}
def audit(peer,root):
    repo=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('telemetry_audit',repo/'scripts/audit-nr-connected-telemetry.py');helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    reference=helper.sealed(repo/'build/nr_three_frame_history/matrix-audit-v1.json','e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657')
    er=helper.sealed(repo/'build/nr_encoder_telemetry/zero-v3/build.json','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
    constants=helper.sealed(repo/'build/nr_resident_review/recovered-constants-v2.json','550fbdbecf0fda9d1a70b28daf5ca3a64aa6fb18fbd11c81949444474253e170')
    expected_files={x['current_path']:x['expected_sha256'] for x in constants['artifacts']}
    tables=helper.sealed(repo/'build/nr_resident_review/recovery-tables-v2.json','0f235fa0b89ab64582c51963e389b97838866f6f0a258bcb975fd39f257e2024')
    for name,t in tables['tables'].items():
        p=peer/('build/nr_ex2_half/table-v1/'+name if name=='ex2-scaled-half.u32x2' else 'build/nr_ptx/tables/'+name)
        expected_files[str(p)]=t['sha256']
    libs={
      'build/nr_ptx/recovery-v17/dec.metallib':'bc10c0e9a68342bdb12d84835ebf22b1831e5b9eca7f6b35f2c52b5422401e6a',
      'build/nr_ptx/recovery-v17/post.metallib':'449f5afe1200a7eb2f425f04d0df68c33cc412a94f3f70ea1fa532d78699fdec',
      'build/nr_ptx/recovery-v23t/post.metallib':'07e322743630007fb8a9e51e78c566b86fa111a8a60c1ea16a52c45544c68297',
      'build/nr_ptx/recovery-v23t/copy.metallib':'343f9f9b0d857c1549c8fc0c78ca2c96128cc014678a486deefbf22c50c42042',
      'build/nr_split/verify2-v13/split.metallib':'41e21caf46a292cc191c63f75c4c7ce95e649c889e167933a6d969f4e6e130e7',
      'build/nr_vit/verify2-v8/vit.metallib':'79eca1898fbc2c9bf04f9c7130627bb5912e458756e22cbeaf51e011901cf467',
      'build/nr_tp/verify-v5/tp.metallib':'3fb0b9b91a616dcdb0a216788269e0c9b98a01ff4ea66028d54e99f2330352a3',
      'build/nr_vit/verify-tp-v8/tp.metallib':'4461be329122c04e07267966fee6a2648a3601c366dd2b177940ada8acfcf171',
      'build/nr_body/verify2-resident/validate.metallib':'f8dc47fc056254505ccf4a6783ae0bccca41c9a75ec078e02be1840649961351'}
    expected_libs={str(peer/p):h for p,h in libs.items()}
    pins={};records={};pids=set();rows=[];raw_bytes=0;telemetry_commands=0
    def observe(p):
        p=Path(p).resolve();h=sha(p);need(str(p) not in pins or pins[str(p)]==h,'artifact drift');pins[str(p)]=h;return h
    def resolve(p):return Path(p).resolve() if Path(p).is_absolute() else (peer/p).resolve()
    order=[('v11',1),('v17',1),('v17',2),('v11',2),('v11',3),('v17',3),('v17',4),('v11',4)]
    admission=(root/'admission.log').read_text();observed_order=[(arm,int(pair)) for arm,pair in re.findall(r'admission (v11|v17) p(\d)',admission)]
    need(observed_order==order,'balanced complete eight-process order');need(len(re.findall(r'free (\d+) GiB',admission))==8 and all(int(v)>=15 for v in re.findall(r'free (\d+) GiB',admission)),'3GiB reserve above12GiB floor');observe(root/'admission.log');observe(root/'confounds.log')
    need(observe(peer/'scripts/nr-baseline-prod.sh')=='0eb862fdef96e26ea79a95d5596601dfaffe74c576c694b5d2b7e5b33ea8bff4','reviewed driver')
    confounds=(root/'confounds.log').read_text();need(confounds.count('[sample 2] Processes:')==8,'eight actual second activity samples')
    patterns=[['a5','5a','a5'],['5a','a5','5a'],['a5']*3,['5a']*3]
    regions={'entry':'block22-middle','skip_c256':'block22-native','skip_c128':'block14-native','skip_c64':'block8-native','skip_c32':'block4-native','block0_skip':'block0-skip'}
    for arm,buildname,exe,bodyhash,descname in [('v11','recovery-v11','951f18b589d8e0a46f07da1e0e87bf60a978338648d0eae1428cce45a35458fa','4a55d1d818f064e5d597ad9b1e9f69e3cc48321608e0f910e8504a4317020ec9','desc-recovery-v1'),('v17','recovery-v17','08aa848099d3ee8690b356507a1fb31220d239f350b79525fb5039b9032a4754','64bbe143ffc11373ddc0d796a86d6adf45104113be52022bdc50e9e647dccc1a','desc-recovery-v1')]:
        build=peer/'build/nr_full'/buildname
        for line in (build/'build-sha256.txt').read_text().splitlines():
            h,path=line.split(None,1);need(observe(resolve(path.strip()))==h,'pinned build artifact')
        need(observe(build/'nr_full_frame')==exe and observe(build/f'source/src/nr_body/nr_body_runtime_{arm}.mm')==bodyhash,'reviewed body/executable')
        need(observe(build/'source/src/nr_body/nr_full_frame.mm')=='463e16b1492a398096f2636924823d6b4d86afbb4ecc4cfef354800b8c54ff60','reviewed frame-local timer harness')
        need(observe(build/'source/encoder/nr_encoder_runtime.o')==er['object_sha256']=='942541e17fa3c733605a65813a35a84853ab781ac4501b8695c4f71c35f953de','same encoder object')
        need((build/'build-defines.txt').read_text().strip()=='defines: none','no fault hooks')
        for pair in range(1,5):
            runroot=root/f'{arm}-p{pair}';directory=runroot/f'process-{pair}';s=read(runroot/'run-summary.json');observe(runroot/'run-summary.json')
            recipe=read(runroot/'recipe-as-run.json');need(observe(runroot/'recipe-as-run.json')==s['recipe_sha256']==RECIPE_PINS[arm][pair-1],'reviewed as-run recipe identity');need(s['runner_sha256']=='340f4c96a500e856a8c722b3dd7ce1c17814efb5b3b7bf3a5da07d5b372df156','reviewed recipe runner')
            need(s['all_replicas_passed'] and s['replicas_requested']==s['replicas_passed']==len(s['runs'])==1,'complete admitted run')
            need(s['runs'][0]['exit_code']==0 and s['runs'][0]['passed'] and s['runs'][0]['frame_status']==['comparison_passed']*3,'all three frames passed')
            need(recipe['harness_executable_sha256']==exe and recipe['harness_extra_args']==['--encoder-telemetry'] and recipe['replicas']==[{'tag':f'process-{pair}','poisons':patterns[pair-1]}],'recipe arm/flags/poisons')
            r=read(directory/'receipt.json');need(observe(directory/'receipt.json')==s['runs'][0]['receipt_sha256'],'native receipt hash')
            need(r['schema']=='neuralstudio.nr.full-frame-inmemory.v13' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==3,'native frame inventory')
            need(r['pid'] not in pids and r['pid']>0 and r['contexts_prepared_once'] and r['context_preparations']==1,'fresh process and persistent context');pids.add(r['pid'])
            cmd=r['command'];need(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==exe,'executed arm')
            desc=resolve(cmd[1]);need(desc==peer/'build/nr_body'/descname/'chain.json' and observe(desc)==recipe['descriptor_sha256']==r['body_descriptor_sha256'],'descriptor identity')
            descriptor=read(desc)
            need(observe(desc)=='ae66950889bc96a50171097c492a26505f0a84d715cbbe47e957a99b9d691c8e' and descriptor['decoder']['checks']=='final' and not descriptor['decoder'].get('write_pool_post'),'reviewed production descriptor')
            need(cmd[cmd.index('--reset-sequence')+1]=='1,0,0' and cmd[cmd.index('--encoder-sampler-variant')+1]=='0','same reset/motion sampler')
            need(observe(resolve(cmd[cmd.index('--motion')+1]))=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','zero motion bytes')
            need('--body-checked' not in cmd and '--encoder-telemetry' in cmd and '--encoder-diagnostics' not in cmd,'actual production instrumentation')
            previous_elapsed=0
            for i,f in enumerate(r['frames']):
                e,b=f['encoder'],f['body'];want=reference['outputs'][i]
                need(f['frame']==i and f['poison']==int(patterns[pair-1][i],16) and f['body_mode']=='production' and f['status']=='comparison_passed' and f['result']=='pass','actual frame configuration/status')
                need(e['code']==b['code']==0 and e['gpu_complete'] and e['input_immutable'] and e['reset']==(i==0) and e['frame_index']==i,'encoder/body execution')
                need(e['guarded_outputs_checked']==(1039 if i==0 else 1040) and e['coverage_values_checked']==(5900107776 if i==0 else 5935497216),'unchanged encoder coverage')
                need(b['checks_skipped']==OMISSIONS[arm] and not b['check_failures'] and b['constants_unchanged_after_frame'] and b['input_texture_immutable'] and f['producer_regions_unchanged_by_body'],'declared production checks and omissions')
                need(b['final_pixels']==2073600 and b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0,'full output coverage')
                need(f['temporal_state_advanced'] and f['frames_since_reset_after']==i+1 and f['history_copy_value_differences']==0 and f['test_fault']=='' and not f['plumbing_only_no_parity_meaning'] and not f['no_reference_no_parity_meaning'],'history state progression')
                need(f['ex2_half_faults']=={'argument':0,'domain':0},'EX2 guards')
                need(f['input_sha256']==e['input_sha256']==observe(resolve(f['input_file']))=='81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','same original input')
                need(f['surface_sha256_caller_readback']==b['final_surface_sha256']==b['history_sha256_after']==want,'final/history RTX identity')
                if i:need(b['history_sha256_before']==reference['outputs'][i-1],'own previous history')
                for key,target in regions.items():need(f['regions'][key]['sha256']==reference['boundary_ports'][2*i+1][target],'RTX encoder port')
                need(f['reference']['cpu_comparison_only'] and f['reference']['mismatching_values']==0 and observe(resolve(f['reference']['file']))==want,'reference only used for comparison')
                for path in (directory/f'final-{i}.rgba16le',directory/f'diag/frame-{i}/tex_out_155.bin',directory/f'diag/frame-{i}/tex_out_156.bin'):
                    need(path.stat().st_size==16588800 and observe(path)==want,'actual output/post/history bytes');raw_bytes+=path.stat().st_size
                need(not (directory/f'diag/frame-{i}/pool_post.bin').exists(),'no hidden pool snapshot')
                middle=directory/f'diag/frame-{i}/out_view_132.bin'
                need(middle.stat().st_size==1105920 and observe(middle)==b['middle_view_sha256'],'actual middle view bytes')
                wanted_files=dict(expected_files)
                if i==0:wanted_files.pop(str(peer/'build/nr_ex2_half/table-v1/ex2-scaled-half.u32x2'))
                wanted_libs=dict(expected_libs)
                if arm=='v11':wanted_libs.pop(str(peer/'build/nr_body/verify2-resident/validate.metallib'))
                if i==0:wanted_libs.pop(str(peer/'build/nr_ptx/recovery-v23t/post.metallib'))
                actual_files={k[5:].rsplit(':',1)[0]:v for k,v in b['loaded'].items() if k.startswith('file:')}
                actual_libs={k[4:]:v for k,v in b['loaded'].items() if k.startswith('lib:')}
                need(actual_files==wanted_files and actual_libs==wanted_libs and b['constants_rehashed_after_frame']==len(wanted_files),'complete pinned loaded inventories')
                for p,h in {**actual_files,**actual_libs}.items():need(observe(p)==h,'actual loaded bytes')
                need(sum(k.startswith('ex2_half_table:') for k in b['loaded'])==int(i>0),'EX2 provenance inventory')
                cp=directory/f'diag/frame-{i}/chain-run.json';c=read(cp);observe(cp);need(c['pid']==r['pid'] and c['poison']==f['poison'],'same-process checkpoints')
                need(f['encoder_telemetry_enabled'] and f['encoder_telemetry_error']=='','telemetry available');telemetry_commands+=helper.validate_telemetry(f['encoder_telemetry'],f,r['pid'],i)
                wall=f['frame_wall_ms'];elapsed=f['process_elapsed_ms']
                need('wall_ms_total' not in f and math.isfinite(wall) and wall>0 and math.isfinite(elapsed) and elapsed>=wall and elapsed>=previous_elapsed,'frame-local and cumulative clock separation');previous_elapsed=elapsed
                need(wall+1>=e['frame_ms']+b['wall_ms'] and 'excludes telemetry copy' in f['frame_wall_boundary'],'timing envelope')
                records[arm,pair,i]=(f,c)
            need(r['files_unchanged_after_run'],'run source/assets unchanged')
            for path,h in r['post_run_file_identity'].items():need(h['unchanged'] and observe(resolve(path))==h['post']==h.get('pre',h.get('load')),'post-run file identities')
    for pair in range(1,5):
        for i in range(3):
            a,c0=records['v11',pair,i];b,c1=records['v17',pair,i]
            need(a['regions']==b['regions'] and a['surface_sha256_caller_readback']==b['surface_sha256_caller_readback'] and a['body']['history_sha256_after']==b['body']['history_sha256_after'] and a['body']['middle_view_sha256']==b['body']['middle_view_sha256'],'paired live values')
            need(len(c0['launches'])==100 and len(c1['launches'])==105,'complete chain records')
            need(c1['validators']=={'expected':126,'run':126,'faults':0} and b['inventory']['command_buffers']==b['inventory']['waits']==1,'resident coverage and inventory')
            validators=[v for launch in c1['launches'] for v in launch['validators']]
            need(len(validators)==126 and sorted(v['slot'] for v in validators)==list(range(126)) and all(v['done'] and v['faults']==0 and v['elements']>0 for v in validators),'complete unique validator records')
            need(len(c1['digest_table'])==(131 if i==0 else 135) and all(x['done'] for x in c1['digest_table']),'supplementary digest records')
            need(c1['decoder_pre_image_source']=='cpu-reconstruction (+gpu-digest-crosscheck)','production pre-image scope')
            def binary_layout(arm):
                d=root/f'{arm}-p{pair}'/f'process-{pair}'
                return {str(p.relative_to(d)):p.stat().st_size for p in d.rglob('*') if p.is_file() and p.suffix in ('.bin','.rgba16le','.e4m3')}
            need(binary_layout('v11')==binary_layout('v17'),'equal retained binary artifacts')
            rows.append({'pair':pair,'frame':i,'class':'reset-first' if i==0 else 'temporal-short','baseline_pid':a['encoder_telemetry']['pid'],'resident_pid':b['encoder_telemetry']['pid'],'baseline_frame_ms':a['frame_wall_ms'],'resident_frame_ms':b['frame_wall_ms'],'baseline_encoder_ms':a['encoder']['frame_ms'],'resident_encoder_ms':b['encoder']['frame_ms'],'baseline_body_ms':a['body']['wall_ms'],'resident_body_ms':b['body']['wall_ms']})
    need(len(pids)==8 and len(rows)==12 and raw_bytes==1194393600 and telemetry_commands==3352,'complete paired production study')
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
    return {'schema':'neuralstudio.nr.production-baseline-independent.v1','result':'pass_output_history_provenance_telemetry_with_scoped_timing','fresh_processes':8,'frames':24,'raw_output_post_history_bytes_checked':raw_bytes,'middle_view_bytes_checked':26542080,'encoder_port_hashes_checked':144,'telemetry_command_records_checked':telemetry_commands,'rows':rows,'timing':stats,'pins':pins,'checks_skipped':OMISSIONS,'default_promoted':False,'full_goal_complete':False,'limits':['One original-image zero-motion reset100 workload; four process pairs, no population/tail/game-FPS inference.','Frame-local envelope includes input handling, checks, readback/hashes/reference comparison and saves; excludes telemetry copy and final process receipt output.','Encoder prepare precedes do_frame: reset-first is not full process-startup latency.','Body v11 launch-GPU sums and v17 whole-buffer GPU time are not equivalent clocks.','Production validation work differs between arms, so a delta is not a pure-residency causal estimate.','Second top samples exist; agent quiet coordination does not prove machine-wide idleness.','Strong intermediate snapshots omitted in production; no automatic runtime promotion.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve audit')
    try:r=audit(a.peer.resolve(),a.root.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.production-baseline-independent.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(__file__);r['receipt_root_sha256']=canonical(r);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('pins','rows')},indent=2));return int(r['result']=='fail')
if __name__=='__main__':raise SystemExit(main())
