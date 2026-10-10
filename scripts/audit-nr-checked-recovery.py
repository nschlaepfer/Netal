#!/usr/bin/env python3
"""Audit recovered checked-mode inference; no timing or promotion claim."""
import argparse,hashlib,importlib.util,json,math,re,statistics
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text())
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def need(ok,why):
    if not ok:raise ValueError(why)
RECIPE_PINS={"v11":["6ab0df630bb4d8d8344f2219138e2e0dceaf0b2b6377386b2b5bf7d84161c90a","804a241b5d0f7fe003c452b43f633f443b080de8b0add8eb9b2e7ab7eb043744","417db2780e0f788561df987744163caba324a125f63793f4161ff388575a652f","fc9d36192b0dcb9d1f2cc070a5409702f2ff5d0623bffbba516693281d1326c1"],"v17":["2104261e380839c4c4608328c0b0468f13e3e4cd7309a3095f7e99f58c213c12","c78d028a9c1018aa237e150f49c9eb207926db8528bd9318f27a7f7c2a0a702d","ca48f7a3fe870d988f5095f493aece94c5321e11ae968565116c06ee1c61615f","7e056fecd6430bc221ca8392349a861a2fd2080edd7a101a72597aa03c413fd1"]}
OMISSIONS={'v11':[],'v17':[]}

def compare_checkpoints(old,new,template):
    def sequence(c):return [(x['launch'],x['kind']) for x in c['launches']]
    need(sequence(old)==sequence(template),'exact baseline launch inventory')
    middle={}
    for x in old['launches']:
        for role,h in x.get('outputs',{}).items():
            key=(x['launch'],role)
            need(key not in middle,'duplicate baseline checkpoint')
            middle[key]=h
    expected={(x['launch'],k) for x in template['launches'] for k in x.get('outputs',{})}
    need(set(middle)==expected and len(middle)==125,'exact middle checkpoint inventory')
    pools=[x for x in old['launches'] if x['kind']=='ptx']
    need([x['launch'] for x in pools]==list(range(133,157)),'exact decoder pool inventory')
    for x in pools:middle[x['launch'],'pool']=x['pool_post_sha256']
    candidate={}
    for x in new['launches']:
        for k,h in x['digests'].items():
            if k.endswith('_sha256'):
                key=(x['launch'],k[:-7])
                need(key not in candidate,'duplicate resident checkpoint')
                candidate[key]=h
    need(len(middle)==149 and middle.keys()<=candidate.keys(),'complete paired checkpoint inventory')
    valid=lambda h:isinstance(h,str) and re.fullmatch('[0-9a-f]{64}',h) is not None
    need(all(valid(h) and valid(candidate[k]) and h==candidate[k] for k,h in middle.items()),'all paired checkpoint SHA256 values match')
    need(new['decoder_pre_image_source']=='gpu-snapshot' and valid(new['decoder_pre_image_sha256']) and new['decoder_pre_image_sha256']==new['decoder_pre_image_expected_sha256']==old['decoder']['pool_pre_sha256'],'actual pre-image equality')
    need(valid(new['pool_final_sha256']) and new['pool_final_sha256']==old['decoder']['pool_final_sha256'],'final pool equality')
    need(valid(new['tex155_output_sha256']) and new['tex155_output_sha256']==next(x for x in pools if x['launch']==155)['texture']['output_sha256'],'post texture equality')
    return 152

def checkpoint_signature(chain):
    return {'outputs':[(x['launch'],x.get('outputs',{}),x.get('pool_post_sha256')) for x in chain['launches']],
            'preimage':chain['decoder']['pool_pre_sha256'],'final_pool':chain['decoder']['pool_final_sha256'],
            'texture':next(x for x in chain['launches'] if x['launch']==155)['texture']['output_sha256']}

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
    for name,h in er['assets'].items():need(observe(repo/'build/nr_encoder_telemetry/zero-v3/assets'/name)==h,'actual encoder assets')
    order=[('v11',1),('v17',1),('v17',2),('v11',2),('v11',3),('v17',3),('v17',4),('v11',4)]
    summary=read(root/'checked-summary.json');observe(root/'checked-summary.json')
    need(summary['all_processes_passed'] and [(r['arm'],r['pair']) for r in summary['runs']]==order and all(r['exit_code']==0 for r in summary['runs']),'complete ordered checked run')
    template_path=peer/'build/nr_full/baseline-prod-v1/v11-p1/process-1/diag/frame-0/chain-run.json'
    need(observe(template_path)=='0e31c288fc9ca3a0e460881657fcae7c9cd4640cddda1468cde23f35c32039ec','accepted baseline structural inventory')
    template=read(template_path)
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
            need(recipe['harness_executable_sha256']==exe and recipe['harness_extra_args']==['--encoder-telemetry','--body-checked'] and recipe['replicas']==[{'tag':f'process-{pair}','poisons':patterns[pair-1]}],'recipe arm/flags/poisons')
            r=read(directory/'receipt.json');need(observe(directory/'receipt.json')==s['runs'][0]['receipt_sha256'],'native receipt hash')
            need(r['schema']=='neuralstudio.nr.full-frame-inmemory.v13' and r['result']=='pass' and len(r['frames'])==r['frames_in_process']==3,'native frame inventory')
            need(r['pid'] not in pids and r['pid']>0 and r['contexts_prepared_once'] and r['context_preparations']==1,'fresh process and persistent context');pids.add(r['pid'])
            cmd=r['command'];need(resolve(cmd[0])==build/'nr_full_frame' and r['executable_sha256']==exe,'executed arm')
            desc=resolve(cmd[1]);need(desc==repo/'build/nr_checked_recovery/config-v1/chain.json' and observe(desc)==recipe['descriptor_sha256']==r['body_descriptor_sha256'],'descriptor identity')
            descriptor=read(desc)
            need(observe(desc)=='3d3189665884f519bf4b1bfd5fc5389bee755d376116b14790bf7c633e57254e' and descriptor['decoder']['write_pool_post']=='true','reviewed checked retention descriptor')
            parent=peer/'build/nr_body/desc-recovery-v1/chain.json'
            need(observe(parent)=='ae66950889bc96a50171097c492a26505f0a84d715cbbe47e957a99b9d691c8e','parent descriptor')
            expected=read(parent);expected['decoder']['write_pool_post']='true'
            need(descriptor==expected,'only pool retention descriptor difference')
            need(cmd[cmd.index('--reset-sequence')+1]=='1,0,0' and cmd[cmd.index('--encoder-sampler-variant')+1]=='0','same reset/motion sampler')
            need(observe(resolve(cmd[cmd.index('--motion')+1]))=='788ae0147bdf979a6575938ca2d7d4403788588f7be2010f03776c968fd1ab49','zero motion bytes')
            need('--body-checked' in cmd and '--encoder-telemetry' in cmd and '--encoder-diagnostics' not in cmd,'actual production instrumentation')
            previous_elapsed=0
            for i,f in enumerate(r['frames']):
                e,b=f['encoder'],f['body'];want=reference['outputs'][i]
                need(f['frame']==i and f['poison']==int(patterns[pair-1][i],16) and f['body_mode']=='checked' and f['status']=='comparison_passed' and f['result']=='pass','actual frame configuration/status')
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
                pool=directory/f'diag/frame-{i}/pool_post.bin'
                need(pool.stat().st_size==descriptor['pool_bytes'] and observe(pool)==b['pool_final_sha256'],'actual retained pool')
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
            need(c1['validators']=={'expected':150,'run':150,'faults':0} and b['inventory']['command_buffers']==b['inventory']['waits']==1,'resident coverage and inventory')
            validators=[v for launch in c1['launches'] for v in launch['validators']]
            need(len(validators)==150 and sorted(v['slot'] for v in validators)==list(range(150)) and all(v['done'] and v['faults']==0 and v['elements']>0 for v in validators),'complete unique validator records')
            need(len(c1['digest_table'])==(155 if i==0 else 159) and all(x['done'] for x in c1['digest_table']),'supplementary digest records')
            compare_checkpoints(c0,c1,template)
            need(b['resident_evidence']['digest_cpu_crosschecks']>=152 and b['resident_evidence']['digest_cpu_mismatches']==0,'snapshot GPU/CPU digest crosschecks')
            def binary_layout(arm):
                d=root/f'{arm}-p{pair}'/f'process-{pair}'
                return {str(p.relative_to(d)):p.stat().st_size for p in d.rglob('*') if p.is_file() and p.suffix in ('.bin','.rgba16le','.e4m3')}
            need(binary_layout('v11')==binary_layout('v17'),'equal retained binary artifacts')
            rows.append({'pair':pair,'frame':i,'class':'reset-first' if i==0 else 'temporal-short','baseline_pid':a['encoder_telemetry']['pid'],'resident_pid':b['encoder_telemetry']['pid'],'baseline_frame_ms':a['frame_wall_ms'],'resident_frame_ms':b['frame_wall_ms'],'baseline_encoder_ms':a['encoder']['frame_ms'],'resident_encoder_ms':b['encoder']['frame_ms'],'baseline_body_ms':a['body']['wall_ms'],'resident_body_ms':b['body']['wall_ms']})
    need(len(pids)==8 and len(rows)==12 and raw_bytes==1194393600 and telemetry_commands==3352,'complete paired production study')
    for i in range(3):
        for poison in (165,90):
            signatures=[checkpoint_signature(c) for (arm,pair,frame),(f,c) in records.items() if arm=='v11' and frame==i and f['poison']==poison]
            need(len(signatures)==2 and signatures[0]==signatures[1],'fresh-process same-poison strong checkpoint repeat')
    for p,h in pins.items():need(sha(p)==h,'audit-time drift')
    return {'schema':'neuralstudio.nr.checked-recovery-independent.v1','result':'pass_live_original_checked_pair','fresh_processes':8,'frames':24,'paired_checkpoint_identities':1824,'raw_output_post_history_bytes_checked':raw_bytes,'retained_pool_bytes_checked':24*descriptor['pool_bytes'],'encoder_port_hashes_checked':144,'telemetry_command_records_checked':telemetry_commands,'rows':rows,'pins':pins,'checks_skipped':OMISSIONS,'both_poisons_repeated_every_frame':True,'default_promoted':False,'full_goal_complete':False,'speedup_accepted':False,'limits':['Original-image zero-motion reset100 only; this does not re-establish the varied-motion strong checkpoint suite or long-run/game acceptance.','152 SHA256 identities per pair compare fresh live v11 computation with fresh live v17 snapshots, not a newly captured RTX intermediate chain.','RTX final/history and six encoder ports remain pinned and independently compared.','Strong per-launch buffers are snapshotted and hashed inside the process; retained raw pool is the final pool, not all intermediate snapshots.','No benchmark or timing acceptance.']}

def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--root',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();need(not a.out.exists(),'preserve audit')
    try:r=audit(a.peer.resolve(),a.root.resolve())
    except Exception as e:r={'schema':'neuralstudio.nr.checked-recovery-independent.v1','result':'fail','error':str(e)}
    r['verifier_sha256']=sha(__file__);r['receipt_root_sha256']=canonical(r);a.out.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('pins','rows')},indent=2));return int(r['result']=='fail')
if __name__=='__main__':raise SystemExit(main())
