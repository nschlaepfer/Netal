#!/usr/bin/env python3
"""Audit reviewed restored live runs; production omissions remain explicit."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OLD = '/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt/'
PINS = {
    11: ('565ef5eefd7f4e59ebf27dbd8ff7875e351408888c328bfbf02805954878951d', '951f18b589d8e0a46f07da1e0e87bf60a978338648d0eae1428cce45a35458fa'),
    17: ('8ba64fe040b5dacf76a31f318c6c62b4d503b8c5bb1fee45587573b4b0b5d928', '08aa848099d3ee8690b356507a1fb31220d239f350b79525fb5039b9032a4754')}
DESC = 'ae66950889bc96a50171097c492a26505f0a84d715cbbe47e957a99b9d691c8e'
VARIED_RECIPES = {
    'checker3':'7dc3331400056664e182b7081d5a3db3149666a3bbcc5b92698832248a79e879',
    'images3':'ab2c5d074ce400e1ead79524d1152a634fd924c68e7c3dce98a82fa92e58e603',
    'midreset8':'a86ca2c901c97a2af78c8577660ffbb77b52b72a94a2bf335db8bbb9b194f461',
    'uniform3':'80ac5388fa6493155c2e573edda86844167324e73817df60e157defd426e91e1',
    'spatial3':'32ccfd43c1691b01b9a28ac23fb4676e63b0c256958a0bc09bc1840a1f3b1a7e',
    'combined3':'8504cf510b93ca65ec878f33c99ff3d5c2b010369062fa6c5539004947da3934',
    'dense':'048a4d07e278160b145c3c73037c2d3ae3319a9a823ed2472dc52ce348df16c0'}
RUNNER = '340f4c96a500e856a8c722b3dd7ce1c17814efb5b3b7bf3a5da07d5b372df156'
SIXTEEN_RECIPE='dca603d0699e1e1f2f7ad41f45dcdc63b48ccb67240542cf86076f059f7be8ef'
SIXTEEN_REMAINING='5c4072b53254727b347d5cf9eeb00a433247fe1d3b1f367e79fb3cbf34120d91'
SIXTEEN_PREFIX={'repeat-a':'8cb6f50040bf4a47fb659d798c877c44d7acd73942d740465360a67687039c4b','repeat-b':'89299ea93d0564f4c6a4061a063e77d3419b38897abd25566c72b7af26f00bb3'}
REGIONS = {'entry': ('block22-middle',1105920), 'skip_c256': ('block22-native',2211840),
           'skip_c128': ('block14-native',4423680), 'skip_c64': ('block8-native',8847360),
           'skip_c32': ('block4-native',17694720), 'block0_skip': ('block0-skip',70778880)}
OMISSIONS = {11: ['decoder per-launch pool hash', 'decoder per-launch pool coverage scan'],
    17: ['decoder per-launch pool digest', 'decoder per-launch pool coverage validator',
         'intermediate SHA-256 snapshots (checked mode only; GPU digests + boundary SHA-256 in production)',
         'decoder pre-image actual GPU snapshot SHA-256 (production reports the CPU reconstruction with the GPU digest cross-check)']}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical(data):
    return hashlib.sha256(json.dumps(data, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sealed(path, root):
    data = read(path)
    need(data.pop('receipt_root_sha256') == root and canonical(data) == root, 'immutable parent seal')
    return data


def reference_contract(case):
    spec=importlib.util.spec_from_file_location('historical_reference_contract',ROOT/'scripts/audit-nr-resident-live.py')
    helper=importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    if case=='sixteen':
        root='4feea5197385aecc8a9784f3ac955cd9362ec9f1f3b7ebfe2c6d7aaf55998576'
        r=sealed(ROOT/'build/nr_sixteen/matrix-audit-v1.json',root);r['receipt_root_sha256']=root
        retrieval=sealed(ROOT/'build/nr_sixteen/raw-reference-recovery-v1/receipt.json','21f516d4b05b88b5edb869b30d245e85f6732765ab1216f7e846b44c0300a78c')
        need(retrieval['result']=='pass_verified_historical_raw_retrieval' and retrieval['remote_payloads_verified']==192 and len(retrieval['local_files'])==16,'verified complete raw RTX reference retrieval')
        for i,f in enumerate(retrieval['local_files']):
            need(f['frame']==i and f['bytes']==16588800 and Path(f['path']).stat().st_size==16588800 and sha(f['path'])==f['sha256']==r['outputs'][i],'actual sixteen-frame reference bytes')
        return r,16,[1]+[0]*15,[helper.ORIGINAL]*16,helper.ZERO_MOTION,False
    if case=='encfill-original3':
        root='e246c8983088c95e318a18c3cf74c8034abf2dbbf2a25c4d30f1da95081fe657'
        r=sealed(ROOT/'build/nr_three_frame_history/matrix-audit-v1.json',root);r['receipt_root_sha256']=root
        return r,3,[1,0,0],[helper.ORIGINAL]*3,helper.ZERO_MOTION,False
    return helper.configuration(ROOT,case)


def audit(peer, directory, version, case='black3', *, resume_from=None, prefix_only=False):
    need(not (resume_from is not None and prefix_only),'choose union or prefix audit')
    if resume_from is not None or prefix_only:need(case=='sixteen' and version==17,'bounded sixteen recovery partition only')
    need(case == 'black3' or ((case in VARIED_RECIPES or case in ('encfill-original3','sixteen')) and version == 17), 'reviewed case/body pairing')
    matrix,n,resets,images,motion_hash,fractional=reference_contract(case)
    temporal_tag='v24t' if fractional else 'v23t'
    descriptor_hash=DESC if not fractional else '60d52fa4c3007e41bd71e6dba208083411b33082be1472ca947ccc361cb665f0'
    if case == 'black3':
        recipe_hash,executable_hash=PINS[version]
        encoder_hash='942541e17fa3c733605a65813a35a84853ab781ac4501b8695c4f71c35f953de'
    elif case=='sixteen':
        recipe_hash=SIXTEEN_RECIPE
        executable_hash='4aa3b4917762fba2efb460a19938916d9bb0bd4199038dc313a0f56a71a1d9c1'
        encoder_hash='830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3'
    elif case=='encfill-original3':
        recipe_hash='ec6b1321cdcc5c895743e3d49d994211a2e1cf28994047730df1f18e82290d54'
        executable_hash='c59c3e59af28fd5f9f8189f04b3d4d981729215c53ca631786faaa4f8cbf0927'
        encoder_hash='3b0e284608941c0e4e6bf246ffa91c88571a141428880712d603afc2a73fad02'
    else:
        recipe_hash=VARIED_RECIPES[case]
        executable_hash='3fc61d2d749e557df88541cbcd493bf09c04059246d8b195c987a5a6f5f40ef8' if fractional else '4aa3b4917762fba2efb460a19938916d9bb0bd4199038dc313a0f56a71a1d9c1'
        encoder_hash='33f92090b058c509d822ee749588c1c694ceac0f2a44b2999fa570fdc9a704e8' if fractional else '830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3'
    parent=sealed(ROOT/'build/nr_resident_review'/('live-v17-uniform3-v1.json' if fractional else 'live-v17-black3-v2.json'),
        '95df8deac6f4018ad94b7150014ebaa76750ea1b61b2f48e9f242eb40c317114' if fractional else '5bd81e7b5094114f3a1efd55a1a952f9abfaea1d3e2aaacda46e1a3326d92a50')
    constants = sealed(ROOT/'build/nr_resident_review/recovered-constants-v2.json', '550fbdbecf0fda9d1a70b28daf5ca3a64aa6fb18fbd11c81949444474253e170')
    required_constants = {a['current_path'] for a in constants['artifacts']}
    required_constants |= {str(peer/'build/nr_ptx/tables'/n) for n in
        ('rtx-rsqrt-half.bin','rtx-rcp-half.bin','sm120-rcp-correction-rom.bin','unit-corrections.bin','fraction-corrections.bin')}
    required_libs = {str(peer/p) for p in ('build/nr_split/verify2-v13/split.metallib',
        'build/nr_vit/verify2-v8/vit.metallib','build/nr_tp/verify-v5/tp.metallib',
        'build/nr_vit/verify-tp-v8/tp.metallib','build/nr_ptx/recovery-v17/dec.metallib',
        'build/nr_ptx/recovery-v17/post.metallib',f'build/nr_ptx/recovery-{temporal_tag}/copy.metallib')}
    if version == 17: required_libs.add(str(peer/'build/nr_body/verify2-resident/validate.metallib'))
    aliases = {'build/nr_split/attempt-v13/':'build/nr_split/verify2-v13/',
        'build/nr_vit/attempt-v8/':'build/nr_vit/verify2-v8/', 'build/nr_tp/attempt-v5/':'build/nr_tp/verify-v5/',
        'build/nr_vit/tp-attempt-v8/':'build/nr_vit/verify-tp-v8/',
        'build/nr_body/resident-lib-v2/':'build/nr_body/verify2-resident/'}
    for variant in ('v17','v23t','v24t'):
        aliases['build/nr_ptx/attempt-'+variant+'/'] = 'build/nr_ptx/recovery-'+variant+'/'
    known = {}
    for old, expected in parent['artifacts'].items():
        if not old.startswith(OLD):
            known[old] = expected
            continue
        relative = old[len(OLD):]
        for previous, new in aliases.items():
            if relative.startswith(previous): relative = new + relative[len(previous):]; break
        if '/decoder-chain/stage/' in relative:
            relative = 'build/nr_body/desc-v31t/production-a5-a/decoder-chain/stage/' + Path(relative).name
        known[str(peer/relative)] = expected
    observed = {}
    def observe(path):
        path = Path(path).resolve(); value = sha(path)
        need(str(path) not in observed or observed[str(path)] == value, 'artifact drift')
        observed[str(path)] = value
        return value
    telemetry_helper=None
    if case=='sixteen':
        enc=ROOT/'build/nr_encoder_temporal/half-fma-v1'
        eb=sealed(enc/'build.json','0a57b3bac8616c7103ae30a2df5a91296f3bda2ab9f84960725bb943df053b2e')
        for path,h in eb['assets'].items():need(observe(enc/'assets'/path)==h,'repaired encoder assets')
        for path,h in eb['source_files'].items():need(observe(enc/'source'/path)==h,'repaired encoder source')
    if case=='encfill-original3':
        build=ROOT/'build/nr_encoder_gpu_fill/block2-v1'
        candidate=sealed(build/'build.json','03b3a9c3bb07f855638a328cd103ca6ae313b96a310fcb8533e85319e09b631b')
        parent_build=sealed(ROOT/'build/nr_encoder_telemetry/zero-v3/build.json','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
        need(candidate['assets']==parent_build['assets'],'unchanged model shader/weight assets')
        for path,h in candidate['assets'].items():need(observe(build/'assets'/path)==h,'actual unchanged candidate asset')
        for path,h in candidate['source_files'].items():need(observe(build/'source'/path)==h,'candidate source identity')
        spec=importlib.util.spec_from_file_location('encfill_telemetry',ROOT/'scripts/audit-nr-connected-telemetry.py')
        telemetry_helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(telemetry_helper)
    partitioned=resume_from is not None or prefix_only
    planned=[]
    if partitioned:
        prefix=directory if prefix_only else resume_from
        need(prefix.resolve()==ROOT/'build/nr_sixteen/native-recovery-v1','exact interrupted source root')
        recipe=read(prefix/'recipe-as-run.json')
        need(observe(prefix/'recipe-as-run.json')==recipe_hash==SIXTEEN_RECIPE,'original four-process contract')
        need([r['tag'] for r in recipe['replicas']]==['repeat-a','repeat-b','all-a5','all-5a'],'fixed original replica inventory')
        for replica in recipe['replicas'][:2]:
            d=prefix/replica['tag']
            need(observe(d/'receipt.json')==SIXTEEN_PREFIX[replica['tag']],'immutable completed prefix receipt')
            planned.append((d,replica,None))  # No original aggregate/OS exit record is invented.
        if not prefix_only:
            need(directory.resolve()!=prefix.resolve(),'new root for unfinished replicas')
            tail=read(directory/'recipe-as-run.json');summary=read(directory/'run-summary.json');observe(directory/'run-summary.json')
            need(observe(directory/'recipe-as-run.json')==summary['recipe_sha256']==SIXTEEN_REMAINING,'reviewed remaining recipe')
            ignored={'name','replicas','evidence_first_run'}
            need({k:v for k,v in tail.items() if k not in ignored}=={k:v for k,v in recipe.items() if k not in ignored} and tail['replicas']==recipe['replicas'][2:],'identical controls and only unfinished replicas')
            need(summary['runner_sha256']==RUNNER and summary['all_replicas_passed'] and summary['replicas_requested']==summary['replicas_passed']==len(summary['runs'])==2,'complete two-process continuation')
            for run,replica in zip(summary['runs'],tail['replicas']):planned.append((directory/replica['tag'],replica,run))
    else:
        summary = read(directory/'run-summary.json'); observe(directory/'run-summary.json')
        recipe = read(directory/'recipe-as-run.json')
        need(observe(directory/'recipe-as-run.json') == summary['recipe_sha256'] == recipe_hash, 'reviewed recipe identity')
        need(summary['runner_sha256'] == RUNNER and summary['all_replicas_passed'] and summary['replicas_requested'] == summary['replicas_passed'] == len(summary['runs']) == 4, 'complete four-process set')
        planned=[(directory/replica['tag'],replica,run) for run,replica in zip(summary['runs'],recipe['replicas'])]
    need(observe(peer/recipe['descriptor']) == descriptor_hash and observe(peer/recipe['harness_build']/'nr_full_frame') == executable_hash, 'reviewed descriptor and executable')
    need(observe(Path(recipe['encoder_object'])) == recipe['encoder_object_sha256'] == encoder_hash, 'frozen case-appropriate encoder object')
    need(observe(peer/recipe['motion']['file']) == motion_hash and recipe['encoder_sampler_variant'] == (3 if fractional else 0), 'case motion and sampler contract')
    need(recipe['reset_sequence'] == ','.join(map(str,resets)) and len(recipe['frames']) == n, 'case reset and frame contract')
    pids = set(); rows = []; raw_bytes = 0
    for d,replica,run in planned:
        if run is not None:need(run['tag'] == replica['tag'] and run['exit_code'] == 0 and run['passed'] and run['frame_status'] == ['comparison_passed']*n, 'all case frames compared in each process')
        native = read(d/'receipt.json')
        need(observe(d/'receipt.json') == (run['receipt_sha256'] if run is not None else SIXTEEN_PREFIX[replica['tag']]), 'native receipt identity')
        need(native['schema'] == 'neuralstudio.nr.full-frame-inmemory.v13' and native['result'] == 'pass' and native['frames_in_process'] == len(native['frames']) == n, 'native frame inventory')
        need(native['pid'] > 0 and native['pid'] not in pids and native['contexts_prepared_once'] and native['context_preparations'] == 1, 'fresh process, persistent context')
        pids.add(native['pid'])
        need(native['executable_sha256'] == executable_hash and native['body_descriptor_sha256'] == descriptor_hash, 'actual runtime identity')
        need(native['files_unchanged_after_run'] and native['execution_completed'], 'completed immutable execution')
        temporal_seen=False; since_reset=0
        for i, frame in enumerate(native['frames']):
            encoder, body = frame['encoder'], frame['body']; expected = matrix['outputs'][i]
            reset=bool(resets[i]); temporal_seen=temporal_seen or not reset
            since_reset=1 if reset else since_reset+1
            need(frame['frame'] == i and frame['poison'] == int(replica['poisons'][i],16) and frame['body_mode'] == 'production' and frame['status'] == 'comparison_passed' and frame['result'] == 'pass', 'frame execution contract')
            need(not frame['plumbing_only_no_parity_meaning'] and not frame['no_reference_no_parity_meaning'] and not frame['test_fault'], 'no bypass or fault hook')
            need(encoder['code'] == body['code'] == 0 and encoder['gpu_complete'] and encoder['input_immutable'], 'successful guarded execution')
            # The harness's global frame label continues through a reset, but
            # NrEncoderFrame.frame_index is explicitly frames_since_reset.
            need(encoder['reset'] == reset and encoder['frame_index'] == since_reset-1 and encoder['guarded_outputs_checked'] == (1039 if reset else 1040) and encoder['coverage_values_checked'] == (5900107776 if reset else 5935497216), 'encoder reset and full guard inventory')
            if not reset: need(encoder['sampler_variant'] == (3 if fractional else 0), 'actual encoder sampler variant')
            if telemetry_helper is not None:
                need(frame['encoder_telemetry_enabled'] and not frame['encoder_telemetry_error'],'candidate telemetry enabled')
                telemetry_helper.validate_telemetry(frame['encoder_telemetry'],frame,native['pid'],i)
                labels=[p['label'] for p in frame['encoder_telemetry']['phases']]
                need(labels.count('block2/allocation_and_guards')==1 and labels.count('block2/gpu_fill_encoding')==1 and 'block2/allocation_and_poison' not in labels,'actual Block2 candidate branch executed')
                need(all(x.startswith('block2/') for x in labels if x.endswith('/gpu_fill_encoding') or x.endswith('/allocation_and_guards')),'candidate remains scoped to Block2')
            need(frame['input_sha256'] == encoder['input_sha256'] == observe(frame['input_file']) == images[i], 'original frame-specific input')
            need(body['checks_skipped'] == OMISSIONS[version] and body['checks_enforced'] > 0 and not body['check_failures'] and body['constants_unchanged_after_frame'], 'declared checks and omissions')
            need(body['final_pixels'] == 2073600 and body['final_pixels_not_written'] == body['final_pixels_written_more_than_once'] == 0 and body['input_texture_immutable'] and frame['producer_regions_unchanged_by_body'], 'full final coverage and input immutability')
            need(frame['ex2_half_faults'] == {'argument':0,'domain':0} and frame['history_copy_value_differences'] == 0 and frame['temporal_state_advanced'] and frame['frames_since_reset_after'] == since_reset, 'temporal state and EX2 guards')
            need(frame['temporal']['enabled'] and frame['temporal']['body_reset'] == frame['temporal']['encoder_reset'] == reset, 'actual reset agreement')
            if not reset: need(body['history_sha256_before'] == matrix['outputs'][i-1] and body['motion_sha256'] == motion_hash, 'own prior exact history and actual motion')
            need(frame['reference']['cpu_comparison_only'] and frame['reference']['mismatching_values'] == 0 and observe(frame['reference']['file']) == expected, 'comparison-only actual RTX output')
            need(frame['surface_sha256_caller_readback'] == body['final_surface_sha256'] == body['history_sha256_after'] == expected, 'caller output and history identities')
            for name, (port, size) in REGIONS.items():
                region = frame['regions'][name]
                need(region['bytes'] == size and region['offset'] == 256 and region['sha256'] == matrix['boundary_ports'][2*i+1][port], 'live encoder boundary '+name)
            for path in (d/f'final-{i}.rgba16le', d/f'diag/frame-{i}/tex_out_155.bin', d/f'diag/frame-{i}/tex_out_156.bin'):
                need(path.stat().st_size == 16588800 and observe(path) == expected, 'actual saved output/history bytes')
                raw_bytes += 16588800
            expected_files, expected_libs = set(required_constants), set(required_libs)
            if temporal_seen:
                expected_files.add(str(peer/'build/nr_ex2_half/table-v1/ex2-scaled-half.u32x2'))
                expected_libs.add(str(peer/f'build/nr_ptx/recovery-{temporal_tag}/post.metallib'))
            actual_files = {k[5:].rsplit(':',1)[0] for k in body['loaded'] if k.startswith('file:')}
            actual_libs = {k[4:] for k in body['loaded'] if k.startswith('lib:')}
            need(actual_files == expected_files and actual_libs == expected_libs, 'complete exact loaded constant/library inventory')
            need(body['constants_rehashed_after_frame'] == len(expected_files), 'all loaded constants rehashed')
            need(sum(k.startswith('ex2_half_table:') for k in body['loaded']) == int(not reset), 'temporal EX2 provenance inventory')
            for key, value in body['loaded'].items():
                if key.startswith('file:'):
                    path = key[5:].rsplit(':',1)[0]
                elif key.startswith('lib:'): path = key[4:]
                elif key.startswith('ex2_half_table:'): path = key[len('ex2_half_table:'):]
                else: raise ValueError('unreviewed provenance key '+key)
                need(observe(path) == known.get(path) == value, 'actual loaded bytes versus historical artifact pin: '+path)
            if version == 17:
                evidence = frame['resident_evidence']
                need(frame['inventory']['command_buffers'] == frame['inventory']['waits'] == 1 and evidence['validators_expected'] == evidence['validators_run'] == 126 and evidence['coverage_faults'] == 0, 'resident command buffer and coverage validators')
                need(evidence['digests_expected'] == evidence['digests_complete'] == (131 if reset else 135) and evidence['digest_cpu_mismatches'] == 0, 'supplementary resident digests')
            rows.append({'pid':native['pid'],'frame':i,'poison':frame['poison'],'output_history_sha256':expected})
        for path, identity in native['post_run_file_identity'].items():
            need(identity['unchanged'] and identity['post'] == identity.get('pre',identity.get('load')) == observe(path), 'post-run file identity')
    for path, expected in observed.items(): need(sha(path) == expected, 'audit-time file drift')
    need(len(planned)==(2 if prefix_only else 4) and len(pids)==len(planned) and len(rows)==len(planned)*n,'complete declared native partition')
    return {'schema':'neuralstudio.nr.recovery-live-audit.v1', 'result':'pass_completed_native_prefix_only' if prefix_only else 'pass_bounded_live_output_ports_history',
        'configuration':case, 'reference_root':matrix['receipt_root_sha256'],
        'body_version':version, 'fresh_processes':len(pids), 'frames':len(rows), 'actual_output_history_bytes_checked':raw_bytes,
        'encoder_boundary_hashes_checked':len(rows)*6, 'rows':rows, 'artifacts':observed,
        'strict_intermediate_checkpoint_acceptance':False, 'full_goal_complete':False, 'performance_claim':False,
        'execution_partition':{'prefix_only':prefix_only,'union_after_interruption':resume_from is not None,'original_aggregate_exit_status':'unavailable; supervisor terminated with signal15 during third process' if partitioned else 'see runner summary','declared_native_directories':[str(d) for d,_,_ in planned],'partial_third_process_reused':False},
        'limits':[f'Only the pinned {case} sequence, {n} frames per process at1920x1080; this receipt alone is not the full regression suite.',
                  'Production body omissions remain explicit; final/history parity and boundary hashes do not replace strong intermediate checks.',
                  'No speedup, real-time, game integration or performance-baseline claim.'], 'verifier_sha256':sha(Path(__file__))}


def main():
    p = argparse.ArgumentParser(); p.add_argument('--peer-root',type=Path,required=True); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--body-version',type=int,choices=(11,17),required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--resume-from',type=Path);p.add_argument('--completed-prefix-only',action='store_true')
    p.add_argument('--configuration',choices=['black3','encfill-original3','sixteen',*VARIED_RECIPES],default='black3'); args=p.parse_args()
    need(not args.out.exists(), 'preserve previous audit')
    try: result=audit(args.peer_root.resolve(),args.run.resolve(),args.body_version,args.configuration,resume_from=args.resume_from.resolve() if args.resume_from else None,prefix_only=args.completed_prefix_only)
    except Exception as error: result={'schema':'neuralstudio.nr.recovery-live-audit.v1','result':'fail','error':str(error),'body_version':args.body_version,'verifier_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result)
    with args.out.open('x') as stream: json.dump(result,stream,indent=2); stream.write('\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ('rows','artifacts')},indent=2))
    return 1 if result['result']=='fail' else 0


if __name__ == '__main__': raise SystemExit(main())
