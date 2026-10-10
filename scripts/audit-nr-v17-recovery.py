#!/usr/bin/env python3
"""Independent v17 failure/no-advance/recovery audit; no new GPU execution."""
import argparse,hashlib,json
from pathlib import Path
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canon(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args();assert not a.out.exists();peer=a.peer.resolve();suite=peer/'build/nr_body/v17-evidence-v1';build=peer/'build/nr_full/v31t-fault';pins={}
    def observe(p):
        h=sha(p);assert str(p) not in pins or pins[str(p)]==h;pins[str(p)]=h;return h
    assert observe(suite/'summary.json')=='83b6e2f38bdf38fb3ce8c8449c9a06790b6b43b0c5f7b46cd2c98c484c9117f0'
    summary=json.loads((suite/'summary.json').read_text());assert len(summary['cases'])==9
    for line in (build/'build-sha256.txt').read_text().splitlines():
        h,path=line.split(None,1);assert observe(peer/path.strip())==h
    assert (build/'build-defines.txt').read_text().strip()=='defines: -DNR_BODY_FAULT_TESTS'
    assert observe(build/'source/src/nr_body/nr_body_runtime_v17.mm')=='64bbe143ffc11373ddc0d796a86d6adf45104113be52022bdc50e9e647dccc1a'
    assert observe(build/'source/src/nr_body/nr_full_frame.mm')=='2d98d752b33fb505b53c91aaea34122539207c1ab3ffa5c07bca37e4bad6c65f'
    assert observe(build/'source/encoder/nr_encoder_runtime.o')=='830d280992f9d2a2032bd6af625a5de462bb3d770bd2c5cb65e0003e7033fff3'
    faults=['ex2-table-fault','history-copy-mismatch','ex2-table-row-corrupt','history-texel-corrupt','constant-late-mutation','validator-skipped','coverage-compensating','restage-corrupt']
    refs=['a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2','d48bf48553e61c9e76d23b008dc1192862a11894e1465a6366bf1476473c9ac0']
    rows=[];pids=set();count=0
    for fault in faults:
        d=suite/'harness'/('fault-'+fault);r=json.loads((d/'receipt.json').read_text());rh=observe(d/'receipt.json')
        s=next(c for c in summary['cases'] if c['case']=='harness-fault-'+fault);assert s['ok'] and s['receipt_sha256']==rh
        assert r['schema']=='neuralstudio.nr.full-frame-inmemory.v12' and r['executable_sha256']==observe(build/'nr_full_frame') and r['pid'] not in pids;pids.add(r['pid'])
        assert r['result']=='fail' and len(r['frames'])==7 and r['context_preparations']==3 and not r['contexts_prepared_once']
        f=r['frames'];assert f[1]['test_fault']==fault and f[1]['body']['code']==3 and f[1]['body']['check_failures']==s['frame1_check_failures']
        assert not f[1]['temporal_state_advanced'] and f[1]['frames_since_reset_after']==1
        assert f[2]['encoder']['code']==2 and f[2]['status']=='rejected' and not f[2].get('temporal_state_advanced',False)
        assert f[3]['body']['code']==2 and 'context damaged' in f[3]['body']['error'] and f[3]['body']['checks_enforced']==0 and not f[3]['temporal_state_advanced']
        steps=r['recovery']['steps'];assert r['recovery']['requested'] and r['recovery']['result']=='pass' and len(steps)==4
        assert [x['observed'] for x in steps]==['rejected','pass','pass','comparison-fail'] and all(x['ok'] and x['expected']==x['observed'] for x in steps)
        for index,want in [(0,refs[0]),(4,refs[0]),(5,refs[1]),(6,refs[0])]:
            fr=f[index];b=fr['body'];assert b['code']==0 and fr['surface_sha256_caller_readback']==b['final_surface_sha256']==b['history_sha256_after']==want
            assert b['final_pixels_not_written']==b['final_pixels_written_more_than_once']==0 and b['input_texture_immutable'] and fr['producer_regions_unchanged_by_body']
            assert not b['check_failures'] and b['constants_unchanged_after_frame'] and fr['temporal_state_advanced']
            for name in ('tex_out_155.bin','tex_out_156.bin'):
                path=d/f'diag/frame-{index}'/name;assert path.stat().st_size==16588800 and observe(path)==want;count+=16588800
        assert f[4]['result']==f[5]['result']=='pass' and f[4]['frames_since_reset_after']==1 and f[5]['frames_since_reset_after']==2
        assert f[6]['result']=='fail' and f[6]['status']=='comparison_failed' and f[6]['reference']['mismatching_values']>0 and f[6]['reference']['cpu_comparison_only'] and observe(Path(f[6]['reference']['file']))==refs[1]
        # Retain/check identities of failed-frame bytes without comparing them
        # to successful expected values or feeding them into any execution.
        fault_files={str(p):observe(p) for p in (d/'diag/frame-1').glob('*.bin')}
        rows.append({'fault':fault,'pid':r['pid'],'receipt_sha256':rh,'state_not_advanced':True,'fresh_recovery_exact':True,'fault_artifacts':fault_files})
    prod=suite/'harness/prod-harness-rejects-fault-hook';r=json.loads((prod/'receipt.json').read_text());observe(prod/'receipt.json')
    assert r['pid'] not in pids and r['result']=='fail' and r['frames'][0]['body']['code']==2 and 'test_fault set on a production build' in r['frames'][0]['body']['error']
    assert r['frames'][0]['body']['checks_enforced']==0 and r['frames'][0]['inventory']['command_buffers']==0 and not r['frames'][0]['temporal_state_advanced']
    assert not list(prod.glob('*.rgba16le'));pids.add(r['pid'])
    assert r['executable_sha256']==observe(peer/'build/nr_full/v31t/nr_full_frame')
    for p,h in pins.items():assert sha(p)==h
    out={'schema':'neuralstudio.nr.v17-recovery-independent.v1','result':'pass_bounded_fault_recovery','fault_patterns':8,'fresh_processes':len(pids),'production_fault_hook_rejected':True,'actual_good_post_history_bytes_checked':count,'rows':rows,'pins':pins,'full_goal_complete':False,'limits':['Original image/zero motion; eight fault patterns, not every possible GPU/runtime failure.','These are test-hook builds; no production default change or performance claim.'],'verifier_sha256':sha(__file__)}
    out['receipt_root_sha256']=canon(out);a.out.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items() if k not in ('rows','pins')},indent=2))
if __name__=='__main__':main()
