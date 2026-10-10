#!/usr/bin/env python3
"""Independent rehash/identity/timing review of the v3 checked producer experiment."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p,root):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==root;return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    root=repo/'build/nr_encoder_telemetry';matrix=root/'matrix-v2';runtime=root/'zero-v3'
    run=sealed(matrix/'verification.json','0b478565d8803d2f1e7d5534a340718563381a77201eec41608be91a73d74ff0')
    rb=sealed(runtime/'build.json','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
    assert sha(runtime/'nr_encoder_runtime.o')==rb['object_sha256']=='942541e17fa3c733605a65813a35a84853ab781ac4501b8695c4f71c35f953de'
    for path,h in run['artifacts'].items():assert sha(Path(path))==h,'artifact drift '+path
    source=(runtime/'source/nr_encoder_runtime.mm').read_text();core=source.split('int nr_encoder_run_frame(',1)[1].split('#include "nr_encoder_temporal_runtime.inc"',1)[0]
    assert 'NSJSONSerialization' not in core and 'NrProfileJson' not in core and 'NrProfileDictionary' in core
    assert 'NSJSONSerialization' in source.split('int nr_encoder_copy_telemetry(',1)[1]
    pids=set();measurements=[];port_bytes=0
    configs=[('off-a5-forward',False,False,165),('on-a5-forward',True,False,165),('on-a5-reverse',True,True,165),('on-5a-forward',True,False,90),('on-5a-reverse',True,True,90)]
    inputs=['81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342','598bb0d29e73b30a2fb74074a8e582b7231e151920bdb52f71d436a0a3c9654e']
    references=[repo/'build/nr_encoder_fscale/live-v1/a5-a',repo/'build/nr_color_encoder/case3-live-v1/a5-a']
    for label,enabled,reverse,poison in configs:
        d=matrix/label;r=read(d/'run.json');assert r['result']=='pass' and r['pid'] not in pids;pids.add(r['pid'])
        assert len(r['frames'])==2 and len(r['negative_controls'])==6 and all(x['code']==2 for x in r['negative_controls'])
        for i,f in enumerate(r['frames']):
            case=1-i if reverse else i;assert f['input_sha256']==inputs[case] and f['poison']==(poison^(255 if i else 0))
            assert f['mismatching_bytes']==0 and f['guarded_outputs_checked']==1039 and f['coverage_values_checked']==5900107776
            assert f['input_immutable'] and f['gpu_complete'] and f['prior_outputs_still_intact'] and not f['expected_gpu_input']
            for name,meta in f['ports'].items():
                actual=d/f'frame{i}'/(name+'.e4m3');expected=references[case]/('block0/skip.e4m3' if name=='block0-skip' else 'encoder/'+name+'.e4m3')
                assert actual.stat().st_size==expected.stat().st_size==meta['bytes'] and sha(actual)==meta['sha256']==sha(expected);port_bytes+=meta['bytes']
            if not enabled:assert not(d/f'telemetry-frame{i}.json').exists();continue
            q=read(d/f'telemetry-frame{i}.json');assert q['pid']==r['pid'] and q['input_sha256']==f['input_sha256'] and q['generation']==i+1 and q['reset'] and q['frame_index']==0
            assert q['serialization_in_copy_accessor'] and q['guards_unchanged'] and not q['tensor_diagnostics_enabled'] and q['input_blits_included']
            commands=[x for x in q['records'] if x['command']];assert len(commands)==139 and sum(x['label']=='encoder.input_texture_blit' for x in commands)==2
            for x in commands:assert x['gpu_timestamp_valid'] and x['gpu_start_s']>0 and math.isclose(x['gpu_ms'],(x['gpu_end_s']-x['gpu_start_s'])*1000,rel_tol=1e-8,abs_tol=1e-8)
            measurements.append({'case':case,'pid':r['pid'],'api_wall_ms':f['wall_ms_diagnostic_only'],'gpu_active_ms':sum(x['gpu_ms'] for x in commands),'profile_sha256':sha(d/f'telemetry-frame{i}.json')})
    ctx=read(root/'context-test-v2/run.json');assert ctx['result']=='pass' and ctx['two_prepared_states'] and ctx['cross_context_isolation'] and ctx['disable_is_context_local'] and ctx['rejection_clears_snapshot'] and ctx['prior_results_intact'] and not ctx['expected_gpu_input']
    for i,f in enumerate(ctx['frames']):
        q=read(root/f'context-test-v2/context{i}.json');assert f['input_sha256']==q['input_sha256']==inputs[i] and q['generation']==1 and q['serialization_in_copy_accessor']
    stats={str(i):{'gpu_active_ms_median':statistics.median(x['gpu_active_ms'] for x in measurements if x['case']==i),'checked_api_wall_ms_median':statistics.median(x['api_wall_ms'] for x in measurements if x['case']==i)} for i in range(2)}
    result={'schema':'neuralstudio.nr.encoder-telemetry-independent-audit.v1','result':'pass_v3_checked_telemetry','runtime_root':rb['receipt_root_sha256'],'matrix_root':run['receipt_root_sha256'],'retained_port_bytes_rechecked':port_bytes,
            'telemetry_frames':len(measurements),'fresh_matrix_processes':len(pids),'prepared_context_isolation_receipt_sha256':sha(root/'context-test-v2/run.json'),'measurements':measurements,'case_medians':stats,
            'telemetry_serialization_outside_encoder_API':True,'shader_changes':False,'validation_relaxed':False,'full_model_connected':False,'accepted_speedup':False,
            'scope':'Producer-only checked instrumentation. JSON serialization is outside the API timer; snapshot construction, guards, cold/warm effects and ambient load remain. Not a resident/quiet benchmark.',
            'supplements':'The immutable matrix-v2 summary used generic earlier wording about optional serialization; source and this audit clarify v3 timing scope without rewriting that receipt.',
            'verifier_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='measurements'},indent=2))


if __name__=='__main__':main()
