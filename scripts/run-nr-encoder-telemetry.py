#!/usr/bin/env python3
"""Compare output-only telemetry on/off with unchanged checked encoder execution."""
import argparse
from collections import defaultdict
from datetime import datetime,timezone
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def sealed(p):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--build',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];build=a.build.resolve();pb=sealed(build/'build.json');runtime=Path(pb['runtime_build']);rb=sealed(runtime/'build.json')
    assert pb['exit_code']==0 and rb['receipt_root_sha256']==pb['runtime_root'] and pb['runtime_root'] in ('076277f9669d9064d3250d97f8f7f207aca7797df2fd6d85a5a7460808a758d5','e4a60e4442cf88efb5cf09bfa631da81268ea86a39fce96c877ad030dacdc6aa')
    if rb.get('serialization_in_copy_accessor'):
        source=(runtime/'source/nr_encoder_runtime.mm').read_text().split('int nr_encoder_run_frame(',1)[1].split('#include "nr_encoder_temporal_runtime.inc"',1)[0]
        assert 'NSJSONSerialization' not in source and 'NrProfileJson' not in source,'telemetry serialization still inside the encoder call'
    assert sha(runtime/'nr_encoder_runtime.o')==pb['runtime_object_sha256']==rb['object_sha256']
    pins={}
    for n,h in pb['files'].items():assert sha(build/n)==h;pins[str(build/n)]=h
    for kind,folder in [('source_files','source'),('assets','assets')]:
        for n,h in rb[kind].items():assert sha(runtime/folder/n)==h;pins[str(runtime/folder/n)]=h
    cases=[(repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le',repo/'build/nr_encoder_fscale/live-v1/a5-a'),
           (repo/'build/nr_color_cases/rtx-v1/case3-retry-v1/input-color.rgba16le',repo/'build/nr_color_encoder/case3-live-v1/a5-a')]
    for image,ref in cases:
        pins[str(image)]=sha(image)
        for p in [ref/'block0/skip.e4m3']+[ref/f'encoder/{n}.e4m3' for n in ('block4-native','block8-native','block14-native','block22-native','block22-middle')]:pins[str(p)]=sha(p)
    assert not a.output.exists();a.output.mkdir(parents=True)
    rows=[];pids=set();seen={}
    configs=[('off-a5-forward','a5',False,False),('on-a5-forward','a5',False,True),('on-a5-reverse','a5',True,True),('on-5a-forward','5a',False,True),('on-5a-reverse','5a',True,True)]
    for label,poison,reverse,enabled in configs:
        assert shutil.disk_usage(a.output).free>=14*1024**3,'storage floor'
        order=cases[::-1] if reverse else cases;dest=a.output/label
        cmd=[str(build/'probe'),str(runtime/'assets'),str(order[0][0]),str(order[0][1]),str(order[1][0]),str(order[1][1]),poison,str(dest)]+(['--telemetry'] if enabled else [])
        started=datetime.now(timezone.utc).isoformat()
        with (a.output/(label+'.log')).open('x') as log:rc=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT).returncode
        execution={'label':label,'command':cmd,'started_utc':started,'ended_utc':datetime.now(timezone.utc).isoformat(),'exit_code':rc};(a.output/(label+'-execution.json')).write_text(json.dumps(execution,indent=2)+'\n')
        assert rc==0,'preserve failed probe '+label
        r=read(dest/'run.json');assert r['result']=='pass' and r['pid'] not in pids and len(r['frames'])==2;pids.add(r['pid'])
        assert len(r['negative_controls'])==6 and all(x['code']==2 for x in r['negative_controls'])
        assert not any(dest.glob('diagnostics*')),'tensor diagnostics unexpectedly written'
        for i,f in enumerate(r['frames']):
            assert f['input_sha256']==sha(order[i][0]) and f['mismatching_bytes']==0 and f['gpu_complete'] and f['input_immutable'] and f['prior_outputs_still_intact'] and not f['expected_gpu_input']
            assert f['guarded_outputs_checked']==1039 and f['coverage_values_checked']==5900107776
            assert f['poison']==int(poison,16)^(255 if i else 0)
            for n,m in f['ports'].items():
                actual=dest/f'frame{i}'/(n+'.e4m3');ref=order[i][1]/('block0/skip.e4m3' if n=='block0-skip' else 'encoder/'+n+'.e4m3')
                assert actual.stat().st_size==m['bytes'] and sha(actual)==m['sha256']==sha(ref)
                key=(f['input_sha256'],n)
                assert key not in seen or seen[key]==m['sha256'];seen[key]=m['sha256']
            item={'label':label,'pid':r['pid'],'frame':i,'input_sha256':f['input_sha256'],'poison':f['poison'],'telemetry_enabled':enabled,'api_wall_ms_checked':f['wall_ms_diagnostic_only'],'ports':f['ports'],'native_receipt_sha256':sha(dest/'run.json')}
            if enabled:
                profile=read(dest/f'telemetry-frame{i}.json');assert profile['schema']=='neuralstudio.nr.encoder-timing.v2' and profile['pid']==r['pid']
                assert profile['generation']==i+1 and profile['input_sha256']==f['input_sha256'] and profile['frame_index']==0 and profile['reset']
                assert profile['guards_unchanged'] and profile['input_blits_included'] and not profile['tensor_diagnostics_enabled']
                if rb.get('serialization_in_copy_accessor'):assert profile['serialization_in_copy_accessor'] is True
                commands=[x for x in profile['records'] if x['command']]
                assert len(commands)>20 and all(x['gpu_timestamp_valid'] and x['gpu_start_s']>0 and x['gpu_end_s']>=x['gpu_start_s'] for x in commands),'GPU timing unavailable or invalid'
                assert sum(x['label']=='encoder.input_texture_blit' for x in commands)==2
                gpu=defaultdict(float)
                for x in commands:assert math.isfinite(x['gpu_ms']) and x['gpu_ms']>=0;gpu[x['label']]+=x['gpu_ms']
                item.update(gpu_command_ms=sum(gpu.values()),gpu_ms_by_label=dict(gpu),command_count=len(commands),gpu_span_including_host_gaps_ms=(max(x['gpu_end_s'] for x in commands)-min(x['gpu_start_s'] for x in commands))*1000,profile_sha256=sha(dest/f'telemetry-frame{i}.json'))
            else:assert not list(dest.glob('telemetry-frame*.json'))
            rows.append(item)
        print(label+' checked',flush=True)
    for p,h in pins.items():assert sha(Path(p))==h,'artifact drift '+p
    r={'schema':'neuralstudio.nr.encoder-telemetry-characterization.v1','result':'pass_checked_producer_and_telemetry','build_root':rb['receipt_root_sha256'],'probe_root':pb['receipt_root_sha256'],'runs':rows,'artifacts':pins,
       'fresh_processes':5,'native_producer_frames':10,'telemetry_frames':8,'shader_changes':False,'validation_relaxed':False,'accepted_speedup':False,'full_model_connected':False,
       'scope':'Original/dense reset frames in both orders and both poisons, unchanged six-port RTX comparisons. Producer-only checked instrumentation; not a resident/quiet benchmark. API wall excludes probe retention/reference comparison and includes all runtime guards and snapshot construction. '+('Telemetry JSON serialization occurs later in the copy accessor, outside the API timer.' if rb.get('serialization_in_copy_accessor') else 'This earlier build also serializes telemetry inside the timed encoder call.'),
       'limits':['GPU active duration is the sum of command GPU intervals, not end-to-end latency.','GPU span includes CPU gaps between submissions.','Host scopes/phases overlap GPU waits and must not be added together.','Reported times include cold/warm context effects and ambient desktop load; no speedup acceptance.'],
       'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);(a.output/'verification.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('runs','artifacts')},indent=2))


if __name__=='__main__':main()
