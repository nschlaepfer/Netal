#!/usr/bin/env python3
"""Four fresh processes, two different live images per prepared runtime context."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from datetime import datetime,timezone

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--build',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1];a.build=a.build.resolve();a.output=a.output.resolve()
    build=sealed(a.build/'build.json');assert build['compile_exit']==0 and build['shader_changes'] is False
    for name,h in build['source_files'].items():assert sha(a.build/'source'/name)==h,name
    for name,h in build['assets'].items():assert sha(a.build/'assets'/name)==h,name
    assert sha(a.build/'nr_encoder_runtime.o')==build['object_sha256']
    original=repo/'build/nr_encoder_fscale/live-v1';dense=repo/'build/nr_color_encoder/case3-live-v1';opr=sealed(original/'verification.json');dpr=sealed(dense/'independent-audit-v1.json');assert opr['result']==dpr['result']=='pass'
    cases=[(repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le',original/'a5-a'),(repo/'build/nr_color_cases/rtx-v1/case3-retry-v1/input-color.rgba16le',dense/'a5-a')]
    a.output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(repo/'src/nr_encoder_runtime/nr_encoder_runtime_probe.mm',a.output/'probe-source.mm')
    files=[a.build/'probe',a.build/'nr_encoder_runtime.o',a.output/'probe-source.mm']+[a.build/'source'/n for n in build['source_files']]+[a.build/'assets'/n for n in build['assets']]
    for image,ref in cases:
        files.append(image);files.extend(ref/'encoder'/n for n in ('block22-middle.e4m3','block22-native.e4m3','block14-native.e4m3','block8-native.e4m3','block4-native.e4m3'));files.append(ref/'block0/skip.e4m3')
    pins={str(f):sha(f) for f in files};plan={'schema':'neuralstudio.nr.encoder-runtime-test-plan.v1','build_root':build['receipt_root_sha256'],'build_dir':str(a.build),'artifacts':pins,'original_producer_root':opr['receipt_root_sha256'],'dense_producer_audit_root':dpr['receipt_root_sha256'],'comparison_role':'CPU after live API returns, never runtime ingress','minimum_free_gib':14}
    plan['receipt_root_sha256']=canonical(plan);(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n');commands=[];rows=[];pids=set();observed={}
    for poison,reverse in [('a5',False),('a5',True),('5a',False),('5a',True)]:
        assert shutil.disk_usage(a.output).free>=14*1024**3,'storage floor; preserve attempt'
        key=poison+('-reverse' if reverse else '-forward');order=list(reversed(cases)) if reverse else cases
        command=[str(a.build/'probe'),str(a.build/'assets'),str(order[0][0]),str(order[0][1]),str(order[1][0]),str(order[1][1]),poison,str(a.output/key)]
        start=datetime.now(timezone.utc).isoformat()
        with (a.output/(key+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'key':key,'command':command,'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode});(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert r.returncode==0,'native runtime failed: '+key
        rec=json.loads((a.output/key/'run.json').read_text());assert rec['result']=='pass' and rec['pid'] not in pids and rec['context_reused'] and rec['activation_file_ingress'] is False;pids.add(rec['pid'])
        assert len(rec['negative_controls'])==6 and all(x['code']==2 for x in rec['negative_controls']) and len(rec['frames'])==2
        for i,frame in enumerate(rec['frames']):
            assert frame['input_sha256']==sha(order[i][0]) and frame['poison']==(int(poison,16)^(255 if i else 0)) and frame['mismatching_bytes']==0 and frame['input_immutable'] and frame['gpu_complete'] and frame['prior_outputs_still_intact'] and frame['expected_gpu_input'] is False
            assert frame['guarded_outputs_checked']>0 and frame['coverage_values_checked']>0
            assert len(frame['ports'])==6
            for name,meta in frame['ports'].items():
                dest=a.output/key/f'frame{i}'/(name+'.e4m3');assert dest.stat().st_size==meta['bytes'] and sha(dest)==meta['sha256'] and meta['mismatching_bytes']==0
                obskey=(frame['input_sha256'],name);assert obskey not in observed or observed[obskey]==meta['sha256'];observed[obskey]=meta['sha256']
        for f,h in pins.items():assert sha(Path(f))==h,'input/build drift: '+f
        rows.append({'key':key,'pid':rec['pid'],'receipt_sha256':sha(a.output/key/'run.json'),'frames':rec['frames']});print('completed '+key,flush=True)
    report={'schema':'neuralstudio.nr.encoder-runtime-characterization-matrix.v1','result':'pass','plan_root':plan['receipt_root_sha256'],'runs':rows,'fresh_processes':4,'live_image_frames':8,'negative_controls':24,'activation_file_ingress':False,'shader_changes':False,'full_model_connected':False,'temporal_history_supported':False,'scope':'Original+dense images in both orders and opposite poisons on each reused context, exact six-buffer comparisons and prior-output lifetime checks. Producer only; body not connected. Coverage scans run in native API; no raw coverage dump in this compact harness.'}
    report['receipt_root_sha256']=canonical(report);(a.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n');print(report['receipt_root_sha256'])
if __name__=='__main__':main()
