#!/usr/bin/env python3
"""Fresh case-bound Block0 runs. All outputs are live; RTX pool is comparison only."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(path):
    r=json.loads(path.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle',required=True,type=Path);p.add_argument('--reference',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--guard-script',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    host=sealed(a.bundle/'case-host.json');ref=sealed(a.reference/'reference.json');assert host['reference_root']==ref['receipt_root_sha256'] and ref['fresh_prefix_matrix_proven'] and ref['runtime_input_use_forbidden']
    assert sha(Path(host['image']))==host['input_sha256']==ref['input_sha256'] and sha(Path(host['comparison_pool']))==ref['comparison_pool_sha256']
    pins={str(a.bundle/'probe'):host['probe_sha256'],str(a.bundle/'pre.metallib'):host['metallib_sha256'],str(a.bundle/'source/nr_pre/pre_probe.mm'):host['source_sha256'],str(a.bundle/'source/source-manifest.json'):host['source_manifest_sha256']}
    for path,h in pins.items():assert sha(Path(path))==h,path
    a.output.mkdir(parents=True,exist_ok=False);commands=[];runs=[];pids=set();failures=[]
    weights=repo/'build/nr_pre/rtx-feature-source-v1/weights.bin';math=repo/'build/nr_noise_math/corrections-v1'
    for label in ('a5-a','a5-b','5a-a','5a-b'):
        assert shutil.disk_usage(a.output).free>=14*1024**3,'less than14GiB free; preserve attempt'
        subprocess.run([sys.executable,a.guard_script,'--me','Codex'],check=True)
        out=a.output/label;command=[str(x) for x in (a.bundle/'probe',a.bundle/'pre.metallib',Path(host['image']),weights,Path(host['comparison_pool']),label[:2],out,math/'instruction-deltas.u8',math/'instruction-escapes.u32')]
        started=datetime.now(timezone.utc).isoformat()
        with (a.output/(label+'.log')).open('x') as log:run=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'label':label,'command':command,'start_utc':started,'end_utc':datetime.now(timezone.utc).isoformat(),'exit_code':run.returncode});(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert run.returncode in (0,3),'runtime failed; preserve log'
        r=json.loads((out/'run.json').read_text());assert r['pid'] not in pids and r['poison']==int(label[:2],16);pids.add(r['pid'])
        assert r['result']==('pass' if run.returncode==0 else 'fail'),'status/exit disagreement'
        assert r['input_sha256']==ref['input_sha256'] and r['expected_gpu_input'] is False and r['live_device_handoffs'] and r['instruction_corrections']
        assert r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'memory correctness failure'
        for name,count in [('skip',70778880),('encoder',17694720)]:
            coverage=np.memmap(out/(name+'.coverage.bin'),dtype='<u4',mode='r');assert coverage.size==count and np.all(coverage==1)
            assert sha(out/(name+'.e4m3'))==r[name+'_sha256']
            assert (r[name+'_sha256']==ref['ports'][name])==(r[name+'_mismatching_bytes']==0),'comparison summary disagrees with actual bytes'
            if r[name+'_sha256']!=ref['ports'][name]:failures.append({'run':label,'port':name,'actual':r[name+'_sha256'],'rtx':ref['ports'][name],'mismatching_bytes':r[name+'_mismatching_bytes']})
        runs.append({'label':label,'pid':r['pid'],'receipt_sha256':sha(out/'run.json'),'skip_sha256':r['skip_sha256'],'encoder_sha256':r['encoder_sha256'],'result':r['result']})
        for path,h in pins.items():assert sha(Path(path))==h,'artifact changed'
    invariance=len({r['skip_sha256'] for r in runs})==len({r['encoder_sha256'] for r in runs})==1
    result={'schema':'neuralstudio.nr.varied-color-native-block0.v1','case':ref['case'],'reference_root':ref['receipt_root_sha256'],'case_host_root':host['receipt_root_sha256'],
      'runs':runs,'failures':failures,'output_poison_invariance':invariance,'fresh_processes':4,'checked_output_bytes':4*(70778880+17694720),'shader_changes':False,
      'expected_gpu_input':False,'result':'pass' if not failures and invariance else 'fail','scope':'This color case through native Block0 only, with original-image ingress, four fresh dual-poison passes and raw coverage. Not encoder/body/final-frame parity for the new color.'}
    result['receipt_root_sha256']=canonical(result);(a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if result['result']!='pass':raise SystemExit(3)
if __name__=='__main__':main()
