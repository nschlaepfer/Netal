#!/usr/bin/env python3
"""Run two controls, then four fresh combined RTX sequences; fail closed per run."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import traceback


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sealed(p,root):
    r=read(p);b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==root;return r
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and job.name=='nr-color-reset-combined-20260906-v1'
    assert not (job/'matrix.json').exists(),'preserve matrix'
    auditor=job/'build-nr-combined-reference.py';parser=job/'build-rtx-oracle-manifest.py'
    assert sha(auditor)=='de077efdeb36de327d94a45f505ec9c0b254e93c1d41c1fa50be23764a5144c7'
    assert sha(parser)=='4620d03e2892a2279e35641b96b9f90937063c8b0b88c7fe13bfbc50cb78d6e2'
    m=load('combined',auditor)
    controls={
      'control-static-spatial-a5':('static',2,job.parent/'nr-color-reset-spatial-20260906-v1/motion2-reset100-a5-a-reference.json','f221d8b0cc3926bb08ecb48c0cc496824fa8deb4cc6b26a6a5fa1006388f2a00'),
      'control-mixed-zero-a5':('mixed',0,job.parent/'nr-color-reset-images-20260906-v1/mixed-reset100-a5-a-reference.json','ca18406994a7fb76f65fe446621d6f9dc766654bd97188ac51333b70f8b6e3c1'),
    }
    rows=[];pids=set();baseline=None;active_label=None
    configs=[(label,x[0],x[1],'a5') for label,x in controls.items()]
    configs += [('combined-'+p+'-'+i,'mixed',2,p) for p in ('a5','5a') for i in ('a','b')]
    try:
        for label,sequence,motion,poison in configs:
            active_label=label
            assert shutil.disk_usage(job).free>=12*1024**3,'storage floor; preserve all captures'
            reference=job/(label+'-reference.json');assert not reference.exists() and not(job/label).exists()
            cmd=['powershell.exe','-NoLogo','-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',str(job/'source/run-nr-reset-sequence.ps1'),'-JobRoot',str(job),'-SecondReset','0','-ColorCase','0','-MotionCase',str(motion),'-ImageSequence',sequence,'-Poison',poison,'-Label',label]
            with (job/(label+'-driver.log')).open('x') as log:run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            assert run.returncode==0,'capture execution failed; inspect preserved driver/capture logs'
            r=m.build(job,label,parser)
            assert r['pid'] not in pids;pids.add(r['pid'])
            if label in controls:
                parent=sealed(controls[label][2],controls[label][3])
                assert r['outputs']==parent['outputs'] and r['history_sha256']==parent['history_sha256']
                assert [(b['launch'],b['ports']) for b in r['boundaries']]==[(b['launch'],b['ports']) for b in parent['boundaries']],'control encoder ports changed'
                r['control_parent_root']=parent['receipt_root_sha256']
            else:
                assert len(rows)>=2 and all('control_parent_root' in x for x in rows[:2])
                if baseline:
                    for key in ('outputs','history_sha256','input_hashes','graph_roots','motion_sha256'):assert r[key]==baseline[key],key
                    assert [b['ports'] for b in r['boundaries']]==[b['ports'] for b in baseline['boundaries']]
                else:baseline=r
            r['receipt_root_sha256']=canonical(r);reference.write_text(json.dumps(r,indent=2)+'\n');rows.append(r)
            print(json.dumps({'label':label,'reference_root':r['receipt_root_sha256'],'outputs':r['outputs']}),flush=True)
        result={'schema':'neuralstudio.nr.combined-RTX-matrix.v1','result':'pass_bounded_combined_RTX_reference','controls':[{k:r[k] for k in ('label','receipt_root_sha256','control_parent_root')} for r in rows[:2]],
                'runs':[{k:r[k] for k in ('label','pid','poison','receipt_root_sha256')} for r in rows[2:]],'source_root':baseline['source_root'],
                'outputs':baseline['outputs'],'input_hashes':baseline['input_hashes'],'motion_sha256':baseline['motion_sha256'],
                'boundary_ports':[b['ports'] for b in baseline['boundaries']],'history_sha256':baseline['history_sha256'],
                'all_payloads_rehashed_on_rig':True,'mac_parity':False,'performance_benchmark':False,
                'auditor_sha256':sha(auditor),'controller_sha256':sha(Path(__file__))}
    except Exception as e:
        result={'schema':'neuralstudio.nr.combined-RTX-matrix.v1','result':'fail','active_label':active_label,'error':str(e),'traceback':traceback.format_exc(),'completed_reference_roots':[r['receipt_root_sha256'] for r in rows],'controller_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result);(job/'matrix.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2),flush=True)
    if result['result']=='fail':raise SystemExit(3)


if __name__=='__main__':main()
