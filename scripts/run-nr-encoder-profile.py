#!/usr/bin/env python3
"""Run the timing-only live encoder test and retain observed workload context."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess
import sys
def context():
    processes=subprocess.check_output(['ps','-axo','pid,pcpu,comm'],text=True)
    netal=[line.strip() for line in processes.splitlines() if '/Netal.app/' in line or line.rstrip().endswith('/Netal')]
    return {'utc':datetime.now(timezone.utc).isoformat(),'cpu':subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
            'netal_processes':netal,'global_gpu_idleness_established':False,'user_apps_changed':False}
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path);p.add_argument('--bundle',required=True,type=Path);a=p.parse_args();a.attempt=a.attempt.resolve();assert not a.attempt.exists()
    repo=Path(__file__).resolve().parents[1];before=context();command=[sys.executable,str(repo/'scripts/test-nr-live-encoder.py'),'--attempt',str(a.attempt),'--candidate-bundle',str(a.bundle.resolve())]
    a.attempt.parent.mkdir(parents=True,exist_ok=True)
    with a.attempt.with_suffix('.stdout.log').open('x') as out,a.attempt.with_suffix('.stderr.log').open('x') as err:r=subprocess.run(command,stdout=out,stderr=err)
    after=context()
    if a.attempt.exists():(a.attempt/'timing-environment.json').write_text(json.dumps({'schema':'neuralstudio.nr.encoder-profile-context.v1','before':before,'after':after,
        'command':command,'exit_code':r.returncode,'timing_scope':'Observed instrumented acceptance harness under normal desktop load; not a clean GPU or game-FPS benchmark.'},indent=2)+'\n')
    return r.returncode
if __name__=='__main__':raise SystemExit(main())
