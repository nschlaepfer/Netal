#!/usr/bin/env python3
"""Fresh checked original-image pairs, fail-stop; correctness only."""
import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'build/nr_checked_recovery/config-v1'
RUNNER_SHA='340f4c96a500e856a8c722b3dd7ce1c17814efb5b3b7bf3a5da07d5b372df156'
DESC_SHA='3d3189665884f519bf4b1bfd5fc5389bee755d376116b14790bf7c633e57254e'
TEMPLATE_SHA='0e31c288fc9ca3a0e460881657fcae7c9cd4640cddda1468cde23f35c32039ec'
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
review=module('checked_review',ROOT/'scripts/audit-nr-checked-recovery.py')

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--peer',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--preflight-only',action='store_true');a=p.parse_args()
    peer=a.peer.resolve();out=a.out.resolve();runner=peer/'scripts/nr-run-recipe.py'
    os.chdir(peer)  # The reviewed recipe gate resolves motion paths against its repo.
    review.need(out.is_relative_to(ROOT/'build/nr_checked_recovery') and not out.exists(),'fresh scoped root')
    review.need(review.sha(runner)==RUNNER_SHA,'runner changed');gate=module('recipe_gate',runner)
    review.need(review.sha(CONFIG/'chain.json')==DESC_SHA,'descriptor changed')
    template_path=peer/'build/nr_full/baseline-prod-v1/v11-p1/process-1/diag/frame-0/chain-run.json'
    review.need(review.sha(template_path)==TEMPLATE_SHA,'structural template changed');template=review.read(template_path)
    order=[('v11',1),('v17',1),('v17',2),('v11',2),('v11',3),('v17',3),('v17',4),('v11',4)]
    for arm,i in order:
        recipe=CONFIG/f'{arm}-p{i}.json'
        review.need(review.sha(recipe)==review.RECIPE_PINS[arm][i-1],'recipe changed')
        errors=gate.preflight(review.read(recipe));review.need(not errors,'; '.join(errors))
    print('Eight checked recipes preflight PASS; no inference yet',flush=True)
    if a.preflight_only:return 0
    held,why=gate.held_reservation('Codex');review.need(held,'reservation required: '+why)
    out.mkdir();summary={'schema':'neuralstudio.nr.checked-recovery-run.v1','all_processes_passed':False,
        'runs':[],'paired_checkpoint_checks':[],'runner_sha256':RUNNER_SHA,'driver_sha256':review.sha(__file__),
        'auditor_sha256':review.sha(ROOT/'scripts/audit-nr-checked-recovery.py'),'reservation':why,
        'speedup_accepted':False,'full_goal_complete':False}
    def save():(out/'checked-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save()
    try:
        for n,(arm,i) in enumerate(order):
            held,why=gate.held_reservation('Codex');review.need(held,'reservation lost: '+why)
            review.need(review.sha(runner)==RUNNER_SHA and review.sha(CONFIG/'chain.json')==DESC_SHA,'runner/descriptor drift')
            review.need(review.sha(ROOT/'scripts/audit-nr-checked-recovery.py')==summary['auditor_sha256'],'auditor drift')
            recipe=CONFIG/f'{arm}-p{i}.json';review.need(review.sha(recipe)==review.RECIPE_PINS[arm][i-1],'recipe drift')
            free=shutil.disk_usage(out).free/2**30
            review.need(free-3>=12,'pilot reserve/floor')
            cmd=[sys.executable,'-u',str(runner),'run',str(recipe),'--owner','Codex','--out',str(out/f'{arm}-p{i}')]
            print('START',arm,i,'free GiB',round(free,2),flush=True)
            with (out/f'{arm}-p{i}.log').open('x') as log:r=subprocess.run(cmd,cwd=peer,stdout=log,stderr=subprocess.STDOUT)
            summary['runs'].append({'arm':arm,'pair':i,'exit_code':r.returncode,'free_gib':free,'command':cmd});save()
            review.need(r.returncode==0,'native run failed; preserve and stop')
            if n%2==1:
                for frame in range(3):
                    chains=[review.read(out/f'{v}-p{i}'/f'process-{i}'/f'diag/frame-{frame}/chain-run.json') for v in ('v11','v17')]
                    count=review.compare_checkpoints(*chains,template)
                    summary['paired_checkpoint_checks'].append({'pair':i,'frame':frame,'identities':count})
                save();print('PAIRED CHECKPOINT PASS',i,flush=True)
        summary['all_processes_passed']=True;save()
    except Exception as error:
        summary['error']=str(error);save();print('STOP',error,flush=True);return 1
    cmd=[sys.executable,str(ROOT/'scripts/audit-nr-checked-recovery.py'),'--peer',str(peer),'--root',str(out),'--out',str(out/'independent-audit-v1.json')]
    with (out/'independent-audit.log').open('x') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    # Do not mutate checked-summary after the auditor hashes it.
    print('INDEPENDENT AUDIT EXIT',r.returncode,flush=True)
    return r.returncode

if __name__=='__main__':raise SystemExit(main())
