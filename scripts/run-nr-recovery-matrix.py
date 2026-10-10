#!/usr/bin/env python3
"""Run the reviewed recovery cases sequentially, audit each, stop on any failure.

Must be launched inside nr-gpu-lock.py run --owner Codex. Writes only beneath
the explicit fresh output root. No benchmark or automatic runtime promotion.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('reviewed',ROOT/'scripts/audit-nr-recovery-run.py')
review=importlib.util.module_from_spec(spec); spec.loader.exec_module(review)


def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--peer-root',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cases',nargs='+',choices=list(review.VARIED_RECIPES),required=True)
    args=parser.parse_args(); peer=args.peer_root.resolve(); out=args.out.resolve()
    if len(set(args.cases)) != len(args.cases):parser.error('duplicate case')
    if not out.is_relative_to(ROOT/'build/nr_recovery_runs') or out.exists():parser.error('fresh scoped output directory required')
    runner=peer/'scripts/nr-run-recipe.py'
    if sha(runner)!=review.RUNNER:parser.error('runner identity changed')
    s=importlib.util.spec_from_file_location('reservation_contract',runner); gate=importlib.util.module_from_spec(s);s.loader.exec_module(gate)
    held,why=gate.held_reservation('Codex')
    if not held:parser.error('coordinated reservation required: '+why)
    # Preflight the entire declared set before allocating the run root.
    recipes={}
    for case in args.cases:
        recipe=peer/f'spec/nr/recipes/recovery/{case}-reacceptance-v17.json'
        if sha(recipe)!=review.VARIED_RECIPES[case]:parser.error('unreviewed recipe '+case)
        errors=gate.preflight(json.loads(recipe.read_text()))
        if errors:parser.error(case+': '+'; '.join(errors))
        recipes[case]=recipe
    out.mkdir()
    summary={'schema':'neuralstudio.nr.recovery-matrix-run.v1','cases_requested':args.cases,'cases':[],
        'all_cases_passed':False,'performance_claim':False,'full_goal_complete':False,
        'driver_sha256':sha(Path(__file__)),'auditor_sha256':sha(ROOT/'scripts/audit-nr-recovery-run.py'),
        'runner_sha256':sha(runner),'reservation_at_start':why}
    def save():
        (out/'matrix-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    save()
    for case,recipe in recipes.items():
        held,why=gate.held_reservation('Codex')
        if not held:summary['error']='reservation lost: '+why;save();return 1
        if sha(recipe)!=review.VARIED_RECIPES[case] or sha(runner)!=summary['runner_sha256'] or sha(ROOT/'scripts/audit-nr-recovery-run.py')!=summary['auditor_sha256']:
            summary['error']='recipe/runner/auditor changed before '+case;save();return 1
        command=[sys.executable,'-u',str(runner),'run',str(recipe),'--owner','Codex','--out',str(out/case)]
        print('START',case,flush=True)
        with (out/(case+'-runner.log')).open('x') as log:
            run=subprocess.run(command,cwd=peer,stdout=log,stderr=subprocess.STDOUT)
        row={'case':case,'recipe_sha256':sha(recipe),'command':command,'run_exit_code':run.returncode}
        summary['cases'].append(row);save()
        if run.returncode:print('STOP inference failed:',case,flush=True);return 1
        command=[sys.executable,str(ROOT/'scripts/audit-nr-recovery-run.py'),'--peer-root',str(peer),'--run',str(out/case),
                 '--body-version','17','--configuration',case,'--out',str(out/(case+'-audit.json'))]
        with (out/(case+'-audit.log')).open('x') as log:
            check=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
        row['audit_command']=command;row['audit_exit_code']=check.returncode
        receipt=out/(case+'-audit.json')
        if receipt.exists():
            data=json.loads(receipt.read_text());row.update(audit_root=data.get('receipt_root_sha256'),audit_result=data.get('result'))
        save()
        if check.returncode:print('STOP independent audit failed:',case,flush=True);return 1
        print('PASS',case,row['audit_root'],flush=True)
    summary['all_cases_passed']=len(summary['cases'])==len(args.cases);save()
    print('ALL REVIEWED CASES PASS',out,flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
