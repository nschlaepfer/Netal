#!/usr/bin/env python3
"""Independent CPU-only preflight mutations and runner control-flow witnesses.

Real sealed recipes are read, never executed. Control tests mock GPU execution,
reservation and free-space queries, retaining tiny synthetic output fixtures.
"""
import argparse
import contextlib
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--peer',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    peer=a.peer.resolve();a.out=a.out.resolve();a.out.mkdir(parents=True,exist_ok=False)
    os.chdir(peer)  # The existing recipe runner resolves motion paths from its repo cwd.
    source=peer/'scripts/nr-run-recipe.py';spec=importlib.util.spec_from_file_location('runner',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    recipe=peer/'spec/nr/recipes/v31t/black3-v31t.json';base=json.loads(recipe.read_text());rows=[]
    def check(name,edit,want_rejection=True):
        r=copy.deepcopy(base);edit(r)
        try:errs=m.preflight(r);unexpected=None
        except Exception as e:errs=[];unexpected=repr(e)
        rejected=bool(errs);row={'test':name,'expected_rejection':want_rejection,'rejected':rejected,'errors':errs,'unexpected_exception':unexpected,'passed':unexpected is None and rejected==want_rejection}
        rows.append(row)
    check('reviewed-valid-recipe',lambda r:None,False)
    check('bool-margin',lambda r:r['admission'].update(margin=True))
    check('nan-frame',lambda r:r['admission'].update(per_frame_mib=float('nan')))
    check('inf-fixed',lambda r:r['admission'].update(fixed_mib=float('inf')))
    check('wrong-mode',lambda r:r['admission']['evidence'].update(body_mode='checked'))
    check('wrong-evidence-build',lambda r:r['admission']['evidence'].update(harness_executable_sha256='0'*64))
    check('bool-floor',lambda r:r.update(storage_floor_gib=True))
    def invented_measurement(r):
        r['admission'].update(per_frame_mib=0.001,fixed_mib=0.001)
        for c in r['admission']['evidence']['comparable_receipts']:c['measured_mib_per_process']=0.001
    check('invented-tiny-footprint-with-real-summary-seals',invented_measurement)
    # Isolate the runner's post-process control flow. These bypass preflight only
    # in the mock; no test recipe or fake artifact is submitted to a GPU runner.
    for name,mode in [('overrun-stops-and-retains','overrun'),('reservation-loss-stops','lock'),('floor-refuses-before-execution','floor')]:
        r=copy.deepcopy(base);r['replicas']=r['replicas'][:2]
        if mode=='overrun':r['admission'].update(per_frame_mib=.1,fixed_mib=.1,margin=2)
        rp=a.out/(name+'.json');rp.write_text(json.dumps(r));out=a.out/name;calls=[]
        def fake_run(R,root,rep):
            calls.append(rep['tag']);d=Path(root)/rep['tag'];d.mkdir()
            (d/'synthetic-output.bin').write_bytes(b'X'*(1024*1024 if mode=='overrun' else 16))
            return {'tag':rep['tag'],'passed':True,'frame_status':['synthetic_cpu_only']}
        reserve=(iter([(True,'CPU mock initial'),(True,'CPU mock first'),(False,'CPU mock loss')]) if mode=='lock' else None)
        def held(_):return next(reserve) if reserve else (True,'CPU mock reservation')
        free=(12*2**30 if mode=='floor' else 100*2**30)
        log=io.StringIO()
        with patch.object(m,'preflight',return_value=[]),patch.object(m,'held_reservation',side_effect=held),patch.object(m,'run_one',side_effect=fake_run),patch.object(m.shutil,'disk_usage',return_value=SimpleNamespace(free=free)),patch.object(m.sys,'argv',[str(source),'run',str(rp),'--out',str(out)]),contextlib.redirect_stdout(log):
            try:m.main();exit_code=0
            except SystemExit as e:exit_code=e.code
        (a.out/(name+'.log')).write_text(log.getvalue())
        s=json.loads((out/'run-summary.json').read_text());ok=exit_code!=0 and not s['all_replicas_passed'] and len(calls)==(0 if mode=='floor' else 1)
        if mode=='overrun':ok=ok and (out/'ADMISSION-OVERRUN.json').exists() and (out/calls[0]/'synthetic-output.bin').stat().st_size==1024*1024
        rows.append({'test':name,'passed':ok,'exit_code':exit_code,'mock_executions':len(calls),'summary_sha256':sha(out/'run-summary.json')})
    result={'schema':'neuralstudio.nr.admission-independent-review.v1','scope':'CPU-only; real preflight reads and mocked control flow, no GPU or NR inference','runner_sha256':sha(source),'recipe_sha256':sha(recipe),'rows':rows,'all_passed':all(r['passed'] for r in rows),'failures':[r['test'] for r in rows if not r['passed']]}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();(a.out/'review.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    return 0 if result['all_passed'] else 1
if __name__=='__main__':raise SystemExit(main())
