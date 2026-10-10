#!/usr/bin/env python3
"""Compile and test the isolated encoder primitive on retained RTX operands.

Reserve Mac GPU first. No PC jobs and no archived model-tensor inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--prepared',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--guard-script',required=True,type=Path);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    meta=json.loads((a.prepared/'source-receipt.json').read_text());body=dict(meta);root=body.pop('receipt_root_sha256')
    assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest()==root
    for name,h in meta['files'].items():assert sha(a.prepared/name)==h,name
    shutil.copytree(a.prepared/'source',a.output/'source');commands=[]
    def run(label,args):
        args=[str(x) for x in args]
        with (a.output/(label+'.log')).open('x') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'label':label,'command':args,'exit_code':r.returncode});(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert r.returncode==0,'failed '+label+'; preserve all evidence'
    run('guard',[sys.executable,a.guard_script,'--me','Codex'])
    metal=Path('/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal')
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    run('metal',[metal,'-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off','-c',a.output/'source/nr_probes/qmma_fixed_probe.metal','-o',a.output/'probe.air'])
    run('metallib',[metal.with_name('metallib'),a.output/'probe.air','-o',a.output/'probe.metallib'])
    run('host',['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',a.output/'source/nr_probes/qmma_fixed_probe.mm','-framework','Foundation','-framework','Metal','-o',a.output/'probe'])
    for kind,parent in [('discriminator','discriminators-rtx-v1'),('launch151','launch151-rtx-v1')]:
        reference=repo/'build/nr_qmma_operands'/parent/'output'
        for label in ('a5-a','a5-b','5a-a','5a-b'):
            run(kind+'-'+label,[a.output/'probe',a.output/'probe.metallib',reference/'a5-a.input-after.bin',reference/'a5-a.bin',label[:2],a.output/(kind+'-'+label)])
    run('verify',[sys.executable,repo/'scripts/verify-nr-qmma-fixed.py',a.output,'--output',a.output/'verification.json'])
    verified=json.loads((a.output/'verification.json').read_text())
    result={'schema':'neuralstudio.nr.encoder-fscale-transfer.v1','prepared_root':root,'primitive_verification_root':verified['receipt_root_sha256'],
      'prepared_receipt_sha256':sha(a.prepared/'source-receipt.json'),'primitive_receipt_sha256':sha(a.output/'verification.json'),
      'result':'pass','full_encoder_validated':False,'performance_measured':False,'pc_used':False,'scope':'65 retained independently captured RTX operand sets, two fresh runs per poison. Not model or performance acceptance.'}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest();(a.output/'transfer.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
