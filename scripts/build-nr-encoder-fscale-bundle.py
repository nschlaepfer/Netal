#!/usr/bin/env python3
"""Derive a model candidate using ONLY the independently accepted new header."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
from nr_candidate_primitive import FSCALE_ROOT,validate_primitive

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(path):
    r=json.loads(path.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h,path;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--primitive',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--guard-script',required=True,type=Path);a=p.parse_args();a.output=a.output.resolve()
    parent=sealed(a.parent/'manifest.json');proof=sealed(a.primitive/'verification.json');assert proof['receipt_root_sha256']==FSCALE_ROOT
    validate_primitive(a.parent,parent)
    for name,h in parent['files'].items():assert sha(a.parent/name)==h,name
    for name,h in proof['files'].items():assert sha(a.primitive/name)==h,name
    assert not a.output.exists();shutil.copytree(a.parent,a.output,ignore=shutil.ignore_patterns('manifest.json'))
    for name in ('pre/source/nr_swin/qmma_fixed.metalh','encoder/source/qmma_fixed.metalh'):
        shutil.copyfile(a.primitive/'source/nr_swin/qmma_fixed.metalh',a.output/name)
    shutil.copyfile(a.primitive/'verification.json',a.output/'primitive-verification.json')
    pre=a.output/'pre/source/source-manifest.json';meta=json.loads(pre.read_text());meta['arithmetic_derivation']={'parent_manifest_sha256':sha(a.parent/'pre/source/source-manifest.json'),'primitive_root':FSCALE_ROOT}
    meta['files']={str(p.relative_to(pre.parent)):sha(p) for p in sorted(pre.parent.rglob('*')) if p.is_file() and p!=pre};pre.write_text(json.dumps(meta,indent=2)+'\n')
    commands=[]
    def run(label,args):
        args=[str(x) for x in args]
        with (a.output/(label+'.log')).open('x') as log:r=subprocess.run(args,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'label':label,'command':args,'exit_code':r.returncode});(a.output/'fscale-build.json').write_text(json.dumps(commands,indent=2)+'\n');assert r.returncode==0,label
    run('fscale-guard',[sys.executable,a.guard_script,'--me','Codex'])
    metal=Path('/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal');sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    flags=[metal,'-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off']
    for dest,source,options in [('pre/pre','pre/source/nr_pre/pre.metal',['-DNR_PRE_EXACT_MATH=1']),('encoder/prefix','encoder/source/prefix8.metal',['-DNR_WIDE_CHANNELS=64']),('encoder/wide128','encoder/source/wide128.metal',['-DNR_WIDE_CHANNELS=128']),('encoder/wide256','encoder/source/wide256.metal',['-DNR_WIDE_CHANNELS=256'])]:
        run('fscale-'+dest.replace('/','-'),flags+options+['-c',a.output/source,'-o',a.output/(dest+'.air')])
        run('fscale-link-'+dest.replace('/','-'),[metal.with_name('metallib'),a.output/(dest+'.air'),'-o',a.output/(dest+'.metallib')])
    for name in ('pre/probe','encoder/probe'):assert sha(a.parent/name)==sha(a.output/name),'host unexpectedly changed'
    parent_root=parent.pop('receipt_root_sha256');parent['parent_bundle_root']=parent_root;parent['arithmetic_reference_root']=parent['primitive_root'];parent['primitive_root']=FSCALE_ROOT
    parent['arithmetic']='peer float-scaled QMMA with original encoder RNE and finite guards';parent['shader_arithmetic_change']='encoder-fscale';parent['model_validation']='pending';parent['timing_only_host_changes']=False
    parent['files']={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    validate_primitive(a.output,parent);parent['receipt_root_sha256']=canonical(parent);(a.output/'manifest.json').write_text(json.dumps(parent,indent=2)+'\n');print(parent['receipt_root_sha256'])
if __name__=='__main__':main()
