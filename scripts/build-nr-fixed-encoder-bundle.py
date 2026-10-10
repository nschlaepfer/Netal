#!/usr/bin/env python3
"""Build a candidate fixed-QMMA Block0/encoder bundle; never promote on build."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);p.add_argument('--hmma',required=True,type=Path);a=p.parse_args();a.output=a.output.resolve();repo=Path(__file__).resolve().parents[1]
    a.output.mkdir(parents=True,exist_ok=False)
    math=json.loads((repo/'build/nr_qmma_fixed/macos-v1/verification.json').read_text());body=dict(math);root=body.pop('receipt_root_sha256')
    assert canonical(body)==root=='e478976e560da01d2fb449252b41e5cfbdcc1f0a0ed976274d4b11f74136862f' and math['result']=='pass'
    for n in ('qmma_fixed.metalh','swin32_mlp.metal'):assert sha(repo/'src/nr_swin'/n)==math['files']['source/nr_swin/'+n]
    metal=Path('/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal')
    try:metal=Path(subprocess.check_output(['xcrun','-f','metal'],text=True,stderr=subprocess.DEVNULL).strip())
    except subprocess.CalledProcessError:pass
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip();commands=[]
    def run(name,args):
        command=[str(x) for x in args]
        with (a.output/(name+'.stdout.log')).open('w') as out,(a.output/(name+'.stderr.log')).open('w') as err:r=subprocess.run(command,stdout=out,stderr=err)
        commands.append({'name':name,'command':command,'exit_code':r.returncode});(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if r.returncode:raise RuntimeError(name+' failed; preserve attempt')
    pre=a.output/'pre';enc=a.output/'encoder'
    run('generate-pre',[sys.executable,repo/'scripts/generate-nr-pre-core.py','--hmma',a.hmma,'--output',pre/'source'])
    run('generate-encoder',[sys.executable,repo/'scripts/generate-nr-prefix22-probe.py','--output',enc/'source'])
    for n in ('qmma_fixed.metalh','mlp_probe.mm','wide_execute.h','wide_boundaries.h','native_encoder_exports.h','native_encoder_exports.metal','swin32_mlp.metal','swin32_qkv.metal','swin32_attention.metal','swin32_block.metal','swin32_chain.metal','swin32_downsample.metal','swin_wide.metal','wide_boundaries.metal'):shutil.copyfile(repo/'src/nr_swin'/n,enc/'source'/n)
    (enc/'weights').mkdir()
    for f in sorted((repo/'build/nr_encoder22/macos-v2/weights').glob('*.bin')):shutil.copyfile(f,enc/'weights'/f.name)
    base=[metal,'-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off']
    run('pre-metal',base+['-DNR_PRE_EXACT_MATH=1','-c',pre/'source/nr_pre/pre.metal','-o',pre/'pre.air'])
    run('pre-metallib',[metal.with_name('metallib'),pre/'pre.air','-o',pre/'pre.metallib'])
    for name,shader,width in [('prefix','prefix8.metal',64),('wide128','wide128.metal',128),('wide256','wide256.metal',256)]:
        run(name+'-metal',base+[f'-DNR_WIDE_CHANNELS={width}','-c',enc/'source'/shader,'-o',enc/(name+'.air')])
        run(name+'-metallib',[metal.with_name('metallib'),enc/(name+'.air'),'-o',enc/(name+'.metallib')])
    cc=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations']
    run('pre-host',cc+[pre/'source/nr_pre/pre_probe.mm','-framework','Foundation','-framework','Metal','-o',pre/'probe'])
    run('encoder-host',cc+[enc/'source/probe.mm','-framework','Foundation','-framework','Metal','-o',enc/'probe'])
    files={str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}
    r={'schema':'neuralstudio.nr.fixed-encoder-candidate-bundle.v1','primitive_root':root,'files':files,
       'comparison_block0_root':'ddc9695ac9012f2450337637c4b0dcb1ae6b5d98b53763aff345e0ad20f53bd3',
       'comparison_encoder_root':'62aafb4f76a52f7bd2430ac3929f33b5c7426b8cc5038993c7eb417caa9f5ee2',
       'model_validation':'pending','arithmetic':'fixed-point QMMA in Block0 and all encoder widths','result':'built'}
    r['receipt_root_sha256']=canonical(r);(a.output/'manifest.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
