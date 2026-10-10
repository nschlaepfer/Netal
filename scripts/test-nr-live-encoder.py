#!/usr/bin/env python3
"""Fresh original-image Block0 -> layout -> encoder22, without recorded ingress.

Reserve the Mac GPU in the shared journal before invoking this workload.
Preserve all outputs/logs on any failure. Fable owns the subsequent body driver.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
from nr_candidate_primitive import validate_primitive

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def checked(p,root):
    r=json.loads(p.read_text());body=dict(r);got=body.pop('receipt_root_sha256')
    if got!=root or canonical(body)!=got or r['result']!='pass':raise ValueError('parent acceptance identity: '+str(p))
    return r
def now():return datetime.now(timezone.utc).isoformat()
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path);p.add_argument('--candidate-bundle',type=Path);p.add_argument('--minimum-free-gib',type=float,default=22);a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    if not 8<=a.minimum_free_gib<=1024:raise ValueError('minimum free space must be8..1024GiB')
    a.attempt=a.attempt.resolve();a.attempt.mkdir(parents=True,exist_ok=False)
    pre=repo/'build/nr_pre/native-macos-v2';enc=repo/'build/nr_encoder22/macos-v2';math=repo/'build/nr_noise_math/corrections-v1'
    pr=checked(pre/'verification.json','ddc9695ac9012f2450337637c4b0dcb1ae6b5d98b53763aff345e0ad20f53bd3')
    er=checked(enc/'verification.json','62aafb4f76a52f7bd2430ac3929f33b5c7426b8cc5038993c7eb417caa9f5ee2')
    for name in ('pre.metallib','probe'):
        if sha(pre/name)!=pr['files'][name]:raise ValueError('pre artifact drift')
    for name,key in [('probe','host_sha256'),('prefix.metallib','metallib_sha256'),('wide128.metallib','c128_metallib_sha256'),('wide256.metallib','c256_metallib_sha256')]:
        if sha(enc/name)!=er[key]:raise ValueError('encoder artifact drift '+name)
    candidate=None
    if a.candidate_bundle:
        bundle=a.candidate_bundle.resolve();candidate=json.loads((bundle/'manifest.json').read_text());body=dict(candidate);claimed=body.pop('receipt_root_sha256')
        if canonical(body)!=claimed or candidate['schema']!='neuralstudio.nr.fixed-encoder-candidate-bundle.v1':raise ValueError('candidate primitive/provenance identity')
        validate_primitive(bundle,candidate)
        if candidate['comparison_block0_root']!=pr['receipt_root_sha256'] or candidate['comparison_encoder_root']!=er['receipt_root_sha256']:raise ValueError('candidate comparison parents')
        for name,h in candidate['files'].items():
            if sha(bundle/name)!=h:raise ValueError('candidate compiled source drift '+name)
        pre=bundle/'pre';enc=bundle/'encoder'
    source=a.attempt/'source';(source/'nr_pre').mkdir(parents=True);(source/'nr_swin').mkdir()
    for name in ('entry_layout.metal','entry_layout_probe.mm'):shutil.copyfile(repo/'src/nr_pre'/name,source/'nr_pre'/name)
    shutil.copyfile(repo/'src/nr_swin/mlp_probe.mm',source/'nr_swin/mlp_probe.mm');shutil.copyfile(Path(__file__),source/'test-nr-live-encoder.py')
    shutil.copyfile(Path(__file__).with_name('nr_candidate_primitive.py'),source/'nr_candidate_primitive.py')
    metal=Path('/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin/metal')
    try:metal=Path(subprocess.check_output(['xcrun','-f','metal'],text=True,stderr=subprocess.DEVNULL).strip())
    except subprocess.CalledProcessError:pass
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip();layout=a.attempt/'entry.metallib';host=a.attempt/'entry_probe'
    commands=[]
    def run(label,args,logroot):
        # Each native stage can retain close to1GiB. Leave headroom above the
        # shared20GiB reserve; preserve a partial attempt rather than clean it.
        if shutil.disk_usage(a.attempt).free<a.minimum_free_gib*1024**3:
            raise RuntimeError(f'less than{a.minimum_free_gib:g}GiB free before '+label+'; preserve attempt and arrange storage')
        start=now();command=[str(x) for x in args];logroot.mkdir(parents=True,exist_ok=True)
        with (logroot/(label+'.stdout.log')).open('w') as out,(logroot/(label+'.stderr.log')).open('w') as err:
            result=subprocess.run(command,stdout=out,stderr=err)
        commands.append({'label':label,'command':command,'start_utc':start,'end_utc':now(),'exit_code':result.returncode})
        (a.attempt/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        if result.returncode:raise RuntimeError(label+' failed; preserve '+str(logroot))
    run('compile-metal',[metal,'-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off','-c',source/'nr_pre/entry_layout.metal','-o',a.attempt/'entry.air'],a.attempt)
    run('link-metal',[metal.with_name('metallib'),a.attempt/'entry.air','-o',layout],a.attempt)
    run('compile-host',['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',source/'nr_pre/entry_layout_probe.mm','-framework','Foundation','-framework','Metal','-o',host],a.attempt)
    image=repo.parent/'work/rtx-oracle-fp8/capture/input-color.rgba16le';weights=repo/'build/nr_pre/rtx-feature-source-v1/weights.bin'
    expected_pool=repo.parent/'work/rtx-oracle-fp8/block0.pool.bin'
    pinned={str(x):sha(x) for x in [pre/'probe',pre/'pre.metallib',enc/'probe',enc/'prefix.metallib',enc/'wide128.metallib',enc/'wide256.metallib',image,weights,math/'instruction-deltas.u8',math/'instruction-escapes.u32']+sorted((enc/'weights').glob('*.bin'))}
    manifest={'schema':'neuralstudio.nr.live-encoder-run-plan.v1','parent_block0_root':pr['receipt_root_sha256'],'parent_encoder_root':er['receipt_root_sha256'],'minimum_free_gib':a.minimum_free_gib,
              'artifacts':pinned,'source_files':{str(x.relative_to(a.attempt)):sha(x) for x in sorted(source.rglob('*')) if x.is_file()},
              'layout_artifacts':{'entry.metallib':sha(layout),'entry_probe':sha(host)},
              'recorded_pool_role':'CPU comparison only inside Block0, opened after native execution; never a GPU or encoder input',
              'entry':'original image -> fresh native Block0 -> live layout permutation -> encoder1-22',
              'handoff_transport':'fresh producer output files between separately launched native processes; not yet a single resident runtime'}
    if candidate:manifest['candidate_bundle']={'path':str(a.candidate_bundle.resolve()),'root':candidate['receipt_root_sha256'],'manifest_sha256':sha(a.candidate_bundle.resolve()/'manifest.json'),'primitive_root':candidate['primitive_root']}
    manifest['receipt_root_sha256']=canonical(manifest);(a.attempt/'plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            out=a.attempt/f'{poison}-{ordinal}';out.mkdir()
            run('block0',[pre/'probe',pre/'pre.metallib',image,weights,expected_pool,poison,out/'block0',math/'instruction-deltas.u8',math/'instruction-escapes.u32'],out)
            run('entry-layout',[host,layout,out/'block0',poison,out/'entry-layout'],out)
            run('encoder',[enc/'probe',enc/'prefix.metallib',out/'entry-layout/encoder-window.e4m3',enc/'weights',poison,out/'encoder',22,enc/'wide128.metallib',enc/'wide256.metallib'],out)
            for path,h in pinned.items():
                if sha(Path(path))!=h:raise ValueError('material input changed: '+path)
            print('completed '+out.name,flush=True)
    print('Execution complete; independent verification still required.',flush=True)
if __name__=='__main__':main()
