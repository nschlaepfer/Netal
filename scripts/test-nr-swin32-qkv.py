#!/usr/bin/env python3
"""Build and verify the connected MLP-to-QKV Metal diagnostic."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--fixtures',type=Path,default=Path('build/nr_swin_mlp/fixtures'))
    p.add_argument('--reference',type=Path,default=Path('build/nr_swin_qkv/reference-v1'))
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    require(not a.output.exists(),'preserve existing attempt')
    build=a.output/'build';build.mkdir(parents=True)
    def call(command,name):
        with (a.output/(name+'.log')).open('w') as log:
            subprocess.run([str(x) for x in command],stdout=log,stderr=subprocess.STDOUT,check=True)
    call(['python3',repo/'scripts/generate-nr-swin-qkv-probe.py','--output',build/'probe.mm'],'generate')
    toolchain=Path('/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v17.6.109.0.ez7LwL/Metal.xctoolchain/usr/bin')
    sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    call([toolchain/'metal','-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off',
          '-c',repo/'src/nr_swin/swin32_qkv.metal','-o',build/'qkv.air'],'metal')
    call([toolchain/'metallib',build/'qkv.air','-o',build/'qkv.metallib'],'metallib')
    call(['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra',
          '-Werror','-Wno-deprecated-declarations',build/'probe.mm','-framework','Foundation',
          '-framework','Metal','-o',build/'probe'],'host')
    reference=json.loads((a.reference/'receipt.json').read_text())
    require(reference['result']=='pass','reference rejected')
    pids=set();compared=0;rows=[]
    for cta in range(9):
        fixture=a.fixtures/f'cta{cta}';expect=a.reference/f'cta{cta}'
        refs={name:(fixture/'comparison'/name).read_bytes() for name in ('expansion.f16','activation.e4m3','partial.f16')}
        refs['qkv.f16']=(expect/'qkv.f16').read_bytes()
        expected={name:hashlib.sha256(data).hexdigest() for name,data in refs.items()}
        require(expected['qkv.f16']==reference['ctas'][cta]['qkv_sha256'],'QKV reference drift')
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                name=f'cta{cta}-{poison}-{ordinal}';run=a.output/name
                call([build/'probe',build/'qkv.metallib',fixture/'inputs/input.e4m3',
                      fixture/'inputs/weight.bin','64',poison,run],name)
                r=json.loads((run/'run.json').read_text())
                require(r['pid'] not in pids,'PID reused');pids.add(r['pid'])
                require(r['result']=='pass' and r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'native guards')
                require(r['dispatch_count']==4 and r['expected_file_opened'] is False,'native ingress/dispatch')
                require(r['poison']==int(poison,16) and r['rows']==64,'native shape/poison')
                require(r['input_sha256']==sha(fixture/'inputs/input.e4m3') and r['weight_sha256']==reference['ctas'][cta]['original_weight_sha256'],'native inputs')
                require(r['outputs']==expected,'intermediate output mismatch')
                for name,data in refs.items():
                    require((run/name).read_bytes()==data,'direct byte mismatch: '+name)
                    compared+=len(data)
                rows.append({'cta':cta,'pid':r['pid'],'poison':r['poison'],
                             'receipt_sha256':sha(run/'run.json'),'outputs':expected})
    sources=['src/nr_swin/qmma_fixed.metalh','src/nr_swin/swin32_mlp.metal','src/nr_swin/swin32_qkv.metal','src/nr_swin/mlp_probe.mm',
             'scripts/generate-nr-swin-qkv-probe.py','scripts/nr_swin_reference.py',
             'scripts/build-nr-swin-qkv-reference.py','scripts/test-nr-swin32-qkv.py']
    receipt={'schema':'neuralstudio.nr.swin32-mlp-qkv-verification.v1',
             'parent_mlp_root_sha256':'ce3acdba4bdd4df5ae2cbded0331d9957e22c788661945f15a4abff6cc98c37e',
             'reference_root_sha256':reference['receipt_root_sha256'],
             'source_sha256':{name:sha(repo/name) for name in sources},
             'generated_host_sha256':sha(build/'probe.mm'),'host_sha256':sha(build/'probe'),
             'metallib_sha256':sha(build/'qkv.metallib'),
             'fresh_processes':len(pids),'poisons':[90,165],'runs':rows,
             'compared_bytes':compared,'mismatched_bytes':0,
             'write_coverage_exactly_once':True,'inputs_immutable':True,'canaries_intact':True,
             'claims':{'native_mlp_to_qkv_live_buffers':True,'isolated_recorded_input_diagnostic':True,
                       'attention_computed':False,'complete_block1':False,'full_model_inference':False,
                       'fresh_rtx_intermediate_capture':False},'result':'pass'}
    receipt['receipt_root_sha256']=hashlib.sha256(json.dumps(receipt,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    (a.output/'verification.json').write_text(json.dumps(receipt,sort_keys=True,indent=2)+'\n')
    print('receipt_root_sha256='+receipt['receipt_root_sha256'])
    print(f'processes={len(pids)} compared_bytes={compared} mismatched_bytes=0')


if __name__=='__main__':main()
