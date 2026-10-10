#!/usr/bin/env python3
"""Verify all six native stages through attention scores and the RSQ domain."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();repo=Path(__file__).resolve().parents[1]
    require(not a.output.exists(),'preserve existing receipt')
    rtx=repo/'build/nr_rsq_half/attempt-v2';mac=repo/'build/nr_rsq_half/macos-v1'
    rsq_job=json.loads((rtx/'receipt.json').read_text(encoding='utf-8-sig'))
    require(rsq_job['source_sha256']==sha(repo/'src/nr_swin/rsq_half_oracle.cu'),'RSQ source drift')
    for entry in rsq_job['artifacts']:
        require(sha(rtx/'output'/entry['name'])==entry['sha256'],'RSQ transfer drift')
    rtx_pids=set();mac_pids=set();rsq_hashes=set();rsq_rows=[]
    for poison in ('a5','5a'):
        for ordinal in ('a','b'):
            stem=f'{poison}-{ordinal}'
            rr=json.loads((rtx/'output'/(stem+'.json')).read_text())
            mr=json.loads((mac/stem/'run.json').read_text())
            require(rr['pid'] not in rtx_pids and mr['pid'] not in mac_pids,'RSQ process reuse')
            rtx_pids.add(rr['pid']);mac_pids.add(mr['pid'])
            for r in (rr,mr):
                require(r['result']=='pass' and r['errors']==0 and r['count']==31743 and r['poison']==int(poison,16),'RSQ run contract')
                require(r['expected_input_opened'] is False,'RSQ expected input ingress')
            rb=(rtx/'output'/(stem+'.bin')).read_bytes();mb=(mac/stem/'output.bin').read_bytes()
            require(len(rb)==63486 and rb==mb,'RSQ mismatch')
            mathematical=b''.join(struct.pack('<e',1/math.sqrt(struct.unpack('<e',struct.pack('<H',i))[0])) for i in range(1,0x7c00))
            require(rb==mathematical,'RSQ correctly rounded half reference mismatch')
            rsq_hashes.add(hashlib.sha256(rb).hexdigest())
            rsq_rows.append({'rtx_pid':rr['pid'],'metal_pid':mr['pid'],'poison':int(poison,16)})
    require(len(rsq_hashes)==1,'RSQ poison dependence')
    pids=set();runs=[];compared=0
    names=['expansion.f16','activation.e4m3','partial.f16','qkv.f16','normalized.e4m3','scores.f16']
    for cta in range(9):
        fixture=repo/f'build/nr_swin_mlp/fixtures/cta{cta}'
        fixture_receipt=json.loads((fixture/'provenance.json').read_text())
        for name,digest in fixture_receipt['files'].items():
            require(sha(fixture/name)==digest,'MLP fixture drift')
        refs={name:fixture/'comparison'/name for name in names[:3]}
        refs['qkv.f16']=repo/f'build/nr_swin_qkv/reference-v1/cta{cta}/qkv.f16'
        for name in names[4:]:refs[name]=repo/f'build/nr_swin_attention/reference-v1/cta{cta}'/name
        expected={name:sha(path) for name,path in refs.items()}
        qref=json.loads((repo/f'build/nr_swin_qkv/reference-v1/cta{cta}/receipt.json').read_text())
        aref=json.loads((repo/f'build/nr_swin_attention/reference-v1/cta{cta}/receipt.json').read_text())
        require(expected['qkv.f16']==qref['qkv_sha256'],'QKV reference drift')
        require(expected['normalized.e4m3']==aref['normalized_sha256'] and expected['scores.f16']==aref['scores_sha256'],'attention reference drift')
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                run=a.attempt/f'cta{cta}-{poison}-{ordinal}'
                r=json.loads((run/'run.json').read_text())
                require(r['pid'] not in pids,'process reuse');pids.add(r['pid'])
                require(r['result']=='pass' and r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'native guard failure')
                require(r['rows']==64 and r['poison']==int(poison,16) and r['dispatch_count']==6,'native shape/poison/dispatch')
                require(r['expected_file_opened'] is False,'native expected ingress')
                require(r['input_sha256']==sha(fixture/'inputs/input.e4m3') and r['weight_sha256']==sha(fixture/'inputs/weight.bin'),'native input identity')
                require(r['outputs']==expected,'attention intermediate mismatch')
                for name,path in refs.items():
                    actual=(run/name).read_bytes();target=path.read_bytes()
                    require(actual==target,'direct comparison mismatch: '+name);compared+=len(actual)
                runs.append({'cta':cta,'pid':r['pid'],'poison':r['poison'],'outputs':expected,
                             'run_receipt_sha256':sha(run/'run.json')})
    sources=['src/nr_swin/qmma_fixed.metalh','src/nr_swin/swin32_mlp.metal','src/nr_swin/swin32_qkv.metal',
             'src/nr_swin/swin32_attention.metal','src/nr_swin/rsq_half.metal',
             'src/nr_swin/rsq_half_oracle.cu','src/nr_swin/rsq_half_probe.mm',
             'src/nr_swin/mlp_probe.mm','scripts/analyze-nr-swin-attention.py',
             'scripts/generate-nr-swin-attention-probe.py','scripts/generate-nr-swin-qkv-probe.py',
             'scripts/verify-nr-swin-attention.py','scripts/run-nr-rsq-half.ps1']
    result={'schema':'neuralstudio.nr.swin32-attention-scores-verification.v1',
            'parent_mlp_root_sha256':'ce3acdba4bdd4df5ae2cbded0331d9957e22c788661945f15a4abff6cc98c37e',
            'parent_qkv_root_sha256':'af07137093069e314668e959b39062def17bc7617e35d0f51be572f1cdc714e9',
            'fresh_metal_processes':len(pids),'poisons':[90,165],'compared_bytes':compared,
            'mismatched_bytes':0,'runs':runs,
            'rsq_domain':{'positive_finite_fp16_inputs':31743,'rtx_processes':4,'metal_processes':4,
                          'rtx_gpu_driver':rsq_job['gpu'],'output_sha256':next(iter(rsq_hashes)),
                          'mathematically_rounded_half_equal':True,
                          'runs':rsq_rows,'mismatched_bytes':0,'job_receipt_sha256':sha(rtx/'receipt.json')},
            'source_sha256':{name:sha(repo/name) for name in sources},
            'host_sha256':sha(a.attempt/'probe'),'metallib_sha256':sha(a.attempt/'attention.metallib'),
            'claims':{'mlp_qkv_normalization_scores_live_device_chain':True,
                      'isolated_recorded_input_diagnostic':True,'softmax_value_projection':False,
                      'complete_block1':False,'full_model_inference':False},'result':'pass'}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256'])
    print(f'compared_bytes={compared} mismatched_bytes=0 rsq_inputs=31743')


if __name__=='__main__':main()
