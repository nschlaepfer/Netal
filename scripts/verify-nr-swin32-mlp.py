#!/usr/bin/env python3
"""Verify compact NR MLP intermediates against isolated RTX block fixtures."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def require(test,message):
    if not test:raise ValueError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--fixtures',type=Path,required=True)
    p.add_argument('--attempt',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    require(not a.output.exists(),'preserve existing receipt')
    pids=set();compared=0;ctas=[]
    files={'expansion.f16':16384,'activation.e4m3':8192,'partial.f16':16384}
    for cta in range(9):
        fixture=a.fixtures/f'cta{cta}'
        provenance=json.loads((fixture/'provenance.json').read_text())
        for name,digest in provenance['files'].items():
            require(sha(fixture/name)==digest,'fixture mutation: '+name)
        hashes={name:sha(fixture/'comparison'/name) for name in files}
        run_rows=[]
        for poison in ('a5','5a'):
            for ordinal in ('a','b'):
                run=a.attempt/f'cta{cta}-{poison}-{ordinal}'
                r=json.loads((run/'run.json').read_text())
                require(r['result']=='pass' and r['rows']==64 and r['poison']==int(poison,16),'run contract')
                require(r['pid'] not in pids,'process ID reused');pids.add(r['pid'])
                require(r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0,'coverage or mutation failure')
                require(r['expected_file_opened'] is False and r['dispatch_count']==3,'execution ingress/count')
                require(r['input_sha256']==sha(fixture/'inputs/input.e4m3'),'input mismatch')
                require(r['weight_sha256']==provenance['weight_sha256'],'weight mismatch')
                require(r['outputs']==hashes,'RTX/Mac intermediate mismatch')
                for name,size in files.items():
                    actual=(run/name).read_bytes();expected=(fixture/'comparison'/name).read_bytes()
                    require(len(actual)==len(expected)==size and actual==expected,'direct byte comparison failed: '+name)
                    compared+=size
                run_rows.append({'pid':r['pid'],'poison':r['poison'],'receipt_sha256':sha(run/'run.json')})
        ctas.append({'cta':provenance['cta'],'outputs':hashes,'runs':run_rows,
                     'fixture_provenance_sha256':sha(fixture/'provenance.json')})
    repo=Path(__file__).resolve().parents[1]
    sources=['src/nr_swin/qmma_fixed.metalh','src/nr_swin/swin32_mlp.metal','src/nr_swin/mlp_probe.mm',
             'scripts/analyze-nr-swin-mlp.py','scripts/test-nr-swin32-mlp.sh',
             'scripts/verify-nr-swin32-mlp.py']
    result={'schema':'neuralstudio.nr.swin32-mlp-verification.v1','ctas':ctas,
            'fresh_metal_process_count':len(pids),'poisons':[90,165],
            'compared_bytes':compared,'mismatched_bytes':0,
            'selected_qmma_operations_per_cta':128,
            'metal_dispatches_per_cta':3,'write_coverage_exactly_once':True,
            'inputs_immutable':True,'canaries_intact':True,
            'source_sha256':{name:sha(repo/name) for name in sources},
            'host_binary_sha256':sha(a.attempt/'build/probe'),
            'metallib_sha256':sha(a.attempt/'build/mlp.metallib'),
            'claims':{'isolated_recorded_input_diagnostic':True,
                      'compact_block1_first_mlp_exact_at_nine_ctas':True,
                      'same_process_live_block0_input':False,
                      'all_small_swin_layers_verified':False,
                      'complete_block1':False,'full_model_inference':False,
                      'fresh_rtx_intermediate_capture':False},
            'result':'pass'}
    result['receipt_root_sha256']=hashlib.sha256(json.dumps(result,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    print('receipt_root_sha256='+result['receipt_root_sha256'])
    print(f'fresh_processes={len(pids)} compared_bytes={compared} mismatched_bytes=0')


if __name__=='__main__':main()
