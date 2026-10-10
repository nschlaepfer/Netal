#!/usr/bin/env python3
"""Strict component checkpoint audit: no success when any old checkpoint is absent."""
import argparse
import hashlib
import json
import math
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text())
def require_all_checkpoints(expected,actual):
    missing=sorted(expected.keys()-actual.keys())
    different=sorted(k for k in expected.keys()&actual.keys() if expected[k]!=actual[k])
    assert not missing and not different,(missing,different)
    return missing,different


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();peer=a.peer.resolve()
    oldp=peer/'build/nr_body/v12-checked-compare-v1/v11-checked/chain-run.json';old=read(oldp)
    expected={(x['launch'],k):v for x in old['launches'] if x['kind']!='ptx' for k,v in x['outputs'].items()}
    assert len(expected)==125 and old['poison']==90
    for x in old['launches']:
        if x['kind']=='ptx':expected[x['launch'],'pool']=x['pool_post_sha256']
    assert len(expected)==149
    pins={str(oldp):sha(oldp)};cases=[]
    for label in ('checked-no-fault-faultbuild','checked-no-fault-prodbuild'):
        d=peer/'build/nr_body/v13-evidence-v1/cli'/label;p=d/'chain-run.json';r=read(p);manifest=read(d/'manifest.json')
        assert manifest['code']==0 and r['command_buffers']==1 and r['poison']==90
        assert r['decoder_pre_image_source']=='gpu-snapshot' and r['decoder_pre_image_sha256']==r['decoder_pre_image_expected_sha256']
        actual={(x['launch'],k[:-7]):v for x in r['launches'] for k,v in x.get('digests',{}).items() if k.endswith('_sha256')}
        missing,different=require_all_checkpoints(expected,actual)
        assert sum(k[1]=='pool' for k in actual)==24
        assert r['decoder_pre_image_sha256']==old['decoder']['pool_pre_sha256'] and r['pool_final_sha256']==old['decoder']['pool_final_sha256']
        tex155=next(x for x in old['launches'] if x['launch']==155)['texture']['output_sha256'];assert r['tex155_output_sha256']==tex155
        assert sha(d/'tex_out_155.bin')==sha(d/'tex_out_156.bin')==r['final_surface_sha256']==tex155=='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2'
        assert r['validators']=={'expected':150,'run':150,'faults':0}
        assert math.isfinite(r['frame_gpu_ms']) and r['frame_gpu_ms']>0
        for name in ('chain-run.json','manifest.json','tex_out_155.bin','tex_out_156.bin'):pins[str(d/name)]=sha(d/name)
        cases.append({'case':label,'pid':r['pid'],'strong_identities_compared':152,'middle_identities':125,'decoder_pool_identities':24,'missing':missing,'different':different,'actual_GPU_preimage_source':r['decoder_pre_image_source']})
    fault=peer/'build/nr_body/v13-evidence-v1/cli/fault-restage-corrupt';fr=read(fault/'chain-run.json');fm=read(fault/'manifest.json')
    assert fm['code']==3 and fr['decoder_pre_image_source']=='gpu-snapshot' and fr['decoder_pre_image_sha256']!=fr['decoder_pre_image_expected_sha256']==old['decoder']['pool_pre_sha256']
    assert sha(fault/'tex_out_156.bin')=='a07835db4be8469dad1e3d3befcc495be396790f89408be62ddb8a5a710585a2'
    for name in ('chain-run.json','manifest.json','tex_out_156.bin'):pins[str(fault/name)]=sha(fault/name)
    pp=peer/'build/nr_body/v13-evidence-v1/probe-results.json';probe=read(pp);pins[str(pp)]=sha(pp);assert probe['all_ok'] and all(x['ok'] for x in probe['cases'])
    wanted={'first-failure-7-and-16777223-count2':(7,2),'first-failure-only-16777223':(16777223,2),'first-failure-final-pool-word-missing':(53129983,0),'at-most-once-final-pool-word-duplicate':(53129983,2)}
    for name,(index,count) in wanted.items():
        q=next(x for x in probe['cases'] if x['case']==name);assert q['faulted'] and not q['contract_pass'] and q['first_element']==index and q['first_observed']==count
    source=peer/'build/nr_body/runtime-v13/source/src/nr_body/nr_body_runtime_v13.mm';s=source.read_text();pins[str(source)]=sha(source)
    assert 'res.decoder_pre_image_sha256 = actual; res.decoder_pre_image_source = "gpu-snapshot"' in s
    assert 'digest("decoder pre-image (restaged pool)", 133u, pool, 0, pool_bytes, nullptr, true, true)' in s
    assert 'digest("launch " + std::to_string(n) + " pool", n, pool, 0, pool_bytes, nullptr, true, true)' in s
    for path,h in pins.items():assert sha(Path(path))==h
    result={'schema':'neuralstudio.nr.v13-checkpoint-independent-audit.v1','result':'pass_bounded_archived_input_component_checkpoints','cases':cases,'actual_restaging_corruption_detected_by_SHA256':True,'final_frame_alone_would_not_detect_this_fault':True,
            'full_range_failure_identity_cases':wanted,'pins':pins,'full_live_model_acceptance':False,'resident_speedup_accepted':False,
            'limits':['Two fresh same-poison archived-encoder-input component runs, not opposite-poison or complete live-model validation.','152 identities per run compare reported SHA-256 from v13 in-memory GPU snapshots with the accepted v11 checked path. Final texture bytes are also independently rehashed.','Production-mode omissions and the pending additive API remain separate gates.'],'verifier_sha256':sha(Path(__file__))}
    result['receipt_root_sha256']=canonical(result);assert not a.output.exists();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='pins'},indent=2))


if __name__=='__main__':main()
