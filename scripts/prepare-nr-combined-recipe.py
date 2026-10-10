#!/usr/bin/env python3
"""Emit Fable-schema combined-run recipe from verified local evidence; never run GPU."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p,expected):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h==expected;return r


def main():
    p=argparse.ArgumentParser();p.add_argument('--peer',type=Path,required=True);p.add_argument('--recipe',type=Path,required=True);p.add_argument('--receipt',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];peer=a.peer.resolve()
    packet=sealed(repo/'build/nr_combined_sequence/packet-audit-v1.json','8fc7129e7dfa5b2457707796319bbc57c0706cf3ddb32f99fe7ae2a544fa4a06')
    data=Path(packet['payload_directory'])
    for n,m in packet['files'].items():assert (data/n).stat().st_size==m['bytes'] and sha(data/n)==m['sha256']
    enc=repo/'build/nr_encoder_temporal/half-fma-spatial-v1'
    eb=sealed(enc/'build.json','9289bb7b21bd36571fffbdf057757fc6397baea3d9377c813965558e0430bfec')
    for n,h in eb['assets'].items():assert sha(enc/'assets'/n)==h
    obj='33f92090b058c509d822ee749588c1c694ceac0f2a44b2999fa570fdc9a704e8'
    build=peer/'build/nr_full/v20t';descriptor=peer/'build/nr_body/desc-v24t/chain.json'
    assert sha(enc/'nr_encoder_runtime.o')==sha(build/'source/encoder/nr_encoder_runtime.o')==obj
    exe='2abf182271d74a0c7383cf145304035cb77d993ca2a5ff149541e285a816c1a0'
    desc='55e80a6c383a89bb95ff99d2750b359d40f5a7d05177e7d776c7b743ec947baf'
    assert sha(build/'nr_full_frame')==exe and sha(descriptor)==desc
    assert sha(build/'source/src/nr_body/nr_full_frame.mm')=='ade17848864e8fbf10c3240eb79423b39c784f45b8f262621816b2c3e4840338'
    assert (build/'build-defines.txt').read_text().strip()=='defines: none'
    frames=[]
    for i,prefix in enumerate(('','frame-1-','frame-2-')):
        inp=data/(prefix+'input-color.rgba16le');ref=data/(prefix+'output.rgba16le')
        assert inp.stat().st_size==ref.stat().st_size==16588800
        assert sha(inp)==packet['input_hashes'][i] and sha(ref)==packet['outputs'][i]
        frames.append({'input':str(inp),'input_bytes':16588800,'input_sha256':sha(inp),'reference':str(ref),'reference_bytes':16588800,'reference_sha256':sha(ref)})
    recipe={'schema':'neuralstudio.nr.run-recipe.v1','name':'combined3-v20t-v1','harness_build':'build/nr_full/v20t',
            'harness_executable_sha256':exe,'descriptor':'build/nr_body/desc-v24t/chain.json','descriptor_sha256':desc,
            'encoder_assets':str(enc/'assets'),'encoder_object':str(enc/'nr_encoder_runtime.o'),'encoder_object_sha256':obj,
            'frames':frames,'motion':{'file':str(data/'input-motion.rg16le'),'bytes':8294400,'sha256':packet['motion_sha256'],'nonzero_admitted':True},
            'reset_sequence':'1,0,0','encoder_sampler_variant':3,
            'replicas':[{'tag':n,'poisons':p} for n,p in [('pilot',['a5','5a','a5']),('replica-2',['a5']*3),('replica-3',['5a']*3),('replica-4',['5a','a5','5a'])]],
            'storage_floor_gib':12,'failure_policy':'stop-config-on-any-failure',
            'reference_packet_root':packet['receipt_root_sha256'],'RTX_matrix_root':packet['RTX_matrix_root'],
            'encoder_build_root':eb['receipt_root_sha256'],'execution_owner':'Fable','independent_audit_owner':'Codex',
            'gpu_reservation_required':True,'comparison_references_required':True,
            'scope':'Three live frames: original/checker/black plus the admitted fixed signed-spatial field. Expected outputs are CPU comparison-only; no archived intermediates.',
            'notes':['Relative build/descriptor paths resolve in the Fable worktree, not this recipe directory.','Use explicit nr-gpu-lock reservation; current runner does not acquire one internally.','This pins the already validated v20t/v8 harness. A new reporting-schema build needs its own reviewed, versioned recipe.']}
    source=peer/'scripts/nr-run-recipe.py';source_pin=sha(source)
    spec=importlib.util.spec_from_file_location('peer_recipe',source);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    errors=m.preflight(recipe);assert not errors,errors;assert sha(source)==source_pin
    assert not a.recipe.exists() and not a.receipt.exists()
    a.recipe.parent.mkdir(parents=True,exist_ok=True);a.recipe.write_text(json.dumps(recipe,indent=2)+'\n')
    r={'schema':'neuralstudio.nr.combined-recipe-preparation.v1','result':'pass_preflight_only','recipe':str(a.recipe.resolve()),'recipe_sha256':sha(a.recipe),
       'packet_root':packet['receipt_root_sha256'],'encoder_root':eb['receipt_root_sha256'],'peer_runner_sha256':source_pin,
       'all_declared_files_rehashed':True,'peer_preflight_errors':errors,'GPU_executed':False,'execution_started':False,
       'limitations':'Preflight pass is not inference or independent audit acceptance. Explicit coordinated GPU reservation remains required.',
       'generator_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);a.receipt.parent.mkdir(parents=True,exist_ok=True);a.receipt.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
