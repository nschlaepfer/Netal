#!/usr/bin/env python3
"""Fresh image -> case-bound Block0 -> layout -> encoder22 boundary checks.

Invoke under the shared Mac lock. Expected buffers are CPU comparisons only.
"""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from nr_candidate_primitive import validate_primitive

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def layout_indices():
    gid=np.arange(17694720,dtype=np.uint32);row=gid//32;c=gid%32;window=row//64;token=row%64;tile=token//16;q=token%16
    x=(window%120)*8+(tile//2)*4+q%4;y=(window//120)*8+(tile%2)*4+q//4
    index=((c//16)*960*576+y*960+x)*16+c%16
    assert np.all(np.bincount(index,minlength=17694720)==1)
    return index
def main():
    p=argparse.ArgumentParser();p.add_argument('--pre-bundle',required=True,type=Path);p.add_argument('--encoder-bundle',required=True,type=Path);p.add_argument('--block0-reference',required=True,type=Path);p.add_argument('--encoder-reference',required=True,type=Path);p.add_argument('--layout-parent',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--minimum-free-gib',type=float,default=13);a=p.parse_args()
    if not 12<=a.minimum_free_gib<=1024:raise ValueError('minimum free space12..1024GiB required')
    repo=Path(__file__).resolve().parents[1];a.output=a.output.resolve();a.output.mkdir(parents=True,exist_ok=False)
    pre=sealed(a.pre_bundle/'case-host.json');enc=sealed(a.encoder_bundle/'manifest.json');b0=sealed(a.block0_reference/'reference.json');boundary=sealed(a.encoder_reference/'reference.json');retrieved=sealed(a.encoder_reference/'retrieval.json')
    assert b0.get('boundary','block0')=='block0' and boundary['boundary']=='encoder22' and b0['case']==boundary['case']==pre['case']
    assert b0['fresh_prefix_matrix_proven'] and boundary['fresh_prefix_matrix_proven'] and retrieved['reference_root']==boundary['receipt_root_sha256'] and retrieved['result']=='pass'
    assert pre['reference_root']==b0['receipt_root_sha256'] and pre['input_sha256']==b0['input_sha256']==boundary['input_sha256']
    assert pre['reference_gpu_input'] is False and pre['shader_changes'] is False
    assert pre['parent_bundle_root']==enc['receipt_root_sha256']
    assert sha(a.pre_bundle/'source/nr_pre/pre_probe.mm')==pre['source_sha256']
    assert sha(a.pre_bundle/'source/source-manifest.json')==pre['source_manifest_sha256']
    for name,h in json.loads((a.pre_bundle/'source/source-manifest.json').read_text())['files'].items():assert sha(a.pre_bundle/'source'/name)==h,name
    assert b0['ports']['skip']==boundary['ports']['block0-skip']
    assert sha(Path(pre['image']))==pre['input_sha256'] and sha(Path(pre['comparison_pool']))==b0['comparison_pool_sha256']
    if retrieved['schema']=='neuralstudio.nr.encoder-hash-reference-retrieval.v1':
        identity=sealed(a.encoder_reference/'identities/verification.json');assert identity['receipt_root_sha256']==retrieved['identity_audit_root'] and identity['result']=='pass' and identity['reference_root']==boundary['receipt_root_sha256']
    else:assert sha(a.encoder_reference/'comparison-pool.bin')==boundary['comparison_pool_sha256']
    assert sha(a.pre_bundle/'probe')==pre['probe_sha256'] and sha(a.pre_bundle/'pre.metallib')==pre['metallib_sha256']
    validate_primitive(a.encoder_bundle,enc)
    for name,h in enc['files'].items():assert sha(a.encoder_bundle/name)==h,name
    weights=repo/'build/nr_pre/rtx-feature-source-v1/weights.bin';math=repo/'build/nr_noise_math/corrections-v1'
    pinned={str(path.resolve()):sha(path) for path in [Path(pre['image']),Path(pre['comparison_pool']),weights,math/'instruction-deltas.u8',math/'instruction-escapes.u32',a.pre_bundle/'probe',a.pre_bundle/'pre.metallib',a.pre_bundle/'case-host.json',a.encoder_bundle/'manifest.json']+[a.encoder_bundle/name for name in enc['files']]}
    layout_plan=sealed(a.layout_parent/'plan.json')
    for name,h in layout_plan['source_files'].items():assert sha(a.layout_parent/name)==h,name
    for name,h in layout_plan['layout_artifacts'].items():assert sha(a.layout_parent/name)==h,name
    index=layout_indices();reference_input=np.memmap(Path(pre['comparison_pool']),mode='r',dtype=np.uint8,offset=71166976,shape=(17694720,))
    assert hashlib.sha256(reference_input).hexdigest()==b0['ports']['encoder'];layout_hash=hashlib.sha256(reference_input[index]).hexdigest();del reference_input
    source=a.output/'source';(source/'nr_pre').mkdir(parents=True);(source/'nr_swin').mkdir()
    for name in ('entry_layout_probe.mm','entry_layout.metal'):shutil.copyfile(a.layout_parent/'source/nr_pre'/name,source/'nr_pre'/name)
    shutil.copyfile(a.layout_parent/'source/nr_swin/mlp_probe.mm',source/'nr_swin/mlp_probe.mm');shutil.copyfile(a.layout_parent/'entry.metallib',a.output/'entry.metallib')
    host=source/'nr_pre/entry_layout_probe.mm';text=host.read_text()
    for old,new in [('de4789425897a4ffa3d5646670c12385d25109dcad60609c1729dca2da4a33a0',b0['ports']['encoder']),('176544cf02971c274f9f8d5de8a0436d5e71945ce4e333dc2ef169b48b10e554',layout_hash)]:
        assert text.count(old)==1;text=text.replace(old,new)
    host.write_text(text);commands=[]
    def run(label,command):
        assert shutil.disk_usage(a.output).free>=a.minimum_free_gib*1024**3,'storage floor: preserve partial run'
        command=[str(x) for x in command];start=datetime.now(timezone.utc).isoformat()
        with (a.output/(label+'.log')).open('x') as log:r=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        commands.append({'label':label,'command':command,'start_utc':start,'end_utc':datetime.now(timezone.utc).isoformat(),'exit_code':r.returncode});(a.output/'commands.json').write_text(json.dumps(commands,indent=2)+'\n')
        assert r.returncode==0,label+' failed; preserve evidence'
    run('compile-layout',['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',host,'-framework','Foundation','-framework','Metal','-o',a.output/'entry_probe'])
    plan={'schema':'neuralstudio.nr.color-encoder-plan.v1','case':b0['case'],'input_sha256':pre['input_sha256'],'pre_case_host_root':pre['receipt_root_sha256'],'encoder_bundle_root':enc['receipt_root_sha256'],'rtx_block0_root':b0['receipt_root_sha256'],'rtx_encoder_boundary_root':boundary['receipt_root_sha256'],
      'layout_expected_sha256':layout_hash,'layout_reference_role':'CPU hash of permutation of RTX Block0 data; no expected buffer staged into GPU','minimum_free_gib':a.minimum_free_gib,
      'layout_source_sha256':sha(host),'layout_probe_sha256':sha(a.output/'entry_probe'),'layout_metallib_sha256':sha(a.output/'entry.metallib'),'encoder_bundle':str(a.encoder_bundle.resolve()),'pre_bundle':str(a.pre_bundle.resolve()),'artifacts':pinned,'driver_sha256':sha(Path(__file__)),'rtx_boundary_reference':str(a.encoder_reference.resolve()),'rtx_block0_reference':str(a.block0_reference.resolve())}
    plan['receipt_root_sha256']=canonical(plan);(a.output/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    encoder=a.encoder_bundle/'encoder';pids=set();rows=[];failures=[]
    for label in ('a5-a','a5-b','5a-a','5a-b'):
        out=a.output/label;out.mkdir();poison=label[:2]
        run(label+'-block0',[a.pre_bundle/'probe',a.pre_bundle/'pre.metallib',pre['image'],weights,pre['comparison_pool'],poison,out/'block0',math/'instruction-deltas.u8',math/'instruction-escapes.u32'])
        run(label+'-layout',[a.output/'entry_probe',a.output/'entry.metallib',out/'block0',poison,out/'entry-layout'])
        run(label+'-encoder',[encoder/'probe',encoder/'prefix.metallib',out/'entry-layout/encoder-window.e4m3',encoder/'weights',poison,out/'encoder',22,encoder/'wide128.metallib',encoder/'wide256.metallib'])
        br=json.loads((out/'block0/run.json').read_text());lr=json.loads((out/'entry-layout/run.json').read_text());er=json.loads((out/'encoder/run.json').read_text())
        for child in (br,lr,er):assert child['pid'] not in pids and child['result']=='pass' and child['poison']==int(poison,16);pids.add(child['pid'])
        for child in (br,lr,er):assert child['coverage_errors']==child['canary_errors']==0
        assert br['immutable_errors']==er['immutable_errors']==0 and lr['input_immutable']
        assert br['input_sha256']==pre['input_sha256'] and br['expected_gpu_input'] is False and lr['expected_gpu_input'] is False and er['expected_intermediate_opened'] is False
        assert br['encoder_sha256']==b0['ports']['encoder']==lr['input_sha256'] and br['skip_sha256']==b0['ports']['skip']
        assert er['live_device_handoffs'] and [v['block'] for v in er['blocks']]==list(range(1,23))
        assert lr['producer_pid']==br['pid'] and lr['producer_receipt_sha256']==sha(out/'block0/run.json') and lr['output_sha256']==er['input_sha256']==layout_hash
        actual=np.memmap(out/'entry-layout/encoder-window.e4m3',mode='r',dtype=np.uint8);original=np.memmap(out/'block0/encoder.e4m3',mode='r',dtype=np.uint8);assert np.array_equal(actual,original[index])
        assert sha(out/'entry-layout/encoder-window.e4m3')==layout_hash and sha(out/'block0/encoder.e4m3')==b0['ports']['encoder']
        for path,count in [(out/'entry-layout/coverage.bin',17694720),(out/'block0/skip.coverage.bin',70778880),(out/'block0/encoder.coverage.bin',17694720)]:
            coverage=np.memmap(path,mode='r',dtype='<u4');assert coverage.size==count and np.all(coverage==1)
        ports={'block0-skip':out/'block0/skip.e4m3',**{f'block{b}-native':out/f'encoder/block{b}-native.e4m3' for b in (4,8,14,22)},'block22-middle':out/'encoder/block22-middle.e4m3'}
        hashes={name:sha(path) for name,path in ports.items()}
        exports=json.loads((out/'encoder/native_exports.json').read_text());assert exports['pid']==er['pid'] and exports['result']=='pass' and exports['live_device_inputs'] and exports['expected_input_opened'] is False
        assert exports['coverage_errors']==exports['canary_errors']==exports['immutable_errors']==0
        for name,path in ports.items():
            if name!='block0-skip':assert hashes[name]==exports['outputs'][path.name]['sha256'] and path.stat().st_size==exports['outputs'][path.name]['bytes']
        for name,h in hashes.items():
            if h!=boundary['ports'][name]:failures.append({'run':label,'port':name,'actual':h,'rtx':boundary['ports'][name]})
        rows.append({'run':label,'block0_pid':br['pid'],'layout_pid':lr['pid'],'encoder_pid':er['pid'],'ports':hashes,'receipt_sha256':{part:sha(out/part/'run.json') for part in ('block0','entry-layout','encoder')}})
        for path,h in pinned.items():assert sha(Path(path))==h,'material input drift: '+path
        assert sha(a.output/'entry_probe')==plan['layout_probe_sha256'] and sha(a.output/'entry.metallib')==plan['layout_metallib_sha256'] and sha(host)==plan['layout_source_sha256']
        print('completed '+label+'; boundary failures='+str(len(failures)),flush=True)
    result={'schema':'neuralstudio.nr.color-encoder-boundary.v1','plan_root':plan['receipt_root_sha256'],'case':b0['case'],'fresh_processes':len(pids),'runs':rows,'failures':failures,'result':'pass' if not failures else 'fail','full_new_color_frame_parity':False,'interior_encoder_rtx_parity':False,'scope':'Four fresh original-image chains through native Block0, bijective live layout and encoder22. Six boundary ports compared against fresh RTX after launch24. Interior outputs retained but not RTX-compared for this new case; body/final frame still pending.'}
    result['receipt_root_sha256']=canonical(result);(a.output/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
    if failures:raise SystemExit(3)
if __name__=='__main__':main()
