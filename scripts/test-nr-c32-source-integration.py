#!/usr/bin/env python3
"""Check tracked source against accepted expressions and replay the bounded witness."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def function(text,name):
    marker='inline half '+name+'(';assert text.count(marker)==1
    start=text.index('{',text.index(marker));depth=1;end=start+1
    while depth:
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    return re.sub(r'\s+','',re.sub(r'//[^\n]*','',text[start:end]))


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];old=repo/'build/nr_encoder_fscale/bundle-v1/encoder/source'
    candidate=repo/'build/nr_sixteen/half-fma-candidate-v1/source'
    current=repo/'src/nr_swin';att=(current/'swin32_attention.metal').read_text();wide=(current/'swin_wide.metal').read_text()
    assert function(att,'nr_norm32')==function((candidate/'swin32_attention.metal').read_text(),'nr_norm32'),'accepted single-half expression changed'
    assert function(att,'nr_norm32_float32_counterfactual')==function((old/'swin32_attention.metal').read_text(),'nr_norm32'),'old counterfactual changed'
    anchor='kernel void nr_swin32_normalize'
    assert att[att.index(anchor):]==(candidate/'swin32_attention.metal').read_text().split(anchor,1)[1].join([anchor,''])
    expected_wide=(old/'swin_wide.metal').read_text();assert expected_wide.count('return nr_norm32(value);')==1
    assert wide==expected_wide.replace('return nr_norm32(value);','return nr_norm32_float32_counterfactual(value);'),'wide default arithmetic changed'
    assert not a.output.exists();a.output.mkdir(parents=True);src=a.output/'source';shutil.copytree(candidate,src)
    for name in ('swin32_attention.metal','swin_wide.metal'):shutil.copyfile(current/name,src/name)
    neg=a.output/'negative-source';shutil.copytree(src,neg)
    text=(neg/'swin32_attention.metal').read_text();target='float(nr_norm32(values))';assert text.count(target)==1
    (neg/'swin32_attention.metal').write_text(text.replace(target,'float(nr_norm32_float32_counterfactual(values))'))
    metal=subprocess.check_output(['xcrun','-f','metal'],text=True).strip();metallib=subprocess.check_output(['xcrun','-f','metallib'],text=True).strip();sdk=subprocess.check_output(['xcrun','--show-sdk-path'],text=True).strip()
    commands=[]
    with (a.output/'compile.log').open('x') as log:
        for label,source in [('default',src),('counterfactual',neg)]:
            air=a.output/(label+'.air');lib=a.output/(label+'.metallib')
            for cmd in [[metal,'-isysroot',sdk,'-std=metal3.1','-fno-fast-math','-ffp-contract=off','-c',str(source/'prefix8.metal'),'-o',str(air)],[metallib,str(air),'-o',str(lib)]]:
                commands.append(cmd);subprocess.run(cmd,check=True,stdout=log,stderr=subprocess.STDOUT)
    host=repo/'build/nr_sixteen/window-probe-v1/probe';inputs=repo/'build/nr_sixteen/early-comparison-v1/input.e4m3';weight=repo/'build/nr_encoder_temporal/half-fma-v1/assets/weights/block4.bin'
    assert sha(host)=='9ff9ad10b7c6a0ca6a92cf5f4350c2c7f8b03762daa8bdbadaffa014c48ebef5'
    assert sha(inputs)=='523e57ef59591bfcd471aa9f4ca83e2f41dc0d70fc56076aebc20b2146b36a5e'
    assert sha(weight)=='f0d05533694577fa313f2655b00ada84f231f87ebc66ebbb7e8f0f7ea1ac88d2'
    rows=[];pids=set()
    for label,lib,poison,expected in [('a5-a','default','a5','half-fma-window-v1'),('a5-b','default','a5','half-fma-window-v1'),('5a-a','default','5a','half-fma-window-v1'),('5a-b','default','5a','half-fma-window-v1'),('negative-a5','counterfactual','a5','window-replay-v1')]:
        output=a.output/label;cmd=[str(host),str(a.output/(lib+'.metallib')),str(inputs),str(weight),poison,str(output)]
        with (a.output/(label+'.stdout')).open('x') as stdout,(a.output/(label+'.stderr')).open('x') as stderr:run=subprocess.run(cmd,stdout=stdout,stderr=stderr)
        assert run.returncode==0,'diagnostic execution failed; preserve output'
        r=json.loads((output/'run.json').read_text());assert r['pid'] not in pids;pids.add(r['pid'])
        assert r['coverage_errors']==r['canary_errors']==r['immutable_errors']==0 and r['archived_component_input'] and not r['expected_file_opened']
        baseline=repo/'build/nr_sixteen'/expected/'a5-a'
        checked=[]
        for file in baseline.iterdir():
            if file.suffix not in ('.f16','.e4m3'):continue
            assert (output/file.name).read_bytes()==file.read_bytes(),'stored stage differs: '+file.name
            checked.append(file.name)
        assert len(checked)==11
        rows.append({'label':label,'pid':r['pid'],'command':cmd,'stored_stages_equal_to':str(baseline),'stored_stage_count':len(checked),'output_sha256':sha(output/'output.e4m3')})
    assert rows[0]['output_sha256']=='1ba222a9cf62ae7eb41367bfee9323852a414dc4b633ecad294160099c39d165'
    assert rows[-1]['output_sha256']=='85dbff354fb9ab2d08ffeb3933b7e9b8919983060cc6708a14d69c02be2c987c'
    r={'schema':'neuralstudio.nr.c32-source-integration.v1','result':'pass_tracked_source_and_bounded_replay','runs':rows,
       'source_expression_equals_accepted_candidate':True,'wide_default_source_unchanged':True,'counterfactual_preserved':True,
       'full_model_execution':False,'archived_component_diagnostic_only':True,'commands':commands,
       'tracked_source_hashes':{n:sha(current/n) for n in ('swin32_attention.metal','swin_wide.metal')},
       'files':{str(p.relative_to(a.output)):sha(p) for p in a.output.rglob('*') if p.is_file()},'verifier_sha256':sha(Path(__file__))}
    r['receipt_root_sha256']=canonical(r);(a.output/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps({k:v for k,v in r.items() if k not in ('files','commands','runs')},indent=2))


if __name__=='__main__':main()
