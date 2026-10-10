#!/usr/bin/env python3
"""Independent read-only audit of a case-bound native Block0 matrix."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--attempt',required=True,type=Path);p.add_argument('--bundle',required=True,type=Path);p.add_argument('--reference',required=True,type=Path);p.add_argument('--identity-audit',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    root=a.attempt.resolve();summary=sealed(root/'verification.json');host=sealed(a.bundle/'case-host.json');ref=sealed(a.reference/'reference.json');identities=sealed(a.identity_audit)
    assert summary['result']=='pass' and identities['result']=='pass' and identities['reference_root']==host['reference_root']==ref['receipt_root_sha256']==summary['reference_root']
    assert summary['case_host_root']==host['receipt_root_sha256'] and ref['fresh_prefix_matrix_proven'] and ref['runtime_input_use_forbidden']
    assert sha(a.bundle/'probe')==host['probe_sha256'] and sha(a.bundle/'pre.metallib')==host['metallib_sha256'] and not host['shader_changes']
    assert sha(a.bundle/'source/nr_pre/pre_probe.mm')==host['source_sha256'] and sha(a.bundle/'source/source-manifest.json')==host['source_manifest_sha256']
    assert sha(Path(host['image']))==ref['input_sha256'] and sha(Path(host['comparison_pool']))==ref['comparison_pool_sha256']
    commands=json.loads((root/'commands.json').read_text());assert [c['label'] for c in commands]==['a5-a','a5-b','5a-a','5a-b'] and all(c['exit_code']==0 for c in commands)
    pids=set();runs=[];count=0
    for command,row in zip(commands,summary['runs']):
        label=command['label'];directory=root/label;assert directory.resolve().is_relative_to(root);r=json.loads((directory/'run.json').read_text());c=command['command']
        assert len(c)==9 and Path(c[0]).resolve()==(a.bundle/'probe').resolve() and Path(c[1]).resolve()==(a.bundle/'pre.metallib').resolve()
        assert Path(c[2]).resolve()==Path(host['image']).resolve() and Path(c[4]).resolve()==Path(host['comparison_pool']).resolve() and Path(c[6]).resolve()==directory.resolve() and c[5]==label[:2]
        assert r['result']=='pass' and r['pid']==row['pid'] and r['pid'] not in pids and r['poison']==int(label[:2],16);pids.add(r['pid'])
        assert sha(directory/'run.json')==row['receipt_sha256'] and r['input_sha256']==ref['input_sha256'] and r['expected_gpu_input'] is False
        assert r['coverage_errors']==r['canary_errors']==r['immutable_errors']==r['skip_mismatching_bytes']==r['encoder_mismatching_bytes']==0
        for name,n in [('skip',70778880),('encoder',17694720)]:
            path=directory/(name+'.e4m3');assert path.stat().st_size==n and sha(path)==ref['ports'][name]==r[name+'_sha256']==row[name+'_sha256']
            coverage=np.memmap(directory/(name+'.coverage.bin'),dtype='<u4',mode='r');assert coverage.size==n and np.all(coverage==1);count+=n
        runs.append({'label':label,'pid':r['pid'],'receipt_sha256':sha(directory/'run.json')})
    assert len(pids)==4
    result={'schema':'neuralstudio.nr.independent-color-block0-audit.v1','case':ref['case'],'native_matrix_root':summary['receipt_root_sha256'],'rtx_prefix_root':ref['receipt_root_sha256'],'rtx_identity_audit_root':identities['receipt_root_sha256'],'runs':runs,'checked_output_bytes':count,'result':'pass','full_new_color_frame_parity':False,'scope':'Fresh dense-color image to native Block0, both outputs byte-matched to four-process RTX prefix, raw coverage and same-attempt executable/input/comparison provenance. Not later encoder/body acceptance.'}
    result['receipt_root_sha256']=canonical(result);a.output.write_text(json.dumps(result,indent=2)+'\n');print(result['receipt_root_sha256'])
if __name__=='__main__':main()
