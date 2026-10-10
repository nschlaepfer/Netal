#!/usr/bin/env python3
"""Bind a case-specific input/comparison identity; leave Metal bytes unchanged."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
from nr_candidate_primitive import validate_primitive

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def sealed(p):
    r=json.loads(p.read_text());b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h;return r
def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',required=True,type=Path);p.add_argument('--reference',required=True,type=Path);p.add_argument('--image',required=True,type=Path);p.add_argument('--output',required=True,type=Path);p.add_argument('--guard-script',required=True,type=Path);a=p.parse_args();a.output=a.output.resolve()
    parent=sealed(a.parent/'manifest.json');validate_primitive(a.parent,parent)
    for name,h in parent['files'].items():assert sha(a.parent/name)==h,name
    ref=sealed(a.reference/'reference.json');transfer=sealed(a.reference/'retrieval.json')
    assert ref.get('boundary','block0')=='block0','Block0 requires its own boundary reference'
    assert ref['result']=='pass' and ref['fresh_prefix_matrix_proven'] and ref['runtime_input_use_forbidden'] and transfer['result']=='pass' and transfer['reference_root']==ref['receipt_root_sha256']
    assert sha(a.image)==ref['input_sha256'] and sha(a.reference/'comparison-pool.bin')==ref['comparison_pool_sha256']==transfer['comparison_pool_sha256']
    for key in ('input_sha256','comparison_pool_sha256'):assert re.fullmatch('[0-9a-f]{64}',ref[key])
    assert not a.output.exists();shutil.copytree(a.parent/'pre',a.output)
    source=a.output/'source/nr_pre/pre_probe.mm';text=source.read_text()
    for old,new in [('81ec1de5d1e76e6b55c4fd987b3732180c9720919fbd760dd5b4e8aa8cdd0342',ref['input_sha256']),('1e340ab4b04af0107bba1a62e1667e304f90d62123651f019efbf1abc77a01db',ref['comparison_pool_sha256'])]:
        assert text.count(old)==1,'parent identity gate changed';text=text.replace(old,new)
    source.write_text(text)
    source_manifest=a.output/'source/source-manifest.json';sm=json.loads(source_manifest.read_text())
    sm['case_binding']={'parent_source_manifest_sha256':sha(a.parent/'pre/source/source-manifest.json'),'reference_root':ref['receipt_root_sha256'],'input_sha256':ref['input_sha256'],'comparison_pool_sha256':ref['comparison_pool_sha256']}
    sm['files']={str(p.relative_to(source_manifest.parent)):sha(p) for p in sorted(source_manifest.parent.rglob('*')) if p.is_file() and p!=source_manifest}
    source_manifest.write_text(json.dumps(sm,indent=2)+'\n')
    # The expected pool is still opened after the final GPU command, unchanged.
    assert text.index('auto expected=Read(argv[4])')>text.rindex('Complete(cmd)')
    subprocess.run([sys.executable,a.guard_script,'--me','Codex'],check=True)
    command=['xcrun','-sdk','macosx','clang++','-std=c++20','-O2','-fobjc-arc','-Wall','-Wextra','-Werror','-Wno-deprecated-declarations',str(source),'-framework','Foundation','-framework','Metal','-o',str(a.output/'probe')]
    with (a.output/'case-host-build.log').open('x') as log:subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert sha(a.output/'pre.metallib')==sha(a.parent/'pre/pre.metallib')
    r={'schema':'neuralstudio.nr.case-bound-block0-host.v1','case':ref['case'],'parent_bundle_root':parent['receipt_root_sha256'],'primitive_root':parent['primitive_root'],'reference_root':ref['receipt_root_sha256'],
       'image':str(a.image.resolve()),'input_sha256':ref['input_sha256'],'comparison_pool':str((a.reference/'comparison-pool.bin').resolve()),'comparison_pool_sha256':ref['comparison_pool_sha256'],
       'source_sha256':sha(source),'source_manifest_sha256':sha(source_manifest),'probe_sha256':sha(a.output/'probe'),'metallib_sha256':sha(a.output/'pre.metallib'),'command':command,'shader_changes':False,'reference_gpu_input':False,'result':'built','validation':'pending'}
    r['receipt_root_sha256']=canonical(r);(a.output/'case-host.json').write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])
if __name__=='__main__':main()
