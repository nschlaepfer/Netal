#!/usr/bin/env python3
"""Compose the accepted changing-image host with the accepted spatial-motion recipe."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1];parent=repo/'build/nr_spatial_motion/source-v1'
    r=json.loads((parent/'source-receipt.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256')
    assert canonical(b)==h=='623ff5aeafdedbbcfe150f1ccd5994e6371afdd07f049bb03114e473e8ccc052'
    for n,digest in r['files'].items():assert sha(parent/n)==digest,n
    helper=repo/'scripts/prepare-nr-image-sequence.py'
    spec=importlib.util.spec_from_file_location('images',helper);images=importlib.util.module_from_spec(spec);spec.loader.exec_module(images)
    assert not a.output.exists();shutil.copytree(parent,a.output,ignore=shutil.ignore_patterns('source-receipt.json'))
    old='rgba16-color-cases-v1_depth-bands-v1_motion-cases-v2'
    temp='rgba16-color-cases-v1_depth-bands-v1_motion-zero'
    temp_new='rgba16-image-sequence-v1_depth-bands-v1_motion-zero'
    new='rgba16-image-sequence-v1_depth-bands-v1_motion-cases-v2'
    for name,transform,count in [('color_cases_trace.cpp',images.host,1),('capture-color.ps1',images.capture,2)]:
        s=(parent/name).read_text();assert s.count(old)==count
        # The reused transform only recognizes the old recipe label; changing
        # that label temporarily does not edit the spatial-motion instructions.
        s=transform(s.replace(old,temp));assert s.count(temp_new)==count
        (a.output/name).write_text(s.replace(temp_new,new))
    (a.output/'run-nr-reset-sequence.ps1').write_text(images.runner((parent/'run-nr-reset-sequence.ps1').read_text()))
    assert sha(a.output/'prefix_proxy.cpp')==r['files']['prefix_proxy.cpp']
    result={'schema':'neuralstudio.nr.combined-sequence-source.v1','parent_source_root':h,'image_transform_sha256':sha(helper),
            'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip(),
            'frames':3,'image_sequences':{'static':[0,0,0],'mixed':[0,2,1]},'motion_cases':[0,1,2],
            'planned_target':{'image_sequence':'mixed','motion_case':2,'reset_mask':[1,0,0]},
            'controls':['static spatial equals accepted spatial reference','mixed zero equals accepted changing-image reference'],
            'proxy_model_unchanged':True,'expected_bytes_staged':False,'executed':False,
            'files':{str(p.relative_to(a.output)):sha(p) for p in sorted(a.output.rglob('*')) if p.is_file()}}
    result['receipt_root_sha256']=canonical(result);(a.output/'source-receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(result['receipt_root_sha256'])


if __name__=='__main__':main()
