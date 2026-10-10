#!/usr/bin/env python3
"""Pack bounded combined-case evidence for CPU comparison, never runtime ingress."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--job-root',type=Path,required=True);a=p.parse_args();job=a.job_root.resolve()
    assert job.drive.upper()=='D:' and job.parent.name=='CodexLab' and job.name=='nr-color-reset-combined-20260906-v1'
    r=read(job/'matrix.json');b=dict(r);h=b.pop('receipt_root_sha256');assert canonical(b)==h=='45c6aedceb8a48eefd79961d37d657c3dfdeb973ca8fa9d91b2401ce6a3c5553'
    files={'matrix.json':job/'matrix.json','source-receipt.json':job/'source/source-receipt.json','build-receipt.json':job/'build-receipt.json'}
    for row in r['controls']+r['runs']:
        label=row['label'];ref=read(job/(label+'-reference.json'));body=dict(ref);root=body.pop('receipt_root_sha256');assert canonical(body)==root==row['receipt_root_sha256']
        files[label+'-reference.json']=job/(label+'-reference.json');files[label+'-job.json']=job/(label+'-receipt.json')
        for alias,name in [('capture','capture-receipt.json'),('inventory','artifact-inventory.json'),('host','host-output.txt')]:
            files[label+'-'+alias+('.txt' if alias=='host' else '.json')]=job/label/name
    capture=job/'combined-a5-a';inv={f['path']:f for f in read(capture/'artifact-inventory.json')['files']}
    for name in ('input-color.rgba16le','frame-1-input-color.rgba16le','frame-2-input-color.rgba16le','input-motion.rg16le','output.rgba16le','frame-1-output.rgba16le','frame-2-output.rgba16le'):
        path=capture/name;assert path.stat().st_size==inv[name]['bytes'] and sha(path)==inv[name]['sha256'];files[name]=path
    helper=job/'pack-nr-reset-retrieval.py';assert sha(helper)=='e9f94068aa9e663eb56c2dd209f6e60d298b549d9b09fa3fd15bdb84acaec9d0'
    spec=importlib.util.spec_from_file_location('pack',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    packed=m.pack(files,job/'transport-v1','combined-packet-v1')
    print(json.dumps({'matrix_root':h,'transport_root':packed['receipt_root_sha256'],'archive_bytes':packed['archive']['bytes'],'parts':len(packed['parts'])}))


if __name__=='__main__':main()
