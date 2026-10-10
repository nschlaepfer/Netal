#!/usr/bin/env python3
"""Losslessly package one comparison pool for transfer; retain all originals."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil

def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('reference',type=Path);a=p.parse_args();r=json.loads((a.reference/'reference.json').read_text());b=dict(r);h=b.pop('receipt_root_sha256')
    assert hashlib.sha256(json.dumps(b,sort_keys=True,separators=(',',':')).encode()).hexdigest()==h
    assert r['result']=='pass' and r['fresh_prefix_matrix_proven'] and r['runtime_input_use_forbidden']
    source=a.reference/'comparison-pool.bin';assert sha(source)==r['comparison_pool_sha256']
    target=a.reference/'comparison-pool.bin.gz';assert not target.exists()
    with source.open('rb') as inp,gzip.open(target,'xb',compresslevel=1) as out:shutil.copyfileobj(inp,out)
    digest=hashlib.sha256();size=0
    with gzip.open(target,'rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):digest.update(chunk);size+=len(chunk)
    assert digest.hexdigest()==sha(source) and size==source.stat().st_size
    out={'schema':'neuralstudio.nr.prefix-transfer.v1','reference_root':h,'reference_sha256':sha(a.reference/'reference.json'),'compressed_sha256':sha(target),'uncompressed_sha256':sha(source),'uncompressed_bytes':size,'compressed_bytes':target.stat().st_size,'original_retained':True,'runtime_input_use_forbidden':True,'result':'pass'}
    out['receipt_root_sha256']=hashlib.sha256(json.dumps(out,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    with (a.reference/'transfer.json').open('x') as f:json.dump(out,f,indent=2)
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
