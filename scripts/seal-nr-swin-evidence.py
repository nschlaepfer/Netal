#!/usr/bin/env python3
"""Freeze a verified NR stage receipt after checking its implementation pins."""
import argparse
import hashlib
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('receipt',type=Path);p.add_argument('destination',type=Path);a=p.parse_args()
    if a.destination.exists():raise ValueError('preserve sealed receipt')
    repo=Path(__file__).resolve().parents[1];r=json.loads(a.receipt.read_text())
    body=dict(r);root=body.pop('receipt_root_sha256')
    if root!=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':')).encode()).hexdigest():raise ValueError('canonical root mismatch')
    if r['result']!='pass' or r['mismatched_bytes']!=0 or r['claims']['full_model_inference'] is not False:raise ValueError('scope/result mismatch')
    if r.get('accepted',True) is not True or r.get('two_poison_repeat_proven',True) is not True:raise ValueError('non-accepted diagnostic cannot be sealed as acceptance')
    for name,digest in r['source_sha256'].items():
        if sha(repo/name)!=digest:raise ValueError('source drift: '+name)
    a.destination.parent.mkdir(parents=True,exist_ok=True)
    a.destination.write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
    print('sealed_root_sha256='+root)


if __name__=='__main__':main()
