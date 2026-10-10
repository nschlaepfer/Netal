#!/usr/bin/env python3
"""Prove the agreed split input-view formula equals a C16-planar export."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve proof')
    seen=bytearray(1105920)
    for r in range(36):
        for c in range(60):
            for ch in range(512):
                consumer=(ch//32)*69120+(r//2+18*((ch%32)//16))*1920+(c//2+30*(r%2))*32+(ch%16)+16*(c%2)
                producer=((ch//16)*36*60+r*60+c)*16+ch%16
                if consumer!=producer:raise ValueError('producer/consumer ABI differs')
                seen[consumer]+=1
    if not all(x==1 for x in seen):raise ValueError('ABI is not bijective')
    r={'schema':'neuralstudio.nr.middle-input-view-bijection.v1','shape':[36,60,512],'checked_byte_offsets':len(seen),'offset_mismatches':0,'exactly_once':True,
       'consumer_launches':[25,26],'same_buffer_reused':True,'proof_kind':'integer address equivalence only','model_execution_proven':False,'result':'pass'}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(r,indent=2)+'\n');print(r['receipt_root_sha256'])


if __name__=='__main__':main()
