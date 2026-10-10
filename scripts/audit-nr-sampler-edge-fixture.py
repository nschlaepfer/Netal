#!/usr/bin/env python3
"""Independent CPU regeneration of synthetic atlas/query bytes; no oracle claim."""
import argparse
import hashlib
import json
from pathlib import Path
import struct
def main():
    p=argparse.ArgumentParser();p.add_argument('fixture',type=Path);a=p.parse_args()
    r=json.loads((a.fixture/'fixture.json').read_text());assert (r['width'],r['height'])==(1920,1080)
    assert len(r['cases'])==22 and len({c['name'] for c in r['cases']})==22
    w,h=r['width'],r['height'];image=bytearray(w*h*8);qs=bytearray();fr=[0,1/512,127/256,.5,129/256,255/256,1]
    for n,c in enumerate(r['cases']):
        assert len(c['taps'])==4 and all(type(v)is int and 0<=v<=65535 for v in c['taps'])
        for tap in range(4):
            x=8*n+2+tap%2;y=8+tap//2
            for ch in range(4):struct.pack_into('<H',image,(y*w+x)*8+ch*2,c['taps'][(tap+ch)%4])
        for y in fr:
            for x in fr:qs.extend(struct.pack('<ff',(8*n+2+.5+x)/w,(8.5+y)/h))
    assert hashlib.sha256(image).hexdigest()==r['input_sha256']
    assert bytes(qs)==(a.fixture/'queries.bin').read_bytes()
    assert hashlib.sha256(qs).hexdigest()==r['queries_sha256'] and len(qs)//8==r['queries']==1078
    print('Independent CPU fixture PASS: 22 cases, 1078 exact queries, full input hash; no RTX/Metal execution')
if __name__=='__main__':main()
