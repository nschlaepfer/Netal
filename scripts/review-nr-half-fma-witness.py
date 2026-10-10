#!/usr/bin/env python3
"""Integer-exact positive square/add census; expected data stays CPU comparison-only."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r): return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def half_integer(b):
    e=(b>>10)&31; f=b&1023
    assert e<31
    n=(1024+f)<<(e-1) if e else f
    return -n if b&32768 else n


def rounded_integer(n,precision,emin):
    # All arithmetic is exact Python integer arithmetic on the 2^-48 lattice.
    assert n>=0
    shift=max(0,n.bit_length()-precision,emin-(precision-1)+48)
    if not shift:return n
    q,r=divmod(n,1<<shift);half=1<<(shift-1)
    return (q+int(r>half or (r==half and q&1)))<<shift


def half_bits(n):
    r=rounded_integer(n,11,-14)
    if not r:return 0
    exponent=r.bit_length()-1-48
    if exponent>15:return 0x7c00
    if exponent< -14:return r>>24
    significand=r>>(exponent+48-10)
    return ((exponent+15)<<10)+(significand-1024)


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--rtx',type=Path);a=p.parse_args()
    src=json.loads((a.source/'inputs.json').read_text());b=dict(src);h=b.pop('receipt_root_sha256');assert canonical(b)==h=='c6fd4c5e02bbb2db0e6264d60dfc854b767edbc80394ff272d5af46e1cf74f80'
    assert sha(a.source/'operands.bin')==src['files']['operands.bin']
    # Identity and tie tests across the complete finite nonnegative half lattice.
    for bits in range(0x7c00):assert half_bits(half_integer(bits)<<24)==bits
    assert half_bits((half_integer(0x4da6)+half_integer(0x4da7))<<23)==0x4da6
    assert half_bits(((half_integer(0x4da6)+half_integer(0x4da7))<<23)+1)==0x4da7
    expected=bytearray();diff=[]
    triples=list(struct.iter_unpack('<HHH',(a.source/'operands.bin').read_bytes()))
    for i,(x,y,z) in enumerate(triples):
        assert x==y
        exact=half_integer(x)*half_integer(y)+(half_integer(z)<<24)
        direct=half_bits(exact);double=half_bits(rounded_integer(exact,24,-126))
        expected.extend(struct.pack('<HH',direct,double))
        if direct!=double:diff.append({'index':i,'operands':[x,y,z],'single':direct,'double':double})
    witness=next(r for r in diff if r['index']==src['witness_index']);assert witness['single']==0x4da7 and witness['double']==0x4da6
    runs=[]
    if a.rtx:
        job=json.loads((a.rtx/'receipt.json').read_text(encoding='utf-8-sig'))
        assert job['result']=='pass' and job['inputs_root_sha256']==h
        for f in job['artifacts']:assert (a.rtx/'output'/f['name']).stat().st_size==f['bytes'] and sha(a.rtx/'output'/f['name'])==f['sha256']
        for label in ('a5-a','a5-b','5a-a','5a-b'):
            r=json.loads((a.rtx/'output'/(label+'.json')).read_text());assert r['errors']==0 and r['input_immutable'] and not r['expected_input_opened'] and r['count']==65536
            assert (a.rtx/'output'/(label+'.bin')).read_bytes()==expected
            counts=(a.rtx/'output'/(label+'.coverage.bin')).read_bytes();assert counts==struct.pack('<I',1)*65536
            runs.append({'label':label,'pid':r['pid'],'poison':r['poison'],'output_sha256':sha(a.rtx/'output'/(label+'.bin'))})
        assert len({r['pid'] for r in runs})==4
        assert [r['poison'] for r in runs]==[165,165,90,90]
    assert not a.output.exists();a.output.mkdir(parents=True)
    (a.output/'expected-comparison-only.bin').write_bytes(expected)
    r={'schema':'neuralstudio.nr.half-fma-witness-review.v1','fixture_root':h,'count':len(triples),'witness':witness,
       'double_rounding_differences':diff,'real_norm_double_rounding_count':sum(d['index']<2048 for d in diff),'rtx_runs':runs,
       'result':'pass_bounded_RTX_primitive' if runs else 'CPU_preliminary','CPU_exact_integer_units':'2^-48',
       'expected_bytes_sha256':hashlib.sha256(expected).hexdigest(),'source_sha256':sha(Path(__file__)),
       'scope':'All real norm triples from one archived window plus finite-half square sweep at one fixed addend; not arbitrary FMA or full inference proof.'}
    r['receipt_root_sha256']=canonical(r);(a.output/'audit.json').write_text(json.dumps(r,indent=2)+'\n')
    print(json.dumps({k:v for k,v in r.items() if k!='double_rounding_differences'},indent=2))


if __name__=='__main__':main()
