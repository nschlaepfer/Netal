#!/usr/bin/env python3
"""Read-only arithmetic hypothesis audit of fresh RTX discriminator operands.

No model/emulator execution, no peer changes and no hypothesis promotion.
Finite E4M3 products and half C fit exactly in the binary64 sum used here.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def f8(b):
    e=(b>>3)&15;f=b&7
    assert not (e==15 and f==7),'finite inputs required'
    value=np.ldexp(float(f if e==0 else 8+f),-9 if e==0 else e-10)
    return -value if b&128 else value
def h(b):return np.array([b],dtype='<u2').view(np.float16).astype(np.float64)[0]
def decode(case):
    A=np.zeros((16,32),dtype=np.float64);B=np.zeros((32,8),dtype=np.float64);C=np.zeros((16,8),dtype=np.float64)
    for lane in case['lanes']:
        l=lane['lane'];g=l//4;t=l%4;a=[int(x,16) for x in lane['a']];b=[int(x,16) for x in lane['b']];c=[int(x,16) for x in lane['c']]
        for j in range(4):
            A[g,4*t+j]=f8((a[0]>>(8*j))&255);A[g+8,4*t+j]=f8((a[1]>>(8*j))&255)
            A[g,16+4*t+j]=f8((a[2]>>(8*j))&255);A[g+8,16+4*t+j]=f8((a[3]>>(8*j))&255)
            B[4*t+j,g]=f8((b[0]>>(8*j))&255);B[16+4*t+j,g]=f8((b[1]>>(8*j))&255)
        for reg in (0,1):C[g+8*reg,2*t]=h(c[reg]&65535);C[g+8*reg,2*t+1]=h(c[reg]>>16)
    assert np.all(np.isfinite(C));return A,B,C
def pack(values):
    v=values.astype(np.float16).view(np.uint16);out=np.zeros((32,2),dtype=np.uint32)
    for l in range(32):
        g=l//4;t=l%4
        for r in (0,1):out[l,r]=int(v[g+8*r,2*t])|(int(v[g+8*r,2*t+1])<<16)
    return out
def main():
    p=argparse.ArgumentParser();p.add_argument('spec',type=Path);p.add_argument('rtx',type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    spec=json.loads(a.spec.read_text());actual=np.fromfile(a.rtx,dtype='<u4').reshape(-1,32,2);cases=spec.get('cases',[spec]);assert len(cases)==len(actual)
    errors=[];aligned={p:0 for p in range(18,33)};field_aligned={p:0 for p in range(23,28)};field_examples={p:[] for p in field_aligned}
    for ci,case in enumerate(cases):
        A,B,C=decode(case);terms=np.stack([np.outer(A[:,k],B[k,:]) for k in range(32)]+[C]);total=terms.sum(axis=0,dtype=np.float64);prediction=pack(total)
        for l,r in zip(*np.nonzero(prediction!=actual[ci])):errors.append({'case':ci,'lane':int(l),'register':int(r),'actual':f'{actual[ci,l,r]:08x}','exact_sum':f'{prediction[l,r]:08x}'})
        # A bounded diagnostic family, not a recovered Tensor Core algorithm.
        exponents=np.frexp(np.abs(terms))[1]-1;exponents=np.where(terms==0,-1024,exponents);maxexp=exponents.max(axis=0)
        for precision in aligned:
            scale=maxexp-precision
            magnitudes=np.floor(np.ldexp(np.abs(terms),-scale)).astype(np.int64)
            integers=np.where(terms<0,-magnitudes,magnitudes).sum(axis=0,dtype=np.int64)
            y=np.ldexp(integers.astype(np.float64),scale);aligned[precision]+=int(np.count_nonzero(pack(y)!=actual[ci]))
        ae=np.where(A==0,-1024,np.frexp(np.abs(A))[1]-1);be=np.where(B==0,-1024,np.frexp(np.abs(B))[1]-1)
        product_exponents=np.stack([ae[:,k,None]+be[k,None,:] for k in range(32)])
        # HMMA-like alignment uses the sum of operand exponents, not the
        # potentially one-bit-larger normalized product exponent.
        maximum=np.maximum(product_exponents.max(axis=0),np.where(C==0,-1024,np.frexp(np.abs(C))[1]-1))
        for precision in field_aligned:
            scale=maximum-precision;magnitudes=np.floor(np.ldexp(np.abs(terms),-scale)).astype(np.int64)
            integers=np.where(terms<0,-magnitudes,magnitudes).sum(axis=0,dtype=np.int64);prediction=pack(np.ldexp(integers.astype(np.float64),scale))
            bad=list(zip(*np.nonzero(prediction!=actual[ci])));field_aligned[precision]+=len(bad)
            field_examples[precision]+=[{'case':ci,'lane':int(l),'register':int(r),'actual':f'{actual[ci,l,r]:08x}','hypothesis':f'{prediction[l,r]:08x}'} for l,r in bad]
    r={'schema':'neuralstudio.nr.qmma-offline-exact-sum-audit.v1','spec_sha256':sha(a.spec),'rtx_output_sha256':sha(a.rtx),'cases':len(cases),
       'exact_sum_mismatching_registers':len(errors),'exact_sum_counterexamples':errors,
       'truncate_terms_at_max_floor_exponent_minus_precision':aligned,
       'truncate_terms_at_max_operand_exponent_sum_minus_precision':field_aligned,
       'operand_exponent_sum_counterexamples':field_examples,
       'runtime_changed':False,'general_rule_proven':False}
    r['receipt_root_sha256']=hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.output.write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))
if __name__=='__main__':main()
