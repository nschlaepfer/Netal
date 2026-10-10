#!/usr/bin/env python3
"""Exhaustive finite-half CPU expression census, not GPU/model validation."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def canonical(r):return hashlib.sha256(json.dumps(r,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    repo=Path(__file__).resolve().parents[1]
    helper=repo/'scripts/review-nr-half-fma-witness.py'
    assert sha(helper)=='b8a76da8390ffa7f4c6727dadf6598a00e323e973cad3bc380bc92196a15b35e'
    spec=importlib.util.spec_from_file_location('exact',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    bundle=repo/'build/nr_encoder_fscale/bundle-v1'
    parent=json.loads((bundle/'manifest.json').read_text());body=dict(parent);root=body.pop('receipt_root_sha256')
    assert canonical(body)==root=='3350d5f53e391a3be260c2d1d5b60065a9e23e24ba3d9f871f8bf9037ef2d633'
    sources={}
    for name in ('swin32_mlp.metal','swin32_block.metal','swin_wide.metal'):
        key='encoder/source/'+name;assert sha(bundle/key)==parent['files'][key];sources[key]=parent['files'][key]
    def half(n):return m.half_bits(abs(n))|(0x8000 if n<0 else 0)
    def fma(x,y,z,double):
        n=m.half_integer(x)*m.half_integer(y)+(m.half_integer(z)<<24)
        if double:n=(-1 if n<0 else 1)*m.rounded_integer(abs(n),24,-126)
        # These two expression families have no negative-zero FMA result.
        return half(n)
    def multiply(x,y):
        n=m.half_integer(x)*m.half_integer(y)
        return half(n) if n else (x^y)&0x8000
    def e4(bits):
        sign=(bits>>8)&128;e=(bits>>10)&31;f=bits&1023
        def shift(n,s):
            q,r=divmod(n,1<<s);mid=1<<(s-1)
            return q+int(r>mid or (r==mid and q&1))
        if e==31:return sign|(127 if f else 126)
        if bits&0x7fff>=0x5f00:return sign|126
        if e<9:return sign|shift(f+1024 if e else f,16-e if e else 15)
        significand=shift(1024+f,7);exponent=e-8
        if significand==16:significand=8;exponent+=1
        return sign|min(126,(exponent<<3)+significand-8)
    counts={'gelu_t0':0,'gelu_t1':0,'gelu_half_output':0,'gelu_fp8_output':0,'score_fma':0,'score_clamp':0,'score_exponential_bits':0}
    differences=[];single=bytearray();double=bytearray();domain=0
    for bits in range(65536):
        if bits&0x7c00==0x7c00:continue
        domain+=1;clip=(bits&0x8000)|min(bits&0x7fff,0x4400)
        u0=fma(clip&0x7fff,0xab28,0x3728,True);u1=fma(clip&0x7fff,0xab28,0x3728,False)
        v0=fma(clip,u0,0x3b28,True);v1=fma(clip,u1,0x3b28,False)
        z0,z1=multiply(bits,v0),multiply(bits,v1)
        counts['gelu_t0']+=u0!=u1;counts['gelu_t1']+=v0!=v1
        counts['gelu_half_output']+=z0!=z1;counts['gelu_fp8_output']+=e4(z0)!=e4(z1)
        if u0!=u1:differences.append({'input_bits':bits,'old_t0':u0,'single_t0':u1,'common_t1':v0 if v0==v1 else None})
        c0=fma(bits,0x29c0,0x3d34,True);c1=fma(bits,0x29c0,0x3d34,False)
        clamp=lambda x:0x3c20 if x&0x8000 else min(max(x,0x3c20),0x3e47)
        out0=((clamp(c0)<<5)+0x8000)&65535;out1=((clamp(c1)<<5)+0x8000)&65535
        counts['score_fma']+=c0!=c1;counts['score_clamp']+=clamp(c0)!=clamp(c1);counts['score_exponential_bits']+=out0!=out1
        double.extend(bytes([e4(z0)])+out0.to_bytes(2,'little'));single.extend(bytes([e4(z1)])+out1.to_bytes(2,'little'))
    assert domain==63488 and single==double
    assert counts=={'gelu_t0':2,'gelu_t1':0,'gelu_half_output':0,'gelu_fp8_output':0,'score_fma':0,'score_clamp':0,'score_exponential_bits':0}
    assert not a.output.exists();a.output.mkdir(parents=True)
    r={'schema':'neuralstudio.nr.scalar-half-expression-census.v1','result':'pass_CPU_expression_equivalence','finite_half_inputs':domain,
       'parent_bundle_root':root,'source_hashes':sources,'exact_helper_sha256':sha(helper),'auditor_sha256':sha(Path(__file__)),
       'differences':counts,'internal_t0_counterexamples':differences,'stored_expression_output_sha256':hashlib.sha256(single).hexdigest(),
       'source_changed':False,'GPU_executed':False,'limits':['CPU comparison of the declared FP32-FMA-then-half and single-half-FMA expression schedules, not compiled GPU placement/precision proof.',
       'All finite half inputs, including signed zero and finite values whose activation saturates; nonfinite inputs are excluded.',
       'GELU t0 is not bit-identical for two inputs; only subsequent t1 and stored activation reconverge. Do not claim all intermediate arithmetic equal.',
       'This does not address the two-input norm expression or new model/sequence inputs; no optimization or backend change.']}
    r['receipt_root_sha256']=canonical(r);(a.output/'audit.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r,indent=2))


if __name__=='__main__':main()
