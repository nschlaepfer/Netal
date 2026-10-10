#!/usr/bin/env python3
"""Independent CPU-only review witness for the modeled-FP32 allowance overflow.

Not a competing census or Metal prototype. All operands are finite E4M3 values;
the seed252 is exactly half(1 * half(252)). Every NVIDIA K32 sum cancels its
products exactly, so the staged target is252, while a legal FP32 order differs.
"""
import argparse
from fractions import Fraction as F
import hashlib
import json
from pathlib import Path
import numpy as np

def p2(e):return F(2**e) if e>=0 else F(1,2**(-e))
def exponent(x):
    e=x.numerator.bit_length()-x.denominator.bit_length()
    if x<p2(e):e-=1
    assert p2(e)<=x<p2(e+1);return e
def half_error(u):return p2(max(exponent(u)-11,-25)) if u else p2(-25)
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True,type=Path);a=p.parse_args();assert not a.output.exists()
    pairs=[];group_products=[]
    for step in range(32):
        big=[(448,448)]*6+([(384,256)]*4 if step<8 else [(384,256)]*3+[(256,256)])
        group=big+[(F(1,8),F(1,4))]*6+[(x,-y) for x,y in big]+[(F(1,8),F(-1,4))]*6
        assert len(group)==32;products=[F(x)*F(y) for x,y in group]
        assert sum(products)==0 and all(v/F(1,512)==int(v/F(1,512)) for v in products)
        pairs.extend(group);group_products.append(products)
    # M=16 for every K32 group (operand exponents8+8), Delta=2^-9.
    # Products and all actual carries (252 or0) are aligned, so all truncations
    # and half roundings preserve each quarter's exact seed; merges give252.
    seed=F(252);target=seed;terms=[v for group in group_products for v in group]
    assert len(terms)==1024 and seed+sum(terms)==252
    approximate=np.float32(seed)
    for x,y in pairs:
        product=np.float32(x)*np.float32(y)
        assert F(float(product))==F(x)*F(y),'products are exact FP32'
        approximate=np.float32(approximate+product)
    l1=abs(seed)+sum(map(abs,terms));scaled=int(l1*(1<<30))
    exact_allowance=F((scaled*1024)//(1<<24)+1,1<<30)
    with np.errstate(over='ignore'):
        wrapped_allowance=F(int(np.int64(scaled)*np.int64(1024))//(1<<24)+1,1<<30)
    assert exact_allowance>abs(F(float(approximate))-target)>wrapped_allowance
    # Reproduce the candidate's oracle-free primary bound using exact fractions,
    # not the candidate's implementation or exact intermediate reference states.
    errors=[]
    for z in range(4):
        ideal=seed if z==0 else F(0);error=F(0)
        for _ in range(8):
            upper=abs(ideal)+error;delta=F(1,512)
            epre=error+32*delta+(delta if upper else 0)
            error=epre+half_error(abs(ideal)+epre)
        errors.append(error)
    nv_bound=errors[0]
    for z in (1,2,3):
        before=nv_bound+errors[z];nv_bound=before+half_error(abs(seed)+before)
    bad_radius=nv_bound+wrapped_allowance
    lo=F(float(approximate))-bad_radius;hi=F(float(approximate))+bad_radius
    assert F(232)<lo<=hi<F(248),'bad approximate-centred interval lies strictly in FP8=240 cell'
    assert F(248)<target<F(272),'NVIDIA target lies strictly in FP8=256 cell'
    ideal_lo=seed-bad_radius;ideal_hi=seed+bad_radius
    assert F(248)<ideal_lo<=ideal_hi<F(272),'ideal-centred census hides predictor shift'
    record={'schema':'neuralstudio.nr.projection-bound-review-counterexample.v1','result':'counterexample_reproduced','cpu_only':True,'metal_executed':False,'terms':1024,
      'seed_half':252,'nvidia_staged_value_proved':252,'nvidia_fp8_value':256,'legal_fp32_chain_value':float(approximate),'fp32_prediction_fp8_value':240,
      'l1_including_seed':float(l1),'correct_modeled_allowance':float(exact_allowance),'wrapped_int64_allowance':float(wrapped_allowance),'primary_nv_bound':float(nv_bound),
      'bad_predictor_centred_interval':[float(lo),float(hi)],'ideal_centred_interval':[float(ideal_lo),float(ideal_hi)],
      'conclusion':'The finite-input int64 multiplication by1024 overflows before division. A legal FP32 chain then falls outside the claimed allowance and a predictor-centred certificate can accept the wrong FP8 byte. This is not a counterexample to the primary ideal-dot census or a measured Metal result.',
      'repair':'Cancel1024/2^24 first and compute an overflow-safe outward ceiling of scaled_L1/2^14; document the summation theorem/conditions. Classify about the actually modeled predictor, not only the exact ideal dot.',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    record['receipt_root_sha256']=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':')).encode()).hexdigest();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record,indent=2))
if __name__=='__main__':main()
