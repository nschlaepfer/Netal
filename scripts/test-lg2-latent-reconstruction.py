#!/usr/bin/env python3
"""Test the proposed rounding reconstruction against a common-latent-value model, not NVIDIA."""
import math,random,struct

def f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def bits(value):return struct.unpack('<I',struct.pack('<f',value))[0]
def value(word):return struct.unpack('<f',struct.pack('<I',word))[0]
def lsb_power(v):
    word=bits(v)&0x7fffffff
    if not word:return -1000
    mantissa=(word&0x7fffff)|0x800000
    return (word>>23)-127-23+((mantissa&-mantissa).bit_length()-1)
def prepare(fraction):
    a,b=f32(fraction),f32(fraction-1)
    v,offset=(a,0) if abs(a)<=abs(b) else (b,1)
    power=lsb_power(v);direction=0
    if -24<=power<=-18:
        exponent=1<<(power+24)
        base=f32(exponent+offset+v);expected=f32(exponent+fraction)
        direction=(expected>base)-(expected<base)
    return a,b,v,offset,power,direction

def reconstruct(table,e):
    a,b,v,offset,power,direction=table
    if e==0:return a
    if e==-1:return b
    result=f32(e+offset+v)
    magnitude=e if e>0 else -e-1
    p=magnitude.bit_length()-1
    if power==p-24 and direction:
        result=value(bits(result)+(direction if result>=0 else -direction))
    return result

rng=random.Random(1307)
fractions=[math.log2(value(0x3f800000|rng.randrange(1<<23))) for _ in range(4096)]
for power in range(-24,-17):
    for sign in (-1,0,1):
        for n in range(1,64,2):fractions.append(n*(2.0**power)+sign*(2.0**(power-8)))
count=0;corrections=0
for fraction in fractions:
    table=prepare(fraction);corrections+=bool(table[-1])
    for e in range(-126,128):
        assert bits(reconstruct(table,e))==bits(f32(e+fraction)),(fraction,e,table,reconstruct(table,e),f32(e+fraction))
        count+=1
print(f'common_latent_cases={count} correction_rows={corrections} result=pass not_an_NVIDIA_implementation=true')
