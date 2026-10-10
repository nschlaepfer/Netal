#!/usr/bin/env python3
"""Test a common-latent-value LG2 hypothesis against retained scalar observations only."""
import argparse,hashlib,json,struct,math
from fractions import Fraction
from pathlib import Path

def word_value(word):return struct.unpack('<f',struct.pack('<I',word))[0]
def bounds(word):
    value=word_value(word)
    if not math.isfinite(value):raise ValueError('nonfinite output')
    if value==0:below,above=-word_value(1),word_value(1)
    elif value>0:below,above=word_value(word-1),word_value(word+1)
    else:below,above=word_value(word+1),word_value(word-1)
    return ((Fraction(below)+Fraction(value))/2,(Fraction(value)+Fraction(above))/2,word%2==0)
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('binade','keys','values','output'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists() or args.output.is_symlink():parser.error('Output exists; preserve the previous audit')
    base=args.binade.read_bytes();keys=args.keys.read_bytes();values=args.values.read_bytes()
    if len(base)!=33554432 or hashlib.sha256(base).hexdigest()!='7fd9be9a924e40a0ee18ba20e7f5158135112947597c8dd86131d3d4d8f9dbfa':parser.error('Binade identity mismatch')
    if len(keys)!=4194304 or len(values)!=len(keys):parser.error('Scalar table geometry mismatch')
    checked=0;excluded=0;failures=[];by_exponent={};intersections={}
    for offset in range(0,len(keys),4):
        key=struct.unpack_from('<I',keys,offset)[0]
        if not key:continue
        exponent=(key>>23)&255
        if key>>31 or exponent in (0,255):excluded+=1;continue
        expected=struct.unpack_from('<I',values,offset)[0]
        if not math.isfinite(word_value(expected)):excluded+=1;continue
        mantissa=key&0x7fffff;e=exponent-127
        base_word=struct.unpack_from('<I',base,mantissa*4)[0]
        low,high,inclusive=bounds(base_word);left,right,other_inclusive=bounds(expected);left-=e;right-=e
        if mantissa in intersections:
            low,high,inclusive_low,inclusive_high=intersections[mantissa]
        else:inclusive_low=inclusive_high=inclusive
        new_low=max(low,left);new_high=min(high,right)
        il=(inclusive_low if low>left else other_inclusive if left>low else inclusive_low and other_inclusive)
        ih=(inclusive_high if high<right else other_inclusive if right<high else inclusive_high and other_inclusive)
        valid=new_low<new_high or (new_low==new_high and il and ih)
        checked+=1;entry=by_exponent.setdefault(str(e),{'checked':0,'inconsistent':0});entry['checked']+=1
        if valid:intersections[mantissa]=(new_low,new_high,il,ih)
        else:
            entry['inconsistent']+=1
            if len(failures)<16:failures.append({'input':hex(key),'exponent':e,'mantissa':mantissa,'binade_output':hex(base_word),'observed_output':hex(expected)})
    bad=sum(r['inconsistent'] for r in by_exponent.values())
    report={'schema':'netal.lg2.latent-model-audit.v1','hypothesis':'single latent log2 mantissa value plus integer exponent, rounded to nearest-even float32','scope':'retained scalar observations; not independent NVIDIA execution',
            'checked':checked,'excluded':excluded,'inconsistent':bad,'consistent_with_hypothesis':bad==0,'general_behavior_verified':False,'by_exponent':by_exponent,'examples':failures,
            'inputs':{name:hashlib.sha256(getattr(args,name).read_bytes()).hexdigest() for name in ('binade','keys','values')}}
    args.output.write_text(json.dumps(report,indent=2)+'\n');print(f'checked={checked} inconsistent={bad} general_behavior_verified=false')
if __name__=='__main__':main()
