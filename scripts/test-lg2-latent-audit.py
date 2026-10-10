#!/usr/bin/env python3
"""Keep conditional CPU mathematics separate from retained hardware observations."""
import importlib.util
from pathlib import Path
from fractions import Fraction
path=Path(__file__).with_name('audit-lg2-latent-model.py');spec=importlib.util.spec_from_file_location('audit',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
a,b,_=module.bounds(0)
c,d,_=module.bounds(0xbf7ffffe)
assert b < c+1 # recorded log2(.5) contradicts latent log2(1) plus -1
c,d,_=module.bounds(0xbf800000)
assert max(a,c+1)<min(b,d+1) # an exact -1 result would be consistent
for word in (0x3f800000,0xbf800000,0x3f000001,0xbf000001):
    low,high,inclusive=module.bounds(word);value=Fraction(module.word_value(word))
    assert low<value<high and inclusive==(word%2==0)
print('LG2 interval tests pass; retained counterexample remains rejected')
