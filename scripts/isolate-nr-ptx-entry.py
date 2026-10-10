#!/usr/bin/env python3
"""Extract one self-contained entry for compiler diagnostics and inspection."""
import argparse
import hashlib
from pathlib import Path
import re


def main():
    p=argparse.ArgumentParser();p.add_argument('source',type=Path);p.add_argument('--entry',required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve isolated source')
    text=a.source.read_text().rstrip('\0')
    hits=list(re.finditer(r'\.visible\s+\.entry\s+'+re.escape(a.entry)+r'\s*\(',text))
    if len(hits)!=1:raise ValueError('entry is missing or ambiguous')
    start=hits[0].start();body=text.index('{',hits[0].end());depth=1;end=body+1
    while depth and end<len(text):
        depth+=(text[end]=='{')-(text[end]=='}');end+=1
    if depth:raise ValueError('entry body is incomplete')
    entry=text[start:end]
    if re.search(r'\bcall(?:\.[\w]+)*\s',entry):raise ValueError('entry calls an external function; preserve its dependencies')
    header='\n'.join(re.search(pattern,text).group(0) for pattern in (
        r'\.version\s+[\d.]+',r'\.target\s+sm_\d+',r'\.address_size\s+\d+'))+'\n\n'
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(header+entry+'\n')
    print('entry='+a.entry+' sha256='+hashlib.sha256(a.output.read_bytes()).hexdigest())


if __name__=='__main__':main()
