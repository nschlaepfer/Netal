#!/usr/bin/env python3
"""Versioned Block0 candidate from the accepted C32 core; no accepted file edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    p=argparse.ArgumentParser();p.add_argument('--hmma',required=True,type=Path);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
    root=Path(__file__).resolve().parents[1];a.output.mkdir(parents=True,exist_ok=False);sources={}
    (a.output/'nr_swin').mkdir();(a.output/'nr_pre').mkdir()
    names=['qmma_fixed.metalh','swin32_mlp.metal','swin32_qkv.metal','swin32_attention.metal','swin32_block.metal']
    for name in names:
        source=root/'src/nr_swin'/name;text=source.read_text();sources[str(source)]=sha(source)
        # Block0 inserts its FP16 16->32 adapter before the ordinary FFN residual.
        text=re.sub(r'\b(8208|8288|11360|19552|19568|20592)u',lambda m:str(int(m[1])+1024)+'u',text)
        if name=='swin32_mlp.metal':
            old='kernel void nr_swin32_contract(device const uchar* input'
            assert text.count(old)==1;text=text.replace(old,'kernel void nr_swin32_contract(device const half* input')
            old='nr_e4m3(input[row*32u+nr_channel_from_dense(n)])';assert text.count(old)==1
            text=text.replace(old,'input[row*32u+n]')
        (a.output/'nr_swin'/name).write_text(text)
    for name in ['pre_features.metal','pre_boundaries.metal','pre_probe.mm','noise_math_exact.metalh']:
        source=root/'src/nr_pre'/name;sources[str(source)]=sha(source);shutil.copyfile(source,a.output/'nr_pre'/name)
    source=root/'src/nr_swin/mlp_probe.mm';sources[str(source)]=sha(source);shutil.copyfile(source,a.output/'nr_swin/mlp_probe.mm')
    sources[str(a.hmma)]=sha(a.hmma);shutil.copyfile(a.hmma,a.output/'nr_pre/hmma_exact.metalh')
    (a.output/'nr_pre/pre.metal').write_text('#include "pre_features.metal"\n#include "../nr_swin/swin32_block.metal"\n#include "hmma_exact.metalh"\n#include "pre_boundaries.metal"\n')
    (a.output/'source-manifest.json').write_text(json.dumps({'schema':'neuralstudio.nr.pre-native-source.v1','parent_sources':sources,'files':{str(x.relative_to(a.output)):sha(x) for x in sorted(a.output.rglob('*')) if x.is_file()},'status':'candidate; approximate noise, unverified adapter/core/exports'},indent=2)+'\n')
if __name__=='__main__':main()
