#!/usr/bin/env python3
"""Extend the frozen MLP diagnostic host without changing its source pins."""
import argparse
import hashlib
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve existing generated host')
    repo=Path(__file__).resolve().parents[1]
    source=(repo/'src/nr_swin/mlp_probe.mm').read_text()
    replacements=[
      ('std::array<id<MTLComputePipelineState>,3>','std::array<id<MTLComputePipelineState>,4>'),
      ('@"nr_swin32_contract"]','@"nr_swin32_contract",@"nr_swin32_qkv"]'),
      ('std::array<Buffer,3> y=','std::array<Buffer,4> y='),
      ('Make(device,size_t(rows)*128*2,poison)};','Make(device,size_t(rows)*128*2,poison),Make(device,size_t(rows)*96*2,poison)};'),
      ('std::array<Buffer,3> counts=','std::array<Buffer,4> counts='),
      ('Make(device,size_t(rows)*128*4,0)};','Make(device,size_t(rows)*128*4,0),Make(device,size_t(rows)*96*4,0)};'),
      ('size_t j=0;j<size_t(rows)*128;++j','size_t j=0;j<counts[i].bytes/4;++j'),
      ('(i==2?32:128)','(i==2?32:(i==3?96:128))'),
      ('std::array<const char*,3> files=','std::array<const char*,4> files='),
      ('"activation.e4m3","partial.f16"','"activation.e4m3","partial.f16","qkv.f16"'),
      ('swin32-mlp-metal.v1','swin32-mlp-qkv-metal.v1'),
      (r'\"dispatch_count\":3',r'\"dispatch_count\":4'),
    ]
    for old,new in replacements:
        if source.count(old)!=1:raise ValueError('generation anchor changed: '+old)
        source=source.replace(old,new)
    if source.count('i<3')!=5:raise ValueError('loop count changed')
    source=source.replace('i<3','i<4')
    anchor='      [encoder dispatchThreads:'
    if source.count(anchor)!=1:raise ValueError('dispatch anchor changed')
    source=source.replace(anchor,'      if(i==3){bind(y[2],0);bind(w,1);bind(y[3],2);bind(counts[3],3);[encoder setBytes:&rows length:4 atIndex:4];}\n'+anchor)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(source)
    print('generated_host_sha256='+hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':main()
