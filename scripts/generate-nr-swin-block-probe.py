#!/usr/bin/env python3
"""Append probabilities, values, projection and packing to the score host."""
import argparse
import hashlib
from pathlib import Path
import subprocess


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve generated host')
    repo=Path(__file__).resolve().parents[1];base=a.output.with_suffix('.attention-base.mm')
    subprocess.run(['python3',str(repo/'scripts/generate-nr-swin-attention-probe.py'),'--output',str(base)],check=True)
    source=base.read_text()
    pairs=[
      ('std::array<id<MTLComputePipelineState>,6>','std::array<id<MTLComputePipelineState>,10>'),
      ('@"nr_swin32_scores"]','@"nr_swin32_scores",@"nr_swin32_probabilities",@"nr_swin32_values",@"nr_swin32_project",@"nr_swin32_pack"]'),
      ('std::array<Buffer,6> y=','std::array<Buffer,10> y='),
      ('Make(device,size_t(rows)*64*2,poison)};',
       'Make(device,size_t(rows)*64*2,poison),Make(device,size_t(rows)*64,poison),Make(device,size_t(rows)*64*2,poison),Make(device,size_t(rows)*32*2,poison),Make(device,size_t(rows)*32,poison)};'),
      ('std::array<Buffer,6> counts=','std::array<Buffer,10> counts='),
      ('Make(device,size_t(rows)*64*4,0)};',
       'Make(device,size_t(rows)*64*4,0),Make(device,size_t(rows)*64*4,0),Make(device,size_t(rows)*64*4,0),Make(device,size_t(rows)*32*4,0),Make(device,size_t(rows)*32*4,0)};'),
      ('(i==2?32:(i==3?96:(i==4?2:(i==5?64:128))))','dispatch_width[i]'),
      ('std::array<const char*,6> files=','std::array<const char*,10> files='),
      ('"normalized.e4m3","scores.f16"','"normalized.e4m3","scores.f16","probabilities.e4m3","value_partial.f16","projection.f16","output.e4m3"'),
      ('swin32-attention-scores-metal.v1','swin32-complete-block-metal.v1'),
      (r'\"dispatch_count\":6',r'\"dispatch_count\":10'),
      ('rows>65536','rows>1048576'),
    ]
    for old,new in pairs:
        if source.count(old)!=1:raise ValueError('generation anchor changed: '+old)
        source=source.replace(old,new)
    if source.count('i<6')!=5:raise ValueError('loop count changed')
    source=source.replace('i<6','i<10')
    anchor='    id<MTLCommandBuffer> command=[queue commandBuffer];'
    source=source.replace(anchor,'    const unsigned dispatch_width[10]={128,128,32,96,2,64,1,32,32,32};\n'+anchor)
    anchor='      [encoder dispatchThreads:'
    source=source.replace(anchor,
      '      if(i==6){bind(y[5],0);bind(y[6],1);bind(counts[6],2);[encoder setBytes:&rows length:4 atIndex:3];}\n'
      '      if(i==7){bind(y[6],0);bind(y[3],1);bind(y[7],2);bind(counts[7],3);[encoder setBytes:&rows length:4 atIndex:4];}\n'
      '      if(i==8){bind(y[7],0);bind(y[2],1);bind(w,2);bind(y[8],3);bind(counts[8],4);[encoder setBytes:&rows length:4 atIndex:5];}\n'
      '      if(i==9){bind(y[8],0);bind(y[9],1);bind(counts[9],2);[encoder setBytes:&rows length:4 atIndex:3];}\n'+anchor)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(source)
    print('generated_host_sha256='+hashlib.sha256(source.encode()).hexdigest())


if __name__=='__main__':main()
