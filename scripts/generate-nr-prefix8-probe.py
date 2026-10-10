#!/usr/bin/env python3
"""Reuse the sealed prefix and extend its live C64 chain through Block 8."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve source')
    repo=Path(__file__).resolve().parents[1];parent=repo/'build/nr_prefix5/macos-v1/source/generated/probe.mm'
    accepted=json.loads((repo/'evidence/specs/nr-prefix5-v1/verification.json').read_text())
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=accepted['generated_host_sha256']:raise ValueError('parent host drift')
    text=parent.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('splice drift '+old)
        text=text.replace(old,new)
    replace('#include "wide_execute.h"','#include "wide_execute.h"\n#include "wide_boundaries.h"')
    replace('count>5','count>8');replace('neuralstudio.nr.prefix5-live-chain.v1','neuralstudio.nr.prefix8-live-chain.v1')
    start=text.index('  if(count==5){@autoreleasepool{');end=text.index('  receipt<<"],',start)
    text=text[:start]+r'''
  const std::array<Grid,4> wide_grids={Grid{60,36,0,0},Grid{61,37,-4,-4},Grid{61,36,-4,0},Grid{60,37,0,-4}};
  const std::array<const char*,4> wide_hashes={"1b905577285978107b21b5730aac2b0105d45f757b5e78334291d9b4ab5646bb","380e368298920277d3bf9bf715730f2b5fdffe763fd1a3ed67ab9c3b0be1d682","cd990fde2971a8550e3bf0010842bc02fa3a07f4497f5b5b71dd7d264a3e3384","687d906bc5d6748b57ee3e519c114f9382d93db560b7f779461c1ab90f5237ae"};
  Buffer wide_input=downsample;
  for(uint32_t b=5;b<=count;++b){@autoreleasepool{
    auto bytes=Read(fs::path(argv[3])/("block"+std::to_string(b)+".bin"));
    if(bytes.size()!=(b==8?69936:61760)||Sha(bytes.data(),bytes.size())!=wide_hashes[b-5])throw std::runtime_error("wide weight identity");
    Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
    Grid grid=wide_grids[b-5];Buffer prepared=b==5?wide_input:WideRewindow(p,lib,wide_input,wide_grids[0],grid,poison),projection;
    Buffer result=ExecuteWide(p.device,p.queue,lib,prepared,weight,grid.width*grid.height*64,poison,output/("block"+std::to_string(b)),b==8?&projection:nullptr);
    if(b==8)WideDownsample(p,lib,projection,weight,grid,poison,output);
    wide_input=b==5?result:WideRewindow(p,lib,result,grid,wide_grids[0],poison);
    Write(output/("block"+std::to_string(b)+".e4m3"),wide_input);
    receipt<<",{\"block\":"<<b<<",\"output_sha256\":\""<<Sha(Data(wide_input),wide_input.bytes)<<"\"}";
    std::cout<<"block="<<b<<" output_sha256="<<Sha(Data(wide_input),wide_input.bytes)<<"\n"<<std::flush;
  }}
''' +text[end:]
    a.output.mkdir(parents=True);(a.output/'probe.mm').write_text(text)
    wide=(repo/'src/nr_swin/swin_wide.metal').read_text().replace('#include "swin32_block.metal"','#include "swin32_downsample.metal"',1)
    (a.output/'prefix8.metal').write_text(wide+'\n'+(repo/'src/nr_swin/wide_boundaries.metal').read_text())
    print('host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
