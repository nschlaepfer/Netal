#!/usr/bin/env python3
"""Connect the accepted downsample's live buffer to the first C64 block."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve source')
    repo=Path(__file__).resolve().parents[1]
    parent=repo/'build/nr_swin_downsample/macos-v3/source/src/nr_swin/probe.mm'
    accepted=json.loads((repo/'evidence/specs/nr-swin32-downsample-v1/verification.json').read_text())
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=accepted['generated_host_sha256']:raise ValueError('parent source drift')
    text=parent.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('splice drift: '+old)
        text=text.replace(old,new)
    replace('void Downsample(', 'Buffer Downsample(')
    replace('  std::cout<<"downsample_sha256="<<Sha(Data(result),result.bytes)<<"\\n"<<std::flush;',
            '  std::cout<<"downsample_sha256="<<Sha(Data(result),result.bytes)<<"\\n"<<std::flush;return result;')
    replace('int main(int argc,const char** argv)', '#include "wide_execute.h"\n\nint main(int argc,const char** argv)')
    replace('count>4','count>5')
    replace('for(uint32_t i=0;i<count;++i)', 'Buffer downsample;for(uint32_t i=0;i<std::min(count,4u);++i)')
    replace('if(i==3)Downsample(', 'if(i==3)downsample=Downsample(')
    replace('neuralstudio.nr.swin-c32-downsample-live-chain.v1','neuralstudio.nr.prefix5-live-chain.v1')
    call=r'''
  if(count==5){@autoreleasepool{
    auto bytes=Read(fs::path(argv[3])/"block5.bin");
    if(bytes.size()!=61760||Sha(bytes.data(),bytes.size())!="1b905577285978107b21b5730aac2b0105d45f757b5e78334291d9b4ab5646bb")throw std::runtime_error("Block 5 weight identity");
    Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
    Buffer result=ExecuteWide(p.device,p.queue,lib,downsample,weight,138240,poison,output/"block5");
    Write(output/"block5.e4m3",result);
    receipt<<",{\"block\":5,\"output_sha256\":\""<<Sha(Data(result),result.bytes)<<"\"}";
    std::cout<<"block=5 output_sha256="<<Sha(Data(result),result.bytes)<<"\n"<<std::flush;
  }}
'''
    replace('  receipt<<"],',call+'  receipt<<"],')
    a.output.mkdir(parents=True);(a.output/'probe.mm').write_text(text)
    wide=(repo/'src/nr_swin/swin_wide.metal').read_text()
    if not wide.startswith('#include "swin32_block.metal"'):raise ValueError('shader closure drift')
    (a.output/'prefix5.metal').write_text(wide.replace('#include "swin32_block.metal"','#include "swin32_downsample.metal"',1))
    print('host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
