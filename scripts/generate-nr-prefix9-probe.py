#!/usr/bin/env python3
"""Continue the frozen prefix8 host into a separately specialized C128 library."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve source')
    repo=Path(__file__).resolve().parents[1];accepted=json.loads((repo/'evidence/specs/nr-prefix8-v1/verification.json').read_text())
    parent=repo/'build/nr_prefix8/macos-v2/source/probe.mm'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=accepted['frozen_artifact_sha256']['source/probe.mm']:raise ValueError('parent host drift')
    text=parent.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('splice drift: '+old)
        text=text.replace(old,new)
    replace('if(argc!=7)','if(argc!=8)')
    replace('OUTPUT BLOCK_COUNT','OUTPUT BLOCK_COUNT C128_LIB')
    replace('count>8','count>9')
    replace('Buffer wide_input=downsample;','Buffer wide_input=downsample,encoder128;')
    replace('b<=count;++b','b<=std::min(count,8u);++b')
    replace('if(b==8)WideDownsample(', 'if(b==8)encoder128=WideDownsample(')
    replace('neuralstudio.nr.prefix8-live-chain.v1','neuralstudio.nr.prefix9-live-chain.v1')
    extra=r'''
  if(count==9){@autoreleasepool{
    auto lib128=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[7]]] error:&error];
    if(!lib128)throw std::runtime_error("C128 library load failed");
    auto bytes=Read(fs::path(argv[3])/"block9.bin");
    if(bytes.size()!=197184||Sha(bytes.data(),bytes.size())!="934cebc771cfa1ed086f47f018426090e14ca4326dcb22843cbc17dff9851c32")throw std::runtime_error("Block9 weights");
    Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
    const auto input_before=Sha(Data(encoder128),encoder128.bytes);
    Buffer result=ExecuteWide(p.device,p.queue,lib128,encoder128,weight,240*144,poison,output/"block9",nullptr,128);
    Write(output/"block9.e4m3",result);
    receipt<<",{\"block\":9,\"input_sha256\":\""<<input_before<<"\",\"output_sha256\":\""<<Sha(Data(result),result.bytes)<<"\"}";
    std::cout<<"block=9 output_sha256="<<Sha(Data(result),result.bytes)<<"\n"<<std::flush;
  }}
'''
    replace('  receipt<<"],',extra+'  receipt<<"],')
    a.output.mkdir(parents=True);(a.output/'probe.mm').write_text(text)
    # Keep the accepted C64 shader exact; the new library owns distinct C128 instances.
    old_shader=repo/'build/nr_prefix8/macos-v2/source/prefix8.metal'
    if hashlib.sha256(old_shader.read_bytes()).hexdigest()!=accepted['frozen_artifact_sha256']['source/prefix8.metal']:raise ValueError('parent shader drift')
    (a.output/'prefix8.metal').write_bytes(old_shader.read_bytes())
    print('host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
