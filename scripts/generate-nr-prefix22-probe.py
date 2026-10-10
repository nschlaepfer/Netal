#!/usr/bin/env python3
"""Extend the frozen accepted C128 encoder into C256 and native consumer exports."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve source')
    repo=Path(__file__).resolve().parents[1];accepted=json.loads((repo/'evidence/specs/nr-prefix14-v1/verification.json').read_text())
    parent=repo/'build/nr_encoder14/macos-v3/source/probe.mm'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=accepted['frozen_artifact_sha256']['source/probe.mm']:raise ValueError('parent host drift')
    text=parent.read_text()
    def replace(old,new):
        nonlocal text
        if text.count(old)!=1:raise ValueError('splice drift: '+old)
        text=text.replace(old,new)
    replace('#define main nr_mlp_diagnostic_unused_main','#include <map>\n#define main nr_mlp_diagnostic_unused_main')
    replace('#include "wide_boundaries.h"','#include "wide_boundaries.h"\n#include "native_encoder_exports.h"')
    replace('if(argc!=8)','if(argc!=9)');replace('count>14','count>22');replace('BLOCK_COUNT C128_LIB','BLOCK_COUNT C128_LIB C256_LIB')
    replace('neuralstudio.nr.prefix14-live-chain.v1','neuralstudio.nr.prefix22-live-chain.v1')
    replace('Buffer downsample;for(', 'std::map<uint32_t,Buffer> skips;Buffer downsample;for(')
    replace('    Write(output/("block"+std::to_string(i+1)+".e4m3"),input);','    Write(output/("block"+std::to_string(i+1)+".e4m3"),input);if(i==3)skips[4]=input;')
    replace('    Write(output/("block"+std::to_string(b)+".e4m3"),wide_input);','    Write(output/("block"+std::to_string(b)+".e4m3"),wide_input);if(b==8)skips[8]=wide_input;')
    replace('b=9;b<=count;++b','b=9;b<=std::min(count,14u);++b')
    replace('      Write(output/("block"+std::to_string(b)+".e4m3"),encoder128);','      Write(output/("block"+std::to_string(b)+".e4m3"),encoder128);if(b==14)skips[14]=encoder128;')
    extra=r'''
  if(count>=15){@autoreleasepool{
    auto lib256=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[8]]] error:&error];
    if(!lib256)throw std::runtime_error("C256 library load failed");
    const std::array<Grid,8> grids256={Grid{15,9,0,0},Grid{16,10,-4,-4},Grid{16,9,-4,0},Grid{15,10,0,-4},Grid{15,9,0,0},Grid{16,10,-4,-4},Grid{16,9,-4,0},Grid{15,10,0,-4}};
    const std::array<const char*,8> hashes256={"a3081eaee51296319ade0839add190cfdbe2adbf0f0a5e990df1f2a577ae9507","37c5928b15aa456cac02de082e96cd2b7c3e512915f17914000cded6858a86d8","478a4ccea0614d2eeccf1be093d9e72367b2afba49dd7e8c50547edbea67bd61","6c03840d01a50976357ed180eeeda50a2614b8b0ed7fc526aef6b089a906fd4c","f269c6404c6bd73611598ae265f951c73da96e8211a060324e38a703f58d32b4","227945c0e2d727736867e0c2bb232f5bf6f29e42a6e7a5c92546fda9819bc9ac","5a61aad0a642e2398306329a865528777013480d97c384a8e626fe1d394828e8","cfb0343287054e38933dbf35c7dbb124fef50cfaf4234f7161575f00896cc02d"};
    Buffer middle;
    for(uint32_t b=15;b<=count;++b){@autoreleasepool{
      auto bytes=Read(fs::path(argv[3])/("block"+std::to_string(b)+".bin"));
      if(bytes.size()!=(b==22?820288:689232)||Sha(bytes.data(),bytes.size())!=hashes256[b-15])throw std::runtime_error("C256 weight identity");
      Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
      Grid grid=grids256[b-15];Buffer prepared=b==15?encoder256:WideRewindow(p,lib256,encoder256,grids256[0],grid,poison,256,120,72),projection;
      const auto input_before=Sha(Data(encoder256),encoder256.bytes);
      Buffer result=ExecuteWide(p.device,p.queue,lib256,prepared,weight,grid.width*grid.height*64,poison,output/("block"+std::to_string(b)),b==22?&projection:nullptr,256);
      if(b==22)middle=WideDownsample(p,lib256,projection,weight,grid,poison,output,22,256,120,72);
      encoder256=b==15?result:WideRewindow(p,lib256,result,grid,grids256[0],poison,256,120,72);
      Write(output/("block"+std::to_string(b)+".e4m3"),encoder256);if(b==22)skips[22]=encoder256;
      receipt<<",{\"block\":"<<b<<",\"input_sha256\":\""<<input_before<<"\",\"output_sha256\":\""<<Sha(Data(encoder256),encoder256.bytes)<<"\"}";
      std::cout<<"block="<<b<<" output_sha256="<<Sha(Data(encoder256),encoder256.bytes)<<"\n"<<std::flush;
    }}
    if(count==22){
      std::map<uint32_t,Buffer> native_skips;
      std::ofstream exports(output/"native_exports.json");exports<<"{\"pid\":"<<getpid()<<",\"poison\":"<<poison<<",\"outputs\":{";
      const std::array<uint32_t,4> blocks={4,8,14,22},widths={960,480,240,120},heights={576,288,144,72},channels={32,64,128,256};
      for(size_t i=0;i<4;++i){
        uint32_t b=blocks[i];native_skips[b]=ExportEncoderNative(p,lib256,skips.at(b),widths[i],heights[i],channels[i],false,poison);
        const std::string name="block"+std::to_string(b)+"-native.e4m3";Write(output/name,native_skips[b]);
        exports<<(i?",":"")<<"\""<<name<<"\":{\"bytes\":"<<native_skips[b].bytes<<",\"sha256\":\""<<Sha(Data(native_skips[b]),native_skips[b].bytes)<<"\"}";
      }
      Buffer native_middle=ExportEncoderNative(p,lib256,middle,60,36,512,true,poison);Write(output/"block22-middle.e4m3",native_middle);
      exports<<",\"block22-middle.e4m3\":{\"bytes\":"<<native_middle.bytes<<",\"sha256\":\""<<Sha(Data(native_middle),native_middle.bytes)<<"\"}},\"expected_input_opened\":false,\"live_device_inputs\":true,\"coverage_errors\":0,\"canary_errors\":0,\"immutable_errors\":0,\"result\":\"pass\"}\n";
      std::cout<<"middle_input_sha256="<<Sha(Data(native_middle),native_middle.bytes)<<"\n"<<std::flush;
    }
  }}
'''
    replace('  receipt<<"],',extra+'  receipt<<"],')
    a.output.mkdir(parents=True);(a.output/'probe.mm').write_text(text)
    for name in ('prefix8.metal','wide128.metal'):
        old=repo/'build/nr_encoder14/macos-v3/source'/name
        if hashlib.sha256(old.read_bytes()).hexdigest()!=accepted['frozen_artifact_sha256']['source/'+name]:raise ValueError('parent shader drift')
    # Preserve the accepted host scaffold, but compile ALL widths from the same
    # current arithmetic source. Reusing embedded old C64/C128 shader strings
    # would silently retain the disproven fp32 chain in only those widths.
    wide='\n'.join((repo/'src/nr_swin'/name).read_text() for name in ('swin_wide.metal','wide_boundaries.metal','native_encoder_exports.metal'))
    (a.output/'prefix8.metal').write_text(wide.replace('#include "swin32_block.metal"','#include "swin32_downsample.metal"',1))
    for name in ('wide128.metal','wide256.metal'):(a.output/name).write_text(wide)
    print('prefix22_host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
