#!/usr/bin/env python3
"""Extend the accepted-prefix8-derived host with the complete C128 encoder stage."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve source')
    repo=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,str(repo/'scripts/generate-nr-prefix9-probe.py'),'--output',str(a.output)],check=True)
    text=(a.output/'probe.mm').read_text()
    if text.count('count>9')!=1:raise ValueError('count splice drift')
    text=text.replace('count>9','count>14').replace('neuralstudio.nr.prefix9-live-chain.v1','neuralstudio.nr.prefix14-live-chain.v1')
    start=text.index('  if(count==9){@autoreleasepool{');end=text.index('  receipt<<"],',start)
    text=text[:start]+r'''
  Buffer encoder256;
  if(count>=9){@autoreleasepool{
    auto lib128=[p.device newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[7]]] error:&error];
    if(!lib128)throw std::runtime_error("C128 library load failed");
    const std::array<Grid,6> grids128={Grid{30,18,0,0},Grid{31,19,-4,-4},Grid{31,18,-4,0},Grid{30,19,0,-4},Grid{30,18,0,0},Grid{31,19,-4,-4}};
    const std::array<const char*,6> hashes128={"934cebc771cfa1ed086f47f018426090e14ca4326dcb22843cbc17dff9851c32","77eacae2f2290c898c53b151b36bc7539bbc7b147628ef3f3b1ecc22b1f910d4","dd5440dccd5fa2be9b63421d2a67715b8c41abf783464601a78bf138f479a40f","6e74d269d3b545cbef475810dc6143d2ed4252d1b905893ed726dceb756f3962","535b8b432b76bd8fe857594a58b76985ec70775390fba55d14edfe2df6d5bdaa","6c7a05c82b823e09b0b843f5dd399af7a006410f3587191f62a43205a872724d"};
    for(uint32_t b=9;b<=count;++b){@autoreleasepool{
      auto bytes=Read(fs::path(argv[3])/("block"+std::to_string(b)+".bin"));
      if(bytes.size()!=(b==14?229936:197184)||Sha(bytes.data(),bytes.size())!=hashes128[b-9])throw std::runtime_error("C128 weight identity");
      Buffer weight=Make(p.device,bytes.size(),0);std::memcpy(Data(weight),bytes.data(),bytes.size());
      Grid grid=grids128[b-9];Buffer prepared=b==9?encoder128:WideRewindow(p,lib128,encoder128,grids128[0],grid,poison,128,240,144),projection;
      const auto input_before=Sha(Data(encoder128),encoder128.bytes);
      Buffer result=ExecuteWide(p.device,p.queue,lib128,prepared,weight,grid.width*grid.height*64,poison,output/("block"+std::to_string(b)),b==14?&projection:nullptr,128);
      if(b==14)encoder256=WideDownsample(p,lib128,projection,weight,grid,poison,output,14,128,240,144);
      encoder128=b==9?result:WideRewindow(p,lib128,result,grid,grids128[0],poison,128,240,144);
      Write(output/("block"+std::to_string(b)+".e4m3"),encoder128);
      receipt<<",{\"block\":"<<b<<",\"input_sha256\":\""<<input_before<<"\",\"output_sha256\":\""<<Sha(Data(encoder128),encoder128.bytes)<<"\"}";
      std::cout<<"block="<<b<<" output_sha256="<<Sha(Data(encoder128),encoder128.bytes)<<"\n"<<std::flush;
    }}
  }}
  if(count==14&&encoder256.bytes!=size_t(120)*72*256)throw std::runtime_error("C256 handoff extent");
''' + text[end:]
    (a.output/'probe.mm').write_text(text)
    (a.output/'wide128.metal').write_text((repo/'src/nr_swin/swin_wide.metal').read_text()+'\n'+(repo/'src/nr_swin/wide_boundaries.metal').read_text())
    print('prefix14_host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
