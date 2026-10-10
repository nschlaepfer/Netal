#!/usr/bin/env python3
"""Extend a pinned three-block host without changing its sealed source."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():raise ValueError('preserve generated host')
    repo=Path(__file__).resolve().parents[1]
    receipt=json.loads((repo/'evidence/specs/nr-swin32-chain-v1/verification.json').read_text())
    parent=repo/'build/nr_swin_chain/macos-v2/source/src/nr_swin/chain_probe.mm'
    if hashlib.sha256(parent.read_bytes()).hexdigest()!=receipt['source_sha256']['src/nr_swin/chain_probe.mm']:raise ValueError('parent source drift')
    text=parent.read_text()
    def replace(old,new,count=1):
        nonlocal text
        if text.count(old)!=count:raise ValueError('host splice drift: '+old)
        text=text.replace(old,new)
    replace('>,11> states','>,13> states')
    replace('const fs::path& output) {','const fs::path& output,Buffer* projection=nullptr) {')
    replace('  return y[9];','  if(projection)*projection=y[8];\n  return y[9];')
    replace('count>3','count>4')
    replace('std::array<const char*,3> hashes','std::array<const char*,4> hashes')
    replace('"8f26be9b75d09960141bbf561a70e9628c269cc901934efdce362426ebbe1589"};',
            '"8f26be9b75d09960141bbf561a70e9628c269cc901934efdce362426ebbe1589","f0d05533694577fa313f2655b00ada84f231f87ebc66ebbb7e8f0f7ea1ac88d2"};')
    replace('std::array<Grid,3> grids','std::array<Grid,4> grids')
    replace('Grid{121,72,-4,0}}','Grid{121,72,-4,0},Grid{120,73,0,-4}}')
    replace('bytes.size()!=20672','bytes.size()!=(i==3?22720:20672)')
    replace('@"nr_swin32_rewindow"]','@"nr_swin32_rewindow",@"nr_swin32_downsample_pool",@"nr_swin32_downsample_project"]')
    replace('i<11;++i','i<13;++i')
    replace('Buffer result=Block(', 'Buffer projection;Buffer result=Block(')
    replace('std::to_string(i+1)));','std::to_string(i+1)),i==3?&projection:nullptr);')
    replace('    input=i==0?result:', '    if(i==3)Downsample(p,projection,weight,poison,output);\n    input=i==0?result:')
    replace('neuralstudio.nr.swin-c32-live-chain.v1','neuralstudio.nr.swin-c32-downsample-live-chain.v1')
    function=r'''
void Downsample(Pipelines& p,Buffer& projected,Buffer& weight,uint8_t poison,const fs::path& output) {
  constexpr size_t rows=480*288;
  Buffer pooled=Make(p.device,rows*32,poison),result=Make(p.device,rows*64,poison);
  Buffer pc=Make(p.device,rows*32*4,0),rc=Make(p.device,rows*64*4,0);
  const auto before=Sha(Data(projected),projected.bytes),weight_before=Sha(Data(weight),weight.bytes);
  auto command=[p.queue commandBuffer];
  for(unsigned i=0;i<2;++i){
    auto encoder=[command computeCommandEncoder];[encoder setComputePipelineState:p.states[11+i]];
    auto bind=[&](Buffer& b,NSUInteger j){[encoder setBuffer:b.metal offset:Guard atIndex:j];};
    if(i==0){bind(projected,0);bind(pooled,1);bind(pc,2);}
    else{bind(pooled,0);bind(weight,1);bind(result,2);bind(rc,3);}
    [encoder dispatchThreads:MTLSizeMake(rows*(i?64:32),1,1) threadsPerThreadgroup:MTLSizeMake(64,1,1)];[encoder endEncoding];
  }
  Complete(command);GuardOutput(pooled,pc);GuardOutput(result,rc);
  if(before!=Sha(Data(projected),projected.bytes)||weight_before!=Sha(Data(weight),weight.bytes)||Canary(projected)||Canary(weight))throw std::runtime_error("downsample mutated source");
  Write(output/"block4-pooled.e4m3",pooled);Write(output/"block4-downsample.e4m3",result);
  std::ofstream receipt(output/"downsample.json");
  receipt<<"{\"schema\":\"neuralstudio.nr.swin-c32-downsample-metal.v1\",\"pid\":"<<getpid()<<",\"poison\":"<<unsigned(poison)
    <<",\"projected_input_sha256\":\""<<before<<"\",\"weight_sha256\":\""<<weight_before
    <<"\",\"pooled_sha256\":\""<<Sha(Data(pooled),pooled.bytes)<<"\",\"output_sha256\":\""<<Sha(Data(result),result.bytes)
    <<"\",\"live_projection_input\":true,\"expected_input_opened\":false,\"coverage_errors\":0,\"canary_errors\":0,\"immutable_errors\":0,\"result\":\"pass\"}\n";
  std::cout<<"downsample_sha256="<<Sha(Data(result),result.bytes)<<"\n"<<std::flush;
}

'''
    replace('int main(int argc,const char** argv)',function+'int main(int argc,const char** argv)')
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(text)
    print('host_sha256='+hashlib.sha256(text.encode()).hexdigest())


if __name__=='__main__':main()
