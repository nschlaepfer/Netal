#!/usr/bin/env python3
"""Generate the selected dynamic L54-L80 RTX custom-core bridge."""

from __future__ import annotations

import argparse
from pathlib import Path


def once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


def section(source: str, start: str, end: str, replacement: str,
            label: str) -> str:
    first = source.find(start)
    last = source.find(end, first)
    if first < 0 or last < 0:
        raise RuntimeError(f"{label}: anchor missing")
    last += len(end)
    return source[:first] + replacement + source[last:]


EXECUTION = r'''CUdeviceptr p_l56=l56.payload(),pa=a.payload(),pb=b.payload(),pc=scalar.payload();
std::uint32_t h=288,w=512;float factor=.5f;
void*initArgs[]={&p_l56,&pa,&pb,&pc,&h,&w,&factor};
Check(cuLaunchKernel(fn[0],32,18,1,16,16,1,0,nullptr,initArgs,nullptr),"l54");
auto conv=[&](CUfunction f,int mi,GuardedDeviceBuffer&in,GuardedDeviceBuffer&out,
              std::uint32_t oc,std::uint32_t ih,std::uint32_t iw,
              std::uint32_t ic,std::uint32_t oh,std::uint32_t ow,
              unsigned gx,unsigned gy,unsigned bx,unsigned by,unsigned bz){
  ConvParams p{wb[mi].payload(),bb[mi].payload(),oc,3,3,ic,in.payload(),1,
               ih,iw,ic,out.payload(),1,oh,ow,oc};
  Packed(f,p,gx,gy,1,bx,by,bz);
};
conv(fn[1],0,l56,l57,32,288,512,10,144,256,64,72,32,2,4);
conv(fn[2],1,l57,l58,32,144,256,32,72,128,32,36,32,2,4);
conv(fn[3],2,l58,l59,64,72,128,32,36,64,16,36,32,1,8);
std::array<GuardedDeviceBuffer*,8>ci={&l59,&l60,&l61,&l62,&l63,&l64,&l65,&l66},
  cr={&l59,&l59,&l61,&l61,&l63,&l63,&l65,&l65},
  co={&l60,&l61,&l62,&l63,&l64,&l65,&l66,&l67};
for(int i=0;i<8;++i){
  CUdeviceptr wp=wb[3+i].payload(),bp=bb[3+i].payload(),ip=ci[i]->payload(),
              rp=cr[i]->payload(),op=co[i]->payload();
  std::uint32_t oc=64,kh=3,kw=3,ic=64,n=1,hh=36,ww=64,ss=64,flag=i&1;
  void*args[]={&wp,&bp,&oc,&kh,&kw,&ic,&ip,&rp,&n,&hh,&ww,&ss,
               &op,&n,&hh,&ww,&ss,&flag};
  Check(cuLaunchKernel(fn[4],8,18,1,32,1,8,0,nullptr,args,nullptr),"conv2");
}
Bot0Params p68{wb[11].payload(),bb[11].payload(),32,3,3,64,l67.payload(),
  1,36,64,64,l68a.payload(),l68b.payload(),l68c.payload(),1,72,128,32};
Packed(fn[5],p68,16,18,3,32,2,4);
Bot1Params p69{wb[12].payload(),bb[12].payload(),8,3,3,32,
  l68a.payload(),l68b.payload(),l68c.payload(),1,72,128,32,
  l69a.payload(),1,144,256,4,l69b.payload(),1,144,256,1,
  l69c.payload(),1,144,256,3};
Packed(fn[6],p69,32,18,3,32,4,1);
UpsampleParams p70{{{l69a.payload(),1,144,256,4},
                     {l69b.payload(),1,144,256,1},
                     {l69c.payload(),1,144,256,3}},
                    {{l70a.payload(),1,288,512,4},
                     {l70b.payload(),1,288,512,1},
                     {l70c.payload(),1,288,512,3}}};
Packed(fn[7],p70,32,18,3,16,16,1);
CUdeviceptr po=l71.payload(),pu0=l70a.payload(),pu1=l70b.payload(),pu2=l70c.payload();
void*centralArgs[]={&po,&pa,&pb,&pc,&pu0,&pu1,&pu2,&w,&h,&factor};
Check(cuLaunchKernel(fn[8],32,18,1,16,16,1,0,nullptr,centralArgs,nullptr),"l69");
conv(fn[9],13,l71,l72,16,288,512,18,144,256,64,72,32,2,2);
conv(fn[10],14,l72,l73,32,144,256,16,72,128,32,36,32,2,4);
std::array<GuardedDeviceBuffer*,8>b1ci={&l73,&l74,&l75,&l76,&l77,&l78,&l79,&l80},
  b1cr={&l73,&l73,&l75,&l75,&l77,&l77,&l79,&l79},
  b1co={&l74,&l75,&l76,&l77,&l78,&l79,&l80,&l81};
for(int i=0;i<8;++i){
  CUdeviceptr wp=wb[15+i].payload(),bp=bb[15+i].payload(),ip=b1ci[i]->payload(),
              rp=b1cr[i]->payload(),op=b1co[i]->payload();
  std::uint32_t oc=32,kh=3,kw=3,ic=32,n=1,hh=72,ww=128,ss=32,flag=i&1;
  void*args[]={&wp,&bp,&oc,&kh,&kw,&ic,&ip,&rp,&n,&hh,&ww,&ss,
               &op,&n,&hh,&ww,&ss,&flag};
  Check(cuLaunchKernel(fn[11],16,18,1,32,2,4,0,nullptr,args,nullptr),
        "block1 conv2");
}
Bot0Params p82{wb[23].payload(),bb[23].payload(),16,3,3,32,l81.payload(),
  1,72,128,32,l82a.payload(),l82b.payload(),l82c.payload(),1,144,256,16};
Packed(fn[12],p82,32,36,1,32,2,6);
FullBot1Params p83{wb[24].payload(),bb[24].payload(),8,3,3,16,
  l82a.payload(),l82b.payload(),l82c.payload(),l70a.payload(),l70b.payload(),
  l70c.payload(),1,144,256,16,static_cast<CUdeviceptr>(l81_output_gpu_va),
  1,288,512,4,1,288,512,1,1,288,512,3};
Packed(fn[13],p83,32,36,3,32,8,1);Check(cuCtxSynchronize(),"sync");'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "core_bridge.h"',
        '#include "core_bridge.h"\n'
        '#include "../fg_connected_subrect_l3_l83/contract.h"\n'
        '#include "../fg_connected_subrect_l3_l83/rtx_subrect_bridges.h"',
        "subrect includes",
    )
    source = once(source, "namespace{namespace c=fg_connected_custom;",
                  "namespace{namespace c=fg_connected_custom;"
                  "namespace sc=fg_connected_subrect_l3_l83;",
                  "namespace")
    source = source.replace("RunFgConnectedCoreL56L83",
                            "RunFgSubrectCoreL54L80Live")
    source = source.replace("FgConnectedCoreBridgeResult", "FgSubrectCoreResult")
    old_signature = (
        "int RunFgSubrectCoreL54L80Live(const char*module_directory,"
        "const char*model_directory,std::uint8_t poison,"
        "std::uint64_t l83_output_gpu_va,FgSubrectCoreResult*result)"
    )
    new_signature = (
        "int RunFgSubrectCoreL54L80Live(const char*module_directory,"
        "const char*model_directory,std::uint8_t poison,"
        "std::uint64_t input_a_cuda_ptr,std::uint64_t input_b_cuda_ptr,"
        "std::uint64_t input_c_cuda_ptr,std::uint64_t l81_output_gpu_va,"
        "FgSubrectCoreResult*result)"
    )
    source = once(source, old_signature, new_signature, "function signature")
    source = once(
        source,
        'if(!result||!l83_output_gpu_va)throw std::runtime_error("bridge argument");',
        'if(!result||!input_a_cuda_ptr||!input_b_cuda_ptr||!input_c_cuda_ptr||'
        '!l81_output_gpu_va)throw std::runtime_error("bridge argument");',
        "argument gate",
    )
    source = once(
        source,
        "auto ext0=c::Generate();",
        "c::ExternalInputs ext0{std::vector<std::uint8_t>(sc::kInputABytes),"
        "std::vector<std::uint8_t>(sc::kInputBBytes),"
        "std::vector<std::uint8_t>(sc::kInputCBytes)};",
        "live input snapshot",
    )
    for name in (
        "kInputABytes", "kInputBBytes", "kInputCBytes", "kL56Bytes",
        "kL57Bytes", "kL58Bytes", "kLow64Bytes", "kL68PlaneBytes",
        "kL69Out0Bytes", "kL69Out1Bytes", "kL69Out2Bytes",
        "kL70Out0Bytes", "kL70Out1Bytes", "kL70Out2Bytes", "kL71Bytes",
        "kL72Bytes", "kBlock1Bytes", "kL82PlaneBytes",
    ):
        source = source.replace(f"c::{name}", f"sc::{name}")
    source = source.replace("ssc::", "sc::")
    source = once(
        source,
        'Check(cuCtxSetCurrent(ctx),"set");',
        'Check(cuCtxSetCurrent(ctx),"set");'
        'Check(cuMemcpyDtoH(ext0.a.data(),static_cast<CUdeviceptr>(input_a_cuda_ptr),ext0.a.size()),"snapshot a");'
        'Check(cuMemcpyDtoH(ext0.b.data(),static_cast<CUdeviceptr>(input_b_cuda_ptr),ext0.b.size()),"snapshot b");'
        'Check(cuMemcpyDtoH(ext0.c.data(),static_cast<CUdeviceptr>(input_c_cuda_ptr),ext0.c.size()),"snapshot c");',
        "snapshot after context",
    )
    source = once(
        source,
        'a.Allocate(sc::kInputABytes);b.Allocate(sc::kInputBBytes);scalar.Allocate(sc::kInputCBytes);Check(cuMemcpyHtoD(a.payload(),ext0.a.data(),ext0.a.size()),"a");Check(cuMemcpyHtoD(b.payload(),ext0.b.data(),ext0.b.size()),"b");Check(cuMemcpyHtoD(scalar.payload(),ext0.c.data(),ext0.c.size()),"c");',
        'a.Allocate(sc::kInputABytes);b.Allocate(sc::kInputBBytes);scalar.Allocate(sc::kInputCBytes);'
        'Check(cuMemcpyDtoD(a.payload(),static_cast<CUdeviceptr>(input_a_cuda_ptr),sc::kInputABytes),"live a");'
        'Check(cuMemcpyDtoD(b.payload(),static_cast<CUdeviceptr>(input_b_cuda_ptr),sc::kInputBBytes),"live b");'
        'Check(cuMemcpyDtoD(scalar.payload(),static_cast<CUdeviceptr>(input_c_cuda_ptr),sc::kInputCBytes),"live c");',
        "live device copies",
    )
    source = section(source, "CUdeviceptr p_l56=", 'Check(cuCtxSynchronize(),"sync");',
                     EXECUTION, "dynamic core execution")
    # Checkpoint/model bookkeeping keeps the original 27-output shape.
    source = source.replace("l83_output_gpu_va", "l81_output_gpu_va")
    return "// Generated exact selected subrect RTX custom core.\n" + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f"preserve existing output: {args.output}")
    result = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f"generated={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
