#!/usr/bin/env python3
"""Generate the selected dynamic L18-L49 RTX prefix bridge."""

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


INJECTION = r'''
#include "rtx_subrect_bridges.h"
#include "contract.h"

namespace {
constexpr std::array<TensorSpec,32> kSubrectTensors={{
    {"l18",18,128,96,32},{"l19",19,64,48,32},
    {"l20",20,64,48,64},{"l21",21,32,24,64},
    {"l22",22,32,24,128},{"l23",23,16,12,128},
    {"l24",24,16,12,128},{"l25",25,8,6,128},
    {"l26",26,8,6,256},{"l27",27,4,3,256},
    {"l28",28,4,3,512},{"l29",29,4,3,256},
    {"l30",30,8,6,256},{"l31",31,8,6,256},
    {"l32",32,8,6,256},{"l33",33,8,6,128},
    {"l34",34,16,12,128},{"l35",35,16,12,128},
    {"l36",36,16,12,128},{"l37",37,16,12,128},
    {"l38",38,32,24,128},{"l39",39,32,24,128},
    {"l40",40,32,24,128},{"l41",41,32,24,64},
    {"l42",42,64,48,64},{"l43",43,64,48,64},
    {"l44",44,64,48,64},{"l45",45,64,48,32},
    {"l46",46,128,96,32},{"l47",47,128,96,32},
    {"l48",48,128,96,32},{"l49",49,128,96,16},
}};

std::vector<std::uint8_t> GenerateSubrectEncoderInput(){
  constexpr std::array<std::uint16_t,13> values={
      0x0000,0x8000,0x2000,0xa000,0x2400,0xa400,0x2800,
      0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400};
  std::vector<std::uint8_t> input(
      fg_connected_subrect_l3_l83::kModelInputBytes);
  for(std::size_t index=0;index<input.size()/2u;++index){
    const auto mixed=ChainSplitMix64(
        0x243f6a8885a308d3ull+index*0x9e3779b97f4a7c15ull);
    StoreHalf(&input,index*2u,values[mixed%values.size()]);
  }
  return input;
}
}  // namespace
'''


EXECUTION = r'''auto l20=fg_conv_3x3::FindLayer("l20");
    auto l22=fg_conv_1x1::FindLayer("l22");
    auto l24=fg_conv_3x3::FindLayer("l24");
    auto l26=fg_conv_3x3::FindLayer("l26");
    auto l28=fg_conv_3x3::FindLayer("l28");
    auto l30=fg_conv_1x1::FindLayer("l30");
    auto l31=fg_conv_1x1::FindLayer("l31");
    auto l34=fg_conv_1x1::FindLayer("l34");
    auto l35=fg_conv_1x1::FindLayer("l35");
    auto l38=fg_conv_1x1::FindLayer("l38");
    auto l39=fg_conv_1x1::FindLayer("l39");
    auto l42=fg_conv_1x1::FindLayer("l42");
    auto l43=fg_conv_1x1::FindLayer("l43");
    auto l46=fg_conv_1x1::FindLayer("l46");
    auto l47=fg_conv_1x1::FindLayer("l47");
    auto l50=fg_conv_1x1::FindLayer("l50");
    auto l51=fg_conv_1x1::FindLayer("l51");
    l20.width=128;l20.grid={8,8,1};
    l22.width=64;l22.grid={4,24,1};
    l24.width=32;l24.grid={2,24,1};
    l26.width=16;l26.grid={1,6,1};
    l28.width=8;l28.grid={1,6,2};
    l30.width=4;l30.grid={1,1,32};
    l31.width=4;l31.grid={1,3,16};
    l34.width=8;l34.grid={1,2,16};
    l35.width=8;l35.grid={1,2,8};
    l38.width=16;l38.grid={1,3,8};
    l39.width=16;l39.grid={1,3,8};
    l42.width=32;l42.grid={2,24,1};
    l43.width=32;l43.grid={2,24,2};
    l46.width=64;l46.grid={4,24,1};
    l47.width=64;l47.grid={4,12,1};
    l50.width=128;l50.grid={8,24,1};
    l51.width=128;l51.grid={8,24,1};

    Check(cuInit(0), "cuInit prefix");
    CUdevice device=0;Check(cuDeviceGet(&device,0),"cuDeviceGet prefix");
    CUcontext context=nullptr;
    Check(cuDevicePrimaryCtxRetain(&context,device),"retain prefix context");
    Check(cuCtxSetCurrent(context),"set prefix context");
    std::array<LoadedKernel,53> kernels{};
    for(const auto& module:kModules){
      std::ostringstream name;name<<"cu-module-"<<std::setw(6)
        <<std::setfill('0')<<module.token<<".bin";
      kernels[module.token]=LoadKernel(module_root/name.str(),module.bytes,
                                       module.sha256,module.function);
    }

    std::vector<std::vector<std::uint8_t>> weights_before;
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> weights;
    for(const auto& weight:kWeights){
      std::ostringstream name;name<<"default-token-"<<std::setw(6)
        <<std::setfill('0')<<weight.token<<".bin";
      auto bytes=Read(model_root/name.str());
      if(bytes.size()!=weight.bytes||Sha256(bytes)!=weight.sha256)
        throw std::runtime_error("prefix model-state identity mismatch");
      auto buffer=Allocate(bytes.size());
      Check(cuMemcpyHtoD(buffer->payload(),bytes.data(),bytes.size()),
            "copy prefix model state");
      weights_before.push_back(std::move(bytes));weights.push_back(std::move(buffer));
    }
    auto input=Allocate(fg_connected_subrect_l3_l83::kModelInputBytes);
    Check(cuMemcpyHtoD(input->payload(),input_before.data(),input_before.size()),
          "copy prefix input");
    std::vector<std::unique_ptr<GuardedDeviceBuffer>> tensors;
    for(const auto& spec:kSubrectTensors){
      auto tensor=Allocate(TensorBytes(spec));
      Check(cuMemsetD8(tensor->payload(),poison,TensorBytes(spec)),
            "poison prefix tensor");
      tensors.push_back(std::move(tensor));
    }

    Launch3x3(l20,kernels[28],*weights[0],*input,*tensors[0]);
    LaunchPool(kernels[29],*tensors[0],*tensors[1],128,96,32);
    Launch1x1(l22,kernels[30],*weights[1],*tensors[1],*tensors[2]);
    LaunchPool(kernels[29],*tensors[2],*tensors[3],64,48,64);
    Launch3x3(l24,kernels[31],*weights[2],*tensors[3],*tensors[4]);
    LaunchPool(kernels[29],*tensors[4],*tensors[5],32,24,128);
    Launch3x3(l26,kernels[32],*weights[3],*tensors[5],*tensors[6]);
    LaunchPool(kernels[29],*tensors[6],*tensors[7],16,12,128);
    Launch3x3(l28,kernels[33],*weights[4],*tensors[7],*tensors[8]);
    LaunchPool(kernels[29],*tensors[8],*tensors[9],8,6,256);
    Launch1x1(l30,kernels[34],*weights[5],*tensors[9],*tensors[10]);
    Launch1x1(l31,kernels[35],*weights[6],*tensors[10],*tensors[11]);
    LaunchUpscale(kernels[36],*tensors[11],*tensors[12],4,3,8,6,
                  {1,2,2},{32,5,5});
    LaunchAdd(kernels[37],*tensors[12],*tensors[8],*tensors[13],8,6,256,2,2,8);
    Launch1x1(l34,kernels[38],*weights[7],*tensors[13],*tensors[14]);
    Launch1x1(l35,kernels[39],*weights[8],*tensors[14],*tensors[15]);
    LaunchUpscale(kernels[40],*tensors[15],*tensors[16],8,6,16,12,
                  {1,2,2},{16,8,8});
    LaunchAdd(kernels[41],*tensors[16],*tensors[6],*tensors[17],16,12,128,4,3,8);
    Launch1x1(l38,kernels[42],*weights[9],*tensors[17],*tensors[18]);
    Launch1x1(l39,kernels[42],*weights[10],*tensors[18],*tensors[19]);
    LaunchUpscale(kernels[40],*tensors[19],*tensors[20],16,12,32,24,
                  {1,4,3},{16,8,8});
    LaunchAdd(kernels[41],*tensors[20],*tensors[4],*tensors[21],32,24,128,8,6,8);
    Launch1x1(l42,kernels[43],*weights[11],*tensors[21],*tensors[22]);
    Launch1x1(l43,kernels[44],*weights[12],*tensors[22],*tensors[23]);
    LaunchUpscale(kernels[45],*tensors[23],*tensors[24],32,24,64,48,
                  {1,6,5},{8,11,11});
    LaunchAdd(kernels[46],*tensors[24],*tensors[2],*tensors[25],64,48,64,16,12,8);
    Launch1x1(l46,kernels[47],*weights[13],*tensors[25],*tensors[26]);
    Launch1x1(l47,kernels[48],*weights[14],*tensors[26],*tensors[27]);
    LaunchUpscale(kernels[49],*tensors[27],*tensors[28],64,48,128,96,
                  {1,8,6},{4,16,16});
    LaunchAdd(kernels[50],*tensors[28],*tensors[0],*tensors[29],128,96,32,32,24,4);
    Launch1x1(l50,kernels[51],*weights[15],*tensors[29],*tensors[30]);
    Launch1x1(l51,kernels[52],*weights[16],*tensors[30],*tensors[31]);'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "connected_l20_l87_bridge.h"',
        INJECTION,
        "injection",
    )
    source = source.replace("RunFgConnectedPrefixL20L51",
                            "RunFgSubrectPrefixL18L49")
    source = source.replace("FgConnectedPrefixResult", "FgSubrectPrefixResult")
    source = source.replace("GenerateEncoderInput()",
                            "GenerateSubrectEncoderInput()")
    source = source.replace("kTensors", "kSubrectTensors")
    source = section(
        source,
        'const auto& l20 = fg_conv_3x3::FindLayer("l20");',
        "Launch1x1(l51, kernels[52], *weights[16], *tensors[30], *tensors[31]);",
        EXECUTION,
        "dynamic prefix execution",
    )
    source = source.replace("kInputBytes",
                            "fg_connected_subrect_l3_l83::kModelInputBytes")
    return "// Generated exact selected subrect RTX prefix.\n" + source


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
