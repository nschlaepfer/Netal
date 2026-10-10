#!/usr/bin/env python3
"""Generate the dynamic 512x288 L18-L83 Metal tail from the exact core."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def once(source: str, before: str, after: str, label: str) -> str:
    count = source.count(before)
    if count != 1:
        raise RuntimeError(f"{label}: expected one anchor, found {count}")
    return source.replace(before, after)


def section(source: str, start: str, end: str, replacement: str,
            label: str) -> str:
    begin = source.find(start)
    if begin < 0:
        raise RuntimeError(f"{label}: start anchor missing")
    finish = source.find(end, begin)
    if finish < 0:
        raise RuntimeError(f"{label}: end anchor missing")
    finish += len(end)
    return source[:begin] + replacement + source[finish:]


INJECTION = r'''
namespace subrect = fg_connected_subrect_l3_l83;

struct SubrectDescriptor {
  std::uint64_t data;
  std::uint32_t width, height, format, channels;
};
static_assert(sizeof(SubrectDescriptor) == 24);

template <std::size_t N>
void SubrectStore32(std::array<std::uint8_t,N>* bytes, std::size_t offset,
                    std::uint32_t value) {
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}
template <std::size_t N>
void SubrectStore64(std::array<std::uint8_t,N>* bytes, std::size_t offset,
                    std::uint64_t value) {
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}

struct SubrectL84Parameters {
  std::uint32_t width, height, padded_height, divisor;
  std::uint32_t output_origin_x, output_origin_y;
  std::uint32_t resource_width, resource_height;
};
struct SubrectL84ReductionParameters {
  std::uint32_t input_width, input_height, output_width, output_height;
};

std::vector<std::uint8_t> SubrectCrop(
    const std::vector<std::uint8_t>& source, std::uint32_t source_width,
    std::uint32_t source_height, std::uint32_t origin_x,
    std::uint32_t origin_y, std::uint32_t width, std::uint32_t height,
    std::uint32_t bytes_per_pixel) {
  if (origin_x + width > source_width || origin_y + height > source_height ||
      source.size() != std::size_t{source_width} * source_height *
                           bytes_per_pixel)
    throw std::runtime_error("subrect crop contract");
  std::vector<std::uint8_t> result(
      std::size_t{width} * height * bytes_per_pixel);
  for (std::uint32_t y = 0; y < height; ++y) {
    const auto source_offset =
        (std::size_t{y + origin_y} * source_width + origin_x) *
        bytes_per_pixel;
    const auto destination_offset = std::size_t{y} * width * bytes_per_pixel;
    std::memcpy(result.data() + destination_offset,
                source.data() + source_offset,
                std::size_t{width} * bytes_per_pixel);
  }
  return result;
}

struct SubrectL55Inputs {
  std::vector<std::uint8_t> rgba_a;
  std::vector<std::uint8_t> rgba_b;
  std::vector<std::uint8_t> motion;
  SubrectL55Inputs() {
    const m55::GeneratedInputs full;
    rgba_a = SubrectCrop(full.rgba_a, 640, 360, 32, 24, 512, 288, 8);
    rgba_b = SubrectCrop(full.rgba_b, 640, 360, 32, 24, 512, 288, 8);
    motion = SubrectCrop(full.motion, 320, 180, 16, 12, 256, 144, 4);
  }
};

std::vector<std::uint8_t> SubrectBinaryDepth() {
  std::vector<std::uint8_t> result(256u*144u);
  for(std::uint32_t y=0;y<144;++y)
    for(std::uint32_t x=0;x<256;++x)
      result[std::size_t{y}*256u+x]=
          (((y+12u)+(x+16u))&1u)?0xffu:0u;
  return result;
}

std::array<std::vector<std::uint8_t>, 4> SubrectL53Inputs() {
  return {
      SubrectCrop(m53::CurrToPrevFlow(), 320, 180, 16, 12, 256, 144, 8),
      SubrectCrop(m53::PrevToCurrFlow(false), 320, 180, 16, 12, 256, 144, 4),
      SubrectCrop(m53::CurrDepth(), 320, 180, 16, 12, 256, 144, 4),
      SubrectCrop(m53::PrevDepth(), 320, 180, 16, 12, 256, 144, 4),
  };
}

std::vector<std::uint8_t> SubrectInitialCandidates() {
  std::vector<std::uint8_t> result(256u*144u*8u);
  for(std::size_t offset=0;offset<result.size();offset+=8u) {
    const std::uint64_t value=m53::kInitialOutput;
    std::memcpy(result.data()+offset,&value,sizeof(value));
  }
  return result;
}

std::vector<std::uint8_t> SubrectTailMask() {
  std::vector<std::uint8_t> result(subrect::kTailMaskBytes);
  for (std::uint32_t y = 0; y < subrect::kHeight; ++y) {
    for (std::uint32_t x = 0; x < subrect::kWidth; ++x) {
      const bool island =
          (x >= 72 && x < 192 && y >= 48 && y < 160) ||
          (x >= 344 && x < 488 && y >= 172 && y < 272);
      const bool hole = ((x * 13u + y * 7u) % 47u) == 0u;
      result[std::size_t{y} * subrect::kWidth + x] =
          island && !hole ? 0xffu : 0u;
    }
  }
  return result;
}

id<MTLTexture> SubrectBufferTexture(
    id<MTLBuffer> buffer, MTLPixelFormat format, NSUInteger width,
    NSUInteger height, MTLTextureUsage usage, NSUInteger bytes_per_pixel) {
  auto* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> result = [buffer newTextureWithDescriptor:descriptor
                                                    offset:kGuardBytes
                                               bytesPerRow:width *
                                                           bytes_per_pixel];
  if (result == nil) throw std::runtime_error("buffer texture allocation");
  return result;
}
'''


PREFIX_TENSORS = '''constexpr std::array<PrefixTensorSpec, 32> kPrefixTensors = {{
    {128,96,32},{64,48,32},{64,48,64},{32,24,64},
    {32,24,128},{16,12,128},{16,12,128},{8,6,128},
    {8,6,256},{4,3,256},{4,3,512},{4,3,256},
    {8,6,256},{8,6,256},{8,6,256},{8,6,128},
    {16,12,128},{16,12,128},{16,12,128},{16,12,128},
    {32,24,128},{32,24,128},{32,24,128},{32,24,64},
    {64,48,64},{64,48,64},{64,48,64},{64,48,32},
    {128,96,32},{128,96,32},{128,96,32},{128,96,16},
}};'''


PREFIX_EXECUTION = '''auto l20 = fg_conv_3x3::FindLayer("l20");
      auto l22 = fg_conv_1x1::FindLayer("l22");
      auto l24 = fg_conv_3x3::FindLayer("l24");
      auto l26 = fg_conv_3x3::FindLayer("l26");
      auto l28 = fg_conv_3x3::FindLayer("l28");
      auto l30 = fg_conv_1x1::FindLayer("l30");
      auto l31 = fg_conv_1x1::FindLayer("l31");
      auto l34 = fg_conv_1x1::FindLayer("l34");
      auto l35 = fg_conv_1x1::FindLayer("l35");
      auto l38 = fg_conv_1x1::FindLayer("l38");
      auto l39 = fg_conv_1x1::FindLayer("l39");
      auto l42 = fg_conv_1x1::FindLayer("l42");
      auto l43 = fg_conv_1x1::FindLayer("l43");
      auto l46 = fg_conv_1x1::FindLayer("l46");
      auto l47 = fg_conv_1x1::FindLayer("l47");
      auto l50 = fg_conv_1x1::FindLayer("l50");
      auto l51 = fg_conv_1x1::FindLayer("l51");
      l20.width=128;l22.width=64;l24.width=32;l26.width=16;l28.width=8;
      l30.width=4;l31.width=4;l34.width=8;l35.width=8;
      l38.width=16;l39.width=16;l42.width=32;l43.width=32;
      l46.width=64;l47.width=64;l50.width=128;l51.width=128;
      id<MTLCommandBuffer> prefix_command = [queue commandBuffer];
      PrefixEncode3x3(prefix_command,conv3,l20,weights[0],prefix_input,tensors[0]);
      PrefixEncodePool(prefix_command,pool,tensors[0],tensors[1],128,96,32);
      PrefixEncode1x1(prefix_command,conv1,l22,weights[1],tensors[1],tensors[2]);
      PrefixEncodePool(prefix_command,pool,tensors[2],tensors[3],64,48,64);
      PrefixEncode3x3(prefix_command,conv3,l24,weights[2],tensors[3],tensors[4]);
      PrefixEncodePool(prefix_command,pool,tensors[4],tensors[5],32,24,128);
      PrefixEncode3x3(prefix_command,conv3,l26,weights[3],tensors[5],tensors[6]);
      PrefixEncodePool(prefix_command,pool,tensors[6],tensors[7],16,12,128);
      PrefixEncode3x3(prefix_command,conv3,l28,weights[4],tensors[7],tensors[8]);
      PrefixEncodePool(prefix_command,pool,tensors[8],tensors[9],8,6,256);
      PrefixEncode1x1(prefix_command,conv1,l30,weights[5],tensors[9],tensors[10]);
      PrefixEncode1x1(prefix_command,conv1,l31,weights[6],tensors[10],tensors[11]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[11],tensors[12],4,3,8,6,256);
      PrefixEncodeAdd(prefix_command,add,tensors[12],tensors[8],tensors[13],8,6,256,2,2,8);
      PrefixEncode1x1(prefix_command,conv1,l34,weights[7],tensors[13],tensors[14]);
      PrefixEncode1x1(prefix_command,conv1,l35,weights[8],tensors[14],tensors[15]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[15],tensors[16],8,6,16,12,128);
      PrefixEncodeAdd(prefix_command,add,tensors[16],tensors[6],tensors[17],16,12,128,4,3,8);
      PrefixEncode1x1(prefix_command,conv1,l38,weights[9],tensors[17],tensors[18]);
      PrefixEncode1x1(prefix_command,conv1,l39,weights[10],tensors[18],tensors[19]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[19],tensors[20],16,12,32,24,128);
      PrefixEncodeAdd(prefix_command,add,tensors[20],tensors[4],tensors[21],32,24,128,8,6,8);
      PrefixEncode1x1(prefix_command,conv1,l42,weights[11],tensors[21],tensors[22]);
      PrefixEncode1x1(prefix_command,conv1,l43,weights[12],tensors[22],tensors[23]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[23],tensors[24],32,24,64,48,64);
      PrefixEncodeAdd(prefix_command,add,tensors[24],tensors[2],tensors[25],64,48,64,16,12,8);
      PrefixEncode1x1(prefix_command,conv1,l46,weights[13],tensors[25],tensors[26]);
      PrefixEncode1x1(prefix_command,conv1,l47,weights[14],tensors[26],tensors[27]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[27],tensors[28],64,48,128,96,32);
      PrefixEncodeAdd(prefix_command,add,tensors[28],tensors[0],tensors[29],128,96,32,32,24,4);
      PrefixEncode1x1(prefix_command,conv1,l50,weights[15],tensors[29],tensors[30]);
      PrefixEncode1x1(prefix_command,conv1,l51,weights[16],tensors[30],tensors[31]);
      Wait(prefix_command);'''


PIPELINES = '''auto l84_pixels = Pipeline(
          device, libs / "postprocess_l84_subrect.metallib",
          "fg_subrect_postprocess_l84_pixels");
      std::array<id<MTLComputePipelineState>,4> l84_levels = {
          Pipeline(device,libs/"postprocess_l84_subrect.metallib","fg_subrect_postprocess_l84_level0"),
          Pipeline(device,libs/"postprocess_l84_subrect.metallib","fg_subrect_postprocess_l84_level1"),
          Pipeline(device,libs/"postprocess_l84_subrect.metallib","fg_subrect_postprocess_l84_level2"),
          Pipeline(device,libs/"postprocess_l84_subrect.metallib","fg_subrect_postprocess_l84_level3")};
      auto l86_pipeline = Pipeline(
          device, libs / "postprocess_l86_subrect.metallib",
          "fg_postprocess_l86_mode1_subrect");'''


PRE_TAIL = '''const auto tail_mask_before = SubrectTailMask();
      id<MTLBuffer> tail_mask =
          Guarded(device,&tail_mask_before,subrect::kTailMaskBytes,0);
      id<MTLBuffer> tail_sigmoid =
          Guarded(device,&sigmoid,kSigmoidBytes,0);
      id<MTLBuffer> tail_scratch =
          Guarded(device,nullptr,subrect::kTailScratchBytes,poison);
      std::array<id<MTLBuffer>,4> pyramid = {
          Guarded(device,nullptr,subrect::kLevelBytes[0],poison),
          Guarded(device,nullptr,subrect::kLevelBytes[1],poison),
          Guarded(device,nullptr,subrect::kLevelBytes[2],poison),
          Guarded(device,nullptr,subrect::kLevelBytes[3],poison)};
      const InitialParams initial_params{288,512,0.5f};
      const ResidualParams residual_off{0},residual_on{1};
      const UpsampleParams up0{144,256,288,512,4};
      const UpsampleParams up1{144,256,288,512,1};
      const UpsampleParams up2{144,256,288,512,3};
      const CentralParams central_params{512,288,0.5f};
      id<MTLBuffer> initial_params_buffer=[device newBufferWithBytes:&initial_params length:sizeof(initial_params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> residual_off_buffer=[device newBufferWithBytes:&residual_off length:sizeof(residual_off) options:MTLResourceStorageModeShared];
      id<MTLBuffer> residual_on_buffer=[device newBufferWithBytes:&residual_on length:sizeof(residual_on) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up0_buffer=[device newBufferWithBytes:&up0 length:sizeof(up0) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up1_buffer=[device newBufferWithBytes:&up1 length:sizeof(up1) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up2_buffer=[device newBufferWithBytes:&up2 length:sizeof(up2) options:MTLResourceStorageModeShared];
      id<MTLBuffer> central_params_buffer=[device newBufferWithBytes:&central_params length:sizeof(central_params) options:MTLResourceStorageModeShared];
      const auto G=NSUInteger(kGuardBytes),Z=NSUInteger(0);'''


TAIL_EXECUTION = r'''const auto l83_before_tail = Payload(l83,subrect::kL83Bytes);
      const std::vector<std::uint8_t> output_initial(
          subrect::kOutputBytes, poison);
      const std::vector<std::uint8_t> output_mask_initial(
          subrect::kTailMaskBytes, poison);
      id<MTLBuffer> output_buffer =
          Guarded(device,&output_initial,subrect::kOutputBytes,0);
      id<MTLBuffer> output_mask_buffer =
          Guarded(device,&output_mask_initial,subrect::kTailMaskBytes,0);
      id<MTLTexture> tail_color = SubrectBufferTexture(
          output_buffer,MTLPixelFormatRGBA8Unorm,640,360,
          MTLTextureUsageShaderWrite,4);
      id<MTLTexture> tail_mask_output = SubrectBufferTexture(
          output_mask_buffer,MTLPixelFormatR8Unorm,512,288,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,1);
      const SubrectL84Parameters l84_parameters{
          512,288,288,1,32,24,640,360};
      id<MTLCommandBuffer> l84_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l84_encoder=
          [l84_command computeCommandEncoder];
      [l84_encoder setComputePipelineState:l84_pixels];
      const std::array<id<MTLBuffer>,6> l84_buffers = {
          l84_color_a,l84_color_b,tail_mask,l83,tail_sigmoid,tail_scratch};
      for(NSUInteger index=0;index<l84_buffers.size();++index)
        [l84_encoder setBuffer:l84_buffers[index] offset:G atIndex:index];
      [l84_encoder setBytes:&l84_parameters length:sizeof(l84_parameters)
                     atIndex:6];
      [l84_encoder setTexture:tail_color atIndex:0];
      [l84_encoder setTexture:tail_mask_output atIndex:1];
      [l84_encoder dispatchThreads:MTLSizeMake(512,288,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l84_encoder endEncoding];
      const std::array<MTLSize,4> l84_grid = {
          MTLSizeMake(256,144,1),MTLSizeMake(128,72,1),
          MTLSizeMake(64,36,1),MTLSizeMake(32,18,1)};
      for(NSUInteger index=0;index<l84_levels.size();++index){
        const SubrectL84ReductionParameters reduction{
            512,288,static_cast<std::uint32_t>(l84_grid[index].width),
            static_cast<std::uint32_t>(l84_grid[index].height)};
        id<MTLComputeCommandEncoder> encoder=
            [l84_command computeCommandEncoder];
        [encoder setComputePipelineState:l84_levels[index]];
        [encoder setBuffer:tail_scratch offset:G atIndex:0];
        [encoder setBuffer:pyramid[index] offset:G atIndex:1];
        [encoder setBuffer:rom_buffer offset:G atIndex:2];
        [encoder setBytes:&reduction length:sizeof(reduction) atIndex:3];
        [encoder dispatchThreads:l84_grid[index]
            threadsPerThreadgroup:MTLSizeMake(16,8,1)];
        [encoder endEncoding];
      }
      Wait(l84_command);
      const auto color84_full=Payload(output_buffer,subrect::kOutputBytes);
      const auto color84=subrect::Region(color84_full);
      const auto mask84=Payload(output_mask_buffer,subrect::kTailMaskBytes);
      const std::array<std::vector<std::uint8_t>,4> pyramid84 = {
          Payload(pyramid[0],subrect::kLevelBytes[0]),
          Payload(pyramid[1],subrect::kLevelBytes[1]),
          Payload(pyramid[2],subrect::kLevelBytes[2]),
          Payload(pyramid[3],subrect::kLevelBytes[3])};
      const auto checkpoint84=JoinVec({color84,mask84,pyramid84[0],
                                       pyramid84[1],pyramid84[2],
                                       pyramid84[3]});

      const SubrectDescriptor mask_descriptor{
          output_mask_buffer.gpuAddress+G,512,288,8,1};
      const SubrectDescriptor output_descriptor{
          output_buffer.gpuAddress+G,640,360,8,4};
      id<MTLBuffer> mask_descriptor_buffer=[device
          newBufferWithBytes:&mask_descriptor length:sizeof(mask_descriptor)
          options:MTLResourceStorageModeShared];
      id<MTLBuffer> output_descriptor_buffer=[device
          newBufferWithBytes:&output_descriptor length:sizeof(output_descriptor)
          options:MTLResourceStorageModeShared];
      std::array<std::uint8_t,80> l86_parameters{};
      for(std::size_t index=0;index<pyramid.size();++index)
        SubrectStore64(&l86_parameters,8u+index*8u,
                       pyramid[index].gpuAddress+G);
      SubrectStore32(&l86_parameters,40,512);
      SubrectStore32(&l86_parameters,44,288);
      SubrectStore32(&l86_parameters,48,1);
      SubrectStore64(&l86_parameters,56,mask_descriptor_buffer.gpuAddress);
      SubrectStore64(&l86_parameters,64,output_descriptor_buffer.gpuAddress);
      SubrectStore32(&l86_parameters,72,32);
      SubrectStore32(&l86_parameters,76,24);
      id<MTLCommandBuffer> l86_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l86_encoder=
          [l86_command computeCommandEncoder];
      [l86_encoder setComputePipelineState:l86_pipeline];
      [l86_encoder setBytes:l86_parameters.data() length:l86_parameters.size()
                     atIndex:0];
      [l86_encoder setBuffer:rom_buffer offset:G atIndex:1];
      for(id<MTLBuffer> item:pyramid)
        [l86_encoder useResource:item usage:MTLResourceUsageRead];
      [l86_encoder useResource:output_mask_buffer usage:MTLResourceUsageRead];
      [l86_encoder useResource:output_buffer
                         usage:MTLResourceUsageRead|MTLResourceUsageWrite];
      [l86_encoder useResource:mask_descriptor_buffer usage:MTLResourceUsageRead];
      [l86_encoder useResource:output_descriptor_buffer usage:MTLResourceUsageRead];
      [l86_encoder dispatchThreadgroups:MTLSizeMake(32,18,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [l86_encoder endEncoding];
      Wait(l86_command);
      const auto final_full=Payload(output_buffer,subrect::kOutputBytes);
      const auto final_frame=subrect::Region(final_full);'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "../fg_mid_l55/fg_mid_l55_inputs.h"\n',
        '#include "../fg_mid_l55/fg_mid_l55_inputs.h"\n'
        '#include "../fg_connected_subrect_l3_l83/contract.h"\n',
        "subrect contract include",
    )
    source = once(source, "namespace m55 = fg_mid_l55;\n",
                  "namespace m55 = fg_mid_l55;\n" + INJECTION,
                  "helper injection")
    source = section(
        source,
        "constexpr std::array<PrefixTensorSpec, 32> kPrefixTensors = {{",
        "}};",
        PREFIX_TENSORS,
        "prefix tensors",
    )
    source = source.replace("kInputBytes", "subrect::kModelInputBytes")
    source = section(
        source,
        'const auto& l20 = fg_conv_3x3::FindLayer("l20");',
        "Wait(prefix_command);",
        PREFIX_EXECUTION,
        "prefix execution",
    )
    source = section(
        source,
        'auto l84_pixels = Pipeline(device, libs / "postprocess_l84.metallib",',
        'auto l87_pipeline = Pipeline(device, libs / "compositor_l87.metallib",\n                                   "fg_compositor_l87_exact");',
        PIPELINES,
        "tail pipelines",
    )

    source = once(source, "if (argc != 8)", "if (argc != 10)", "argc")
    source = once(
        source,
        '"<sigmoid> <output> <run-id> <poison-hex>\\n";',
        '"<sigmoid> <ex2-unit> <ex2-fraction> <output> <run-id> <poison-hex>\\n";',
        "usage",
    )
    source = source.replace("const std::filesystem::path output = argv[5];",
                            "const std::filesystem::path output = argv[7];")
    source = source.replace("std::strtoul(argv[7], &end, 16)",
                            "std::strtoul(argv[9], &end, 16)")
    source = source.replace("end == argv[7]", "end == argv[9]")
    source = source.replace("const auto depth_before = BinaryDepthMac();",
                            "const auto depth_before = SubrectBinaryDepth();")
    source = source.replace(
        "const std::array<std::vector<std::uint8_t>, 4> l53_input_before = {\n"
        "          m53::CurrToPrevFlow(), m53::PrevToCurrFlow(false),\n"
        "          m53::CurrDepth(), m53::PrevDepth()};",
        "const auto l53_input_before = SubrectL53Inputs();",
    )
    source = source.replace("const m55::GeneratedInputs l55_input;",
                            "const SubrectL55Inputs l55_input;")
    source = source.replace("const auto tail_mask_before = TailMask();",
                            "const auto tail_mask_before = SubrectTailMask();")

    mid_start = source.index("id<MTLBuffer> rom_buffer")
    custom_start = source.index("std::array<std::vector<std::uint8_t>,25>",
                                mid_start)
    mid = source[mid_start:custom_start]
    mid = mid.replace("320u * 180u", "256u * 144u")
    mid = mid.replace("320, 180", "256, 144")
    mid = mid.replace("320,180", "256,144")
    mid = mid.replace("m53::kOutputBytes", "256u*144u*8u")
    mid = mid.replace("m53::kPixelCount", "256u*144u")
    mid = mid.replace("fg_mid_l54::kOutputBytes", "256u*144u*4u")
    mid = mid.replace("const auto l53_initial = m53::InitialOutput();",
                      "const auto l53_initial = SubrectInitialCandidates();")
    mid = mid.replace("12.96000003814697265625f",
                      "std::bit_cast<float>(0x4104b5ddu)")
    mid = mid.replace("MTLSizeMake(20,12,1)", "MTLSizeMake(16,9,1)")
    mid = mid.replace("const L52Params l52_params{256,144,160,96,0,0,3,3};",
                      "const L52Params l52_params{256,144,128,96,0,0,12,12};")
    mid = mid.replace("m55::kActiveWidth", "512")
    mid = mid.replace("m55::kActiveHeight", "288")
    mid = mid.replace("m55::kCandidateWidth", "256")
    mid = mid.replace("m55::kCandidateHeight", "144")
    mid = mid.replace("m55::kSurfaceRgbaBytes", "subrect::kInputABytes")
    mid = mid.replace("m55::kSurfaceMaskBytes", "subrect::kTailMaskBytes")
    mid = mid.replace("m55::kLinearRgbaBytes", "subrect::kInputABytes")
    mid = mid.replace("m55::kLinearMaskBytes", "subrect::kInputCBytes")
    mid = mid.replace("m55::kPhysicalWidth", "512")
    mid = mid.replace("m55::kPhysicalHeight", "288")
    mid = mid.replace("MTLSizeMake(640,360,1)", "MTLSizeMake(512,288,1)")
    mid = mid.replace("MTLSizeMake(640, 360, 1)", "MTLSizeMake(512, 288, 1)")
    mid = mid.replace("destinationBytesPerRow:640u*8u",
                      "destinationBytesPerRow:512u*8u")
    mid = mid.replace("destinationBytesPerImage:640u*360u*8u",
                      "destinationBytesPerImage:512u*288u*8u")
    source = source[:mid_start] + mid + source[custom_start:]

    for name in (
        "kL56Bytes", "kL57Bytes", "kL58Bytes", "kLow64Bytes",
        "kL68PlaneBytes", "kL69Out0Bytes", "kL69Out1Bytes",
        "kL69Out2Bytes", "kL70Out0Bytes", "kL70Out1Bytes",
        "kL70Out2Bytes", "kL71Bytes", "kL72Bytes", "kBlock1Bytes",
        "kL82PlaneBytes", "kL83Bytes",
    ):
        source = source.replace(f"c::{name}", f"subrect::{name}")
    source = re.sub(r"(?<!subrect::)\bkTailMaskBytes\b",
                    "subrect::kTailMaskBytes", source)
    source = re.sub(r"(?<!subrect::)\bkTailScratchBytes\b",
                    "subrect::kTailScratchBytes", source)
    source = re.sub(r"(?<!subrect::)\bkPyramidBytes\b",
                    "subrect::kLevelBytes", source)

    pre_tail_start = source.index("const auto tail_mask_before = SubrectTailMask();")
    pre_tail_end = source.index("const auto G=NSUInteger(kGuardBytes),Z=NSUInteger(0);",
                                pre_tail_start)
    pre_tail_end += len("const auto G=NSUInteger(kGuardBytes),Z=NSUInteger(0);")
    source = source[:pre_tail_start] + PRE_TAIL + source[pre_tail_end:]

    custom_anchor = source.index("id<MTLCommandBuffer> core_command")
    tail_anchor = source.index("const auto l83_before_tail", custom_anchor)
    custom = source[custom_anchor:tail_anchor]
    replacements = {
        "{368,640,0.5f}": "{288,512,0.5f}",
        "{184,320,368,640,4}": "{144,256,288,512,4}",
        "{184,320,368,640,1}": "{144,256,288,512,1}",
        "{184,320,368,640,3}": "{144,256,288,512,3}",
        "{640,368,0.5f}": "{512,288,0.5f}",
        "MTLSizeMake(640,368,1)": "MTLSizeMake(512,288,1)",
        "MTLSizeMake(320,184,32)": "MTLSizeMake(256,144,32)",
        "MTLSizeMake(160,92,32)": "MTLSizeMake(128,72,32)",
        "MTLSizeMake(80,46,64)": "MTLSizeMake(64,36,64)",
        "MTLSizeMake(160,92,96)": "MTLSizeMake(128,72,96)",
        "MTLSizeMake(320,184,8)": "MTLSizeMake(256,144,8)",
        "MTLSizeMake(640,368,4)": "MTLSizeMake(512,288,4)",
        "MTLSizeMake(640,368,3)": "MTLSizeMake(512,288,3)",
        "MTLSizeMake(320,184,16)": "MTLSizeMake(256,144,16)",
        "MTLSizeMake(160,92,32)": "MTLSizeMake(128,72,32)",
        "MTLSizeMake(320,184,48)": "MTLSizeMake(256,144,48)",
        "MTLSizeMake(640,368,8)": "MTLSizeMake(512,288,8)",
    }
    for before, after in replacements.items():
        custom = custom.replace(before, after)
    source = source[:custom_anchor] + custom + source[tail_anchor:]
    source = section(source, "const auto l83_before_tail",
                     "const auto final_frame=TailReadTexture(tail_color,4);",
                     TAIL_EXECUTION, "subrect L82/L83")

    verification_start = "std::array<std::string, 68> checkpoints{};"
    verification_end = "return pass ? 0 : 3;"
    verification = r'''std::array<std::string,66> checkpoints{};
      std::size_t canaries=0;
      for(std::size_t index=0;index<tensors.size();++index){
        const auto bytes=Payload(tensors[index],PrefixTensorBytes(kPrefixTensors[index]));
        checkpoints[index]=Sha256(bytes);
        canaries+=CanaryMismatches(tensors[index],PrefixTensorBytes(kPrefixTensors[index]));
      }
      const auto l52_a_bytes=ConnectedReadTexture(l52_a,4);
      const auto l52_b_bytes=ConnectedReadTexture(l52_b,4);
      const auto l52_confidence_bytes=ConnectedReadTexture(l52_confidence,2);
      const auto l52_join=JoinVec({l52_a_bytes,l52_b_bytes,l52_confidence_bytes});
      checkpoints[32]=Sha256(l52_join);
      const auto l53_a_bytes=Payload(l53_a,256u*144u*8u);
      const auto l53_b_bytes=Payload(l53_b,256u*144u*8u);
      const auto l53_join=JoinVec({l53_a_bytes,l53_b_bytes});
      checkpoints[33]=Sha256(l53_join);
      const auto l54_bytes=ConnectedReadTexture(l54_blend,4);
      checkpoints[34]=Sha256(l54_bytes);
      checkpoints[35]=Sha256(l55_join_before_tail);
      checkpoints[36]=Sha256(Payload(l56,subrect::kL56Bytes));
      checkpoints[37]=Sha256(Payload(l57,subrect::kL57Bytes));
      checkpoints[38]=Sha256(Payload(l58,subrect::kL58Bytes));
      const std::array<id<MTLBuffer>,9> low64={l59,l60,l61,l62,l63,l64,l65,l66,l67};
      for(std::size_t index=0;index<low64.size();++index)
        checkpoints[39+index]=Sha256(Payload(low64[index],subrect::kLow64Bytes));
      checkpoints[48]=Sha256(Concat({{l68a,subrect::kL68PlaneBytes},{l68b,subrect::kL68PlaneBytes},{l68c,subrect::kL68PlaneBytes}}));
      checkpoints[49]=Sha256(Concat({{l69a,subrect::kL69Out0Bytes},{l69b,subrect::kL69Out1Bytes},{l69c,subrect::kL69Out2Bytes}}));
      checkpoints[50]=Sha256(Concat({{l70a,subrect::kL70Out0Bytes},{l70b,subrect::kL70Out1Bytes},{l70c,subrect::kL70Out2Bytes}}));
      checkpoints[51]=Sha256(Payload(l71,subrect::kL71Bytes));
      checkpoints[52]=Sha256(Payload(l72,subrect::kL72Bytes));
      const std::array<id<MTLBuffer>,9> block1={l73,l74,l75,l76,l77,l78,l79,l80,l81};
      for(std::size_t index=0;index<block1.size();++index)
        checkpoints[53+index]=Sha256(Payload(block1[index],subrect::kBlock1Bytes));
      checkpoints[62]=Sha256(Concat({{l82a,subrect::kL82PlaneBytes},{l82b,subrect::kL82PlaneBytes},{l82c,subrect::kL82PlaneBytes}}));
      checkpoints[63]=Sha256(l83_before_tail);
      checkpoints[64]=Sha256(checkpoint84);
      checkpoints[65]=Sha256(final_frame);
      std::size_t mismatches=
          Mismatches(prefix_input_before,Payload(prefix_input,subrect::kModelInputBytes))+
          Mismatches(depth_before,ConnectedReadTexture(depth,1))+
          Mismatches(rom,Payload(rom_buffer,rom.size()))+
          Mismatches(ex2_unit,Payload(ex2_unit_buffer,ex2_unit.size()))+
          Mismatches(ex2_fraction,Payload(ex2_fraction_buffer,ex2_fraction.size()));
      for(std::size_t index=0;index<l53_input_before.size();++index){
        const std::array<id<MTLTexture>,4> textures={l53_flow_a,l53_flow_b,l53_depth_a,l53_depth_b};
        const std::array<NSUInteger,4> bpp={8,4,4,4};
        mismatches+=Mismatches(l53_input_before[index],ConnectedReadTexture(textures[index],bpp[index]));
      }
      mismatches+=Mismatches(l55_input.rgba_a,ConnectedReadTexture(rgba_a,8));
      mismatches+=Mismatches(l55_input.rgba_b,ConnectedReadTexture(rgba_b,8));
      mismatches+=Mismatches(l55_input.motion,ConnectedReadTexture(motion,4));
      mismatches+=Mismatches(l55_linear_a_before_tail,Payload(linear_a,subrect::kInputABytes));
      mismatches+=Mismatches(l55_linear_b_before_tail,Payload(linear_b,subrect::kInputBBytes));
      mismatches+=Mismatches(l55_linear_mask_before_tail,Payload(linear_mask,subrect::kInputCBytes));
      mismatches+=Mismatches(l83_before_tail,Payload(l83,subrect::kL83Bytes));
      const auto pyramid_after=std::array<std::vector<std::uint8_t>,4>{
        Payload(pyramid[0],subrect::kLevelBytes[0]),Payload(pyramid[1],subrect::kLevelBytes[1]),
        Payload(pyramid[2],subrect::kLevelBytes[2]),Payload(pyramid[3],subrect::kLevelBytes[3])};
      std::size_t readonly_tail_mismatches=0;
      for(std::size_t index=0;index<4;++index)
        readonly_tail_mismatches+=Mismatches(pyramid84[index],pyramid_after[index]);
      mismatches+=readonly_tail_mismatches;
      canaries+=CanaryMismatches(prefix_input,subrect::kModelInputBytes)+
                CanaryMismatches(rom_buffer,rom.size())+
                CanaryMismatches(ex2_unit_buffer,ex2_unit.size())+
                CanaryMismatches(ex2_fraction_buffer,ex2_fraction.size())+
                CanaryMismatches(l53_a,256u*144u*8u)+CanaryMismatches(l53_b,256u*144u*8u)+
                CanaryMismatches(high_a,high_initial.size())+CanaryMismatches(low_a,low_initial.size())+
                CanaryMismatches(high_b,high_initial.size())+CanaryMismatches(low_b,low_initial.size())+
                CanaryMismatches(linear_a,subrect::kInputABytes)+CanaryMismatches(linear_b,subrect::kInputBBytes)+
                CanaryMismatches(linear_mask,subrect::kInputCBytes)+CanaryMismatches(l84_color_a,subrect::kInputABytes)+
                CanaryMismatches(l84_color_b,subrect::kInputBBytes)+CanaryMismatches(tail_mask,subrect::kTailMaskBytes)+
                CanaryMismatches(tail_sigmoid,kSigmoidBytes)+CanaryMismatches(tail_scratch,subrect::kTailScratchBytes)+
                CanaryMismatches(output_buffer,subrect::kOutputBytes)+CanaryMismatches(output_mask_buffer,subrect::kTailMaskBytes);
      const std::array<std::pair<id<MTLBuffer>,std::size_t>,36> tail_buffers={{{l56,subrect::kL56Bytes},{l57,subrect::kL57Bytes},{l58,subrect::kL58Bytes},{l59,subrect::kLow64Bytes},{l60,subrect::kLow64Bytes},{l61,subrect::kLow64Bytes},{l62,subrect::kLow64Bytes},{l63,subrect::kLow64Bytes},{l64,subrect::kLow64Bytes},{l65,subrect::kLow64Bytes},{l66,subrect::kLow64Bytes},{l67,subrect::kLow64Bytes},{l68a,subrect::kL68PlaneBytes},{l68b,subrect::kL68PlaneBytes},{l68c,subrect::kL68PlaneBytes},{l69a,subrect::kL69Out0Bytes},{l69b,subrect::kL69Out1Bytes},{l69c,subrect::kL69Out2Bytes},{l70a,subrect::kL70Out0Bytes},{l70b,subrect::kL70Out1Bytes},{l70c,subrect::kL70Out2Bytes},{l71,subrect::kL71Bytes},{l72,subrect::kL72Bytes},{l73,subrect::kBlock1Bytes},{l74,subrect::kBlock1Bytes},{l75,subrect::kBlock1Bytes},{l76,subrect::kBlock1Bytes},{l77,subrect::kBlock1Bytes},{l78,subrect::kBlock1Bytes},{l79,subrect::kBlock1Bytes},{l80,subrect::kBlock1Bytes},{l81,subrect::kBlock1Bytes},{l82a,subrect::kL82PlaneBytes},{l82b,subrect::kL82PlaneBytes},{l82c,subrect::kL82PlaneBytes},{l83,subrect::kL83Bytes}}};
      for(const auto& item:tail_buffers)canaries+=CanaryMismatches(item.first,item.second);
      for(std::size_t index=0;index<pyramid.size();++index)
        canaries+=CanaryMismatches(pyramid[index],subrect::kLevelBytes[index]);
      for(std::size_t index=0;index<weights.size();++index){
        mismatches+=Mismatches(weight_before[index],Payload(weights[index],kPrefixWeights[index].bytes));
        canaries+=CanaryMismatches(weights[index],kPrefixWeights[index].bytes);
      }
      for(std::size_t index=0;index<tail_weight.size();++index){
        mismatches+=Mismatches(tail_weight_before[index],Payload(tail_weight[index],c::kFullModels[index].weight_bytes));
        mismatches+=Mismatches(tail_bias_before[index],Payload(tail_bias[index],c::kFullModels[index].bias_bytes));
        canaries+=CanaryMismatches(tail_weight[index],c::kFullModels[index].weight_bytes);
        canaries+=CanaryMismatches(tail_bias[index],c::kFullModels[index].bias_bytes);
      }
      const auto report_canary=[](const char* label,id<MTLBuffer> buffer,
                                  std::size_t bytes){
        const auto count=CanaryMismatches(buffer,bytes);
        if(count)std::cerr<<"canary."<<label<<"="<<count<<'\n';
      };
      for(std::size_t index=0;index<tensors.size();++index){
        const auto label="tensor-"+std::to_string(index);
        report_canary(label.c_str(),tensors[index],PrefixTensorBytes(kPrefixTensors[index]));
      }
      for(std::size_t index=0;index<tail_buffers.size();++index){
        const auto label="tail-"+std::to_string(index);
        report_canary(label.c_str(),tail_buffers[index].first,tail_buffers[index].second);
      }
      for(std::size_t index=0;index<pyramid.size();++index){
        const auto label="pyramid-"+std::to_string(index);
        report_canary(label.c_str(),pyramid[index],subrect::kLevelBytes[index]);
      }
      report_canary("tail-scratch",tail_scratch,subrect::kTailScratchBytes);
      report_canary("output",output_buffer,subrect::kOutputBytes);
      report_canary("output-mask",output_mask_buffer,subrect::kTailMaskBytes);
      const bool pass=mismatches==0&&canaries==0;
      Write(output/"l51.bin",Payload(tensors[31],PrefixTensorBytes(kPrefixTensors[31])));
      Write(output/"l52.bin",l52_join);Write(output/"l53.bin",l53_join);
      Write(output/"l54.bin",l54_bytes);Write(output/"l55.bin",l55_join_before_tail);
      Write(output/"l81.bin",Payload(l81,subrect::kBlock1Bytes));
      Write(output/"l82.bin",checkpoint84);Write(output/"output-full.bin",final_full);
      Write(output/"output-region.bin",final_frame);
      std::ofstream receipt(output/"run-receipt.json");
      receipt<<"{\n  \"schema\": \"neuralstudio.fg.connected-subrect-l18-l83-run.v1\",\n"
             <<"  \"backend\": \"mac-metal\",\n  \"run_id\": \""<<argv[8]<<"\",\n"
             <<"  \"process_id\": "<<getpid()<<",\n  \"poison_byte\": "<<unsigned(poison)<<",\n"
             <<"  \"region\": [32, 24, 512, 288],\n  \"launch_range\": [18, 83],\n"
             <<"  \"launch_count\": 66,\n  \"checkpoint_sha256\": {";
      for(std::size_t index=0;index<checkpoints.size();++index)
        receipt<<(index?",":"")<<"\n    \"launch-"<<std::setfill('0')<<std::setw(6)<<(18u+index)<<"\": \""<<checkpoints[index]<<"\"";
      receipt<<"\n  },\n  \"input_model_mismatched_bytes\": "<<mismatches
             <<",\n  \"canary_mismatched_bytes\": "<<canaries
             <<",\n  \"readonly_tail_mismatched_bytes\": "<<readonly_tail_mismatches
             <<",\n  \"live_device_handoff_count\": 65,\n"
             <<"  \"host_intermediate_ingress\": false,\n"
             <<"  \"expected_or_archived_tensor_ingress\": false,\n"
             <<"  \"address_fields_recorded\": false,\n"
             <<"  \"output_full_bytes\": "<<final_full.size()<<",\n"
             <<"  \"output_full_sha256\": \""<<Sha256(final_full)<<"\",\n"
             <<"  \"output_region_bytes\": "<<final_frame.size()<<",\n"
             <<"  \"output_region_sha256\": \""<<checkpoints.back()<<"\",\n"
             <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
      std::cout<<"l51_sha256="<<checkpoints[31]<<'\n'
               <<"l52_sha256="<<checkpoints[32]<<'\n'
               <<"l82_sha256="<<checkpoints[64]<<'\n'
               <<"l83_region_sha256="<<checkpoints[65]<<'\n'
               <<"result="<<(pass?"pass":"fail")<<'\n';
      return pass?0:3;'''
    source = section(source, verification_start, verification_end,
                     verification, "verification tail")

    # Reference EX2 tables are explicit live inputs to L52/L54.
    source = once(
        source,
        "const auto sigmoid = ReadB(argv[4]);",
        "const auto sigmoid = ReadB(argv[4]);\n"
        "      const auto ex2_unit = ReadB(argv[5]);\n"
        "      const auto ex2_fraction = ReadB(argv[6]);",
        "table reads",
    )
    source = once(
        source,
        'throw std::runtime_error("sigmoid architecture LUT identity mismatch");',
        'throw std::runtime_error("sigmoid architecture LUT identity mismatch");\n'
        '      if(ex2_unit.size()!=67108868u||Sha256(ex2_unit)!=\n'
        '         "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75"||\n'
        '         ex2_fraction.size()!=33554436u||Sha256(ex2_fraction)!=\n'
        '         "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86")\n'
        '        throw std::runtime_error("EX2 reference table identity mismatch");',
        "EX2 table identity",
    )
    source = once(
        source,
        "id<MTLBuffer> rom_buffer = Guarded(device, &rom, rom.size(), 0);",
        "id<MTLBuffer> rom_buffer = Guarded(device, &rom, rom.size(), 0);\n"
        "      id<MTLBuffer> ex2_unit_buffer = Guarded(device,&ex2_unit,ex2_unit.size(),0);\n"
        "      id<MTLBuffer> ex2_fraction_buffer = Guarded(device,&ex2_fraction,ex2_fraction.size(),0);",
        "table buffers",
    )
    source = once(
        source,
        "[l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];",
        "[l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];\n"
        "      [l52_encoder setBuffer:ex2_unit_buffer offset:kGuardBytes atIndex:3];\n"
        "      [l52_encoder setBuffer:ex2_fraction_buffer offset:kGuardBytes atIndex:4];",
        "L52 tables",
    )
    source = once(
        source,
        "[l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];",
        "[l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];\n"
        "      [l54_encoder setBuffer:ex2_unit_buffer offset:kGuardBytes atIndex:4];\n"
        "      [l54_encoder setBuffer:ex2_fraction_buffer offset:kGuardBytes atIndex:5];",
        "L54 tables",
    )
    return "// Generated exact subrect tail; do not edit.\n" + source


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
