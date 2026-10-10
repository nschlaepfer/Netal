#!/usr/bin/env python3
"""Generate the exact selected 512x288 RTX L18-L83 tail."""

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
namespace subrect = fg_connected_subrect_l3_l83;

std::vector<std::uint8_t> SubrectCropRtx(
    const std::vector<std::uint8_t>& source,std::uint32_t source_width,
    std::uint32_t source_height,std::uint32_t origin_x,
    std::uint32_t origin_y,std::uint32_t width,std::uint32_t height,
    std::uint32_t bytes_per_pixel){
  if(origin_x+width>source_width||origin_y+height>source_height||
     source.size()!=std::size_t{source_width}*source_height*bytes_per_pixel)
    throw std::runtime_error("subrect crop contract");
  std::vector<std::uint8_t> result(std::size_t{width}*height*bytes_per_pixel);
  for(std::uint32_t y=0;y<height;++y)
    std::memcpy(result.data()+std::size_t{y}*width*bytes_per_pixel,
                source.data()+
                    (std::size_t{y+origin_y}*source_width+origin_x)*bytes_per_pixel,
                std::size_t{width}*bytes_per_pixel);
  return result;
}

struct SubrectL55InputsRtx{
  std::vector<std::uint8_t> rgba_a,rgba_b,motion;
  SubrectL55InputsRtx(){
    const GeneratedInputs full;
    rgba_a=SubrectCropRtx(full.rgba_a,640,360,32,24,512,288,8);
    rgba_b=SubrectCropRtx(full.rgba_b,640,360,32,24,512,288,8);
    motion=SubrectCropRtx(full.motion,320,180,16,12,256,144,4);
  }
};

std::vector<std::uint8_t> SubrectBinaryDepthRtx(){
  std::vector<std::uint8_t> result(256u*144u);
  for(std::uint32_t y=0;y<144;++y)
    for(std::uint32_t x=0;x<256;++x)
      result[std::size_t{y}*256u+x]=(((y+12u)+(x+16u))&1u)?0xffu:0u;
  return result;
}

std::array<std::vector<std::uint8_t>,4> SubrectL53InputsRtx(){
  return {SubrectCropRtx(l53::CurrToPrevFlow(),320,180,16,12,256,144,8),
          SubrectCropRtx(l53::PrevToCurrFlow(false),320,180,16,12,256,144,4),
          SubrectCropRtx(l53::CurrDepth(),320,180,16,12,256,144,4),
          SubrectCropRtx(l53::PrevDepth(),320,180,16,12,256,144,4)};
}

std::vector<std::uint8_t> SubrectCandidateInitialRtx(){
  std::vector<std::uint8_t> result(256u*144u*8u);
  for(std::size_t offset=0;offset<result.size();offset+=8u){
    const std::uint64_t value=l53::kInitialOutput;
    std::memcpy(result.data()+offset,&value,sizeof(value));
  }
  return result;
}

std::vector<std::uint8_t> SubrectTailMaskRtx(){
  std::vector<std::uint8_t> result(subrect::kTailMaskBytes);
  for(std::uint32_t y=0;y<subrect::kHeight;++y)
    for(std::uint32_t x=0;x<subrect::kWidth;++x){
      const bool island=(x>=72&&x<192&&y>=48&&y<160)||
                        (x>=344&&x<488&&y>=172&&y<272);
      const bool hole=((x*13u+y*7u)%47u)==0u;
      result[std::size_t{y}*subrect::kWidth+x]=island&&!hole?0xffu:0u;
    }
  return result;
}

std::vector<std::uint8_t> SubrectRegionRtx(
    const std::vector<std::uint8_t>& full){return subrect::Region(full);}
'''


TAIL_EXECUTION = r'''const auto tail_mask_input=SubrectTailMaskRtx();
    GpuTexture tail_mask=UploadTexture(
        gpu,DXGI_FORMAT_R8_UNORM,512,288,1,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,tail_mask_input);
    const std::vector<std::uint8_t> final_initial(subrect::kOutputBytes,poison);
    GpuTexture final_color=UploadTexture(
        gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,final_initial);
    GpuTexture tail_output_mask=UploadTexture(
        gpu,DXGI_FORMAT_R8_UNORM,512,288,1,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
        std::vector<std::uint8_t>(subrect::kTailMaskBytes,poison));
    const std::vector<std::uint8_t> canary_texture_before(
        subrect::kOutputBytes,static_cast<std::uint8_t>(poison^0xffu));
    GpuTexture canary_texture=UploadTexture(
        gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
        D3D12_RESOURCE_STATE_UNORDERED_ACCESS,canary_texture_before);
    std::array<GpuBuffer,4> pyramid{};
    std::array<std::vector<std::uint8_t>,4> pyramid_initial{};
    for(std::size_t index=0;index<pyramid.size();++index){
      pyramid_initial[index]=PoisonGuarded(subrect::kLevelBytes[index],poison);
      pyramid[index]=UploadBuffer(gpu,pyramid_initial[index]);
    }
    ExecuteAndReset(gpu);
    const auto tail_mask_texture=make_texture(tail_mask,18);
    const auto final_color_surface=make_surface(final_color,19);
    const auto tail_output_mask_surface=make_surface(tail_output_mask,20);
    const auto tail_output_mask_texture=make_texture(tail_output_mask,21);
    const auto l55_surface_a_texture=make_texture(surface_a,22);
    const auto l55_surface_b_texture=make_texture(surface_b,23);

    const auto module17_bytes=Read(modules/ModuleName(17));
    if(module17_bytes.size()!=33936u||Sha256(module17_bytes)!=
       "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76")
      throw std::runtime_error("module17 identity");
    NvObjectHandle module17=nullptr,function17=nullptr;
    RequireNvapi(create_module(gpu.device.Get(),module17_bytes.data(),
                               static_cast<std::uint32_t>(module17_bytes.size()),
                               &module17),"CreateCuModule(module17)");
    RequireNvapi(create_function(gpu.device.Get(),module17,"main_kernel",
                                 &function17),"CreateCuFunction(module17)");
    std::array<std::uint8_t,152> p82{};
    PackU64(&p82,0,final_color_surface);PackU32(&p82,8,32);PackU32(&p82,12,24);
    PackU64(&p82,16,l55_surface_a_texture);PackU64(&p82,24,l55_surface_b_texture);
    PackU64(&p82,64,l83.d3d.resource->GetGPUVirtualAddress()+kGuardBytes);
    PackU64(&p82,72,tail_mask_texture);
    for(std::size_t index=0;index<pyramid.size();++index)
      PackU64(&p82,80u+index*8u,
              pyramid[index].resource->GetGPUVirtualAddress()+kGuardBytes);
    PackU64(&p82,112,tail_output_mask_surface);
    PackU32(&p82,128,512);PackU32(&p82,132,288);
    PackU32(&p82,136,512);PackU32(&p82,140,288);
    PackU32(&p82,144,0x3f000000u);PackU32(&p82,148,1);
    const KernelLaunchParams launch82{function17,{32,18,1},{256,1,1},0,
      p82.data(),static_cast<std::uint32_t>(p82.size())};
    RequireNvapi(launch_chain(gpu.list.Get(),&launch82,1),"Launch L82");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(),function17),"destroy17f");
    RequireNvapi(destroy_module(gpu.device.Get(),module17),"destroy17m");
    auto color82_full=ReadTexture(gpu,final_color);
    auto color82=SubrectRegionRtx(color82_full);
    auto mask82=ReadTexture(gpu,tail_output_mask);
    auto surface_a_after82=ReadTexture(gpu,surface_a);
    auto surface_b_after82=ReadTexture(gpu,surface_b);
    auto l83_after82_guarded=ReadBuffer(gpu,l83.d3d);
    std::array<std::vector<std::uint8_t>,4> pyramid82_guarded{},pyramid82{};
    for(std::size_t index=0;index<pyramid.size();++index){
      pyramid82_guarded[index]=ReadBuffer(gpu,pyramid[index]);
      pyramid82[index]=Payload(pyramid82_guarded[index],subrect::kLevelBytes[index]);
    }
    const auto checkpoint82=Join({&color82,&mask82,&pyramid82[0],&pyramid82[1],
                                  &pyramid82[2],&pyramid82[3]});
    Restore(gpu,final_color,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,tail_output_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    for(auto& item:pyramid)Restore(gpu,item,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);

    const auto module19_bytes=Read(modules/ModuleName(19));
    if(module19_bytes.size()!=57792u||Sha256(module19_bytes)!=
       "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0")
      throw std::runtime_error("module19 identity");
    NvObjectHandle module19=nullptr,function19=nullptr;
    RequireNvapi(create_module(gpu.device.Get(),module19_bytes.data(),
                               static_cast<std::uint32_t>(module19_bytes.size()),
                               &module19),"CreateCuModule(module19)");
    RequireNvapi(create_function(gpu.device.Get(),module19,"main_kernel",
                                 &function19),"CreateCuFunction(module19)");
    std::array<std::uint8_t,80> p83{};
    for(std::size_t index=0;index<pyramid.size();++index)
      PackU64(&p83,8u+index*8u,
              pyramid[index].resource->GetGPUVirtualAddress()+kGuardBytes);
    PackU32(&p83,40,512);PackU32(&p83,44,288);PackU32(&p83,48,1);
    PackU64(&p83,56,tail_output_mask_texture);
    PackU64(&p83,64,final_color_surface);
    PackU32(&p83,72,32);PackU32(&p83,76,24);
    const KernelLaunchParams launch83{function19,{32,18,1},{256,1,1},0,
      p83.data(),static_cast<std::uint32_t>(p83.size())};
    RequireNvapi(launch_chain(gpu.list.Get(),&launch83,1),"Launch L83");
    ExecuteAndReset(gpu);
    RequireNvapi(destroy_function(gpu.device.Get(),function19),"destroy19f");
    RequireNvapi(destroy_module(gpu.device.Get(),module19),"destroy19m");
    auto final_full=ReadTexture(gpu,final_color);
    auto final_bytes=SubrectRegionRtx(final_full);
    auto canary_texture_after=ReadTexture(gpu,canary_texture);'''


VERIFICATION = r'''const auto l52_checkpoint=Join({&l52_a_bytes,&l52_b_bytes,&l52_confidence_bytes});
    const auto l53_checkpoint=Join({&l53_a_bytes,&l53_b_bytes});
    const auto l55_checkpoint=Join({&surface_a_bytes,&surface_b_bytes,&surface_mask_bytes,
                                    &linear_a_bytes,&linear_b_bytes,&linear_mask_bytes});
    const auto l53_a_post=ReadBuffer(gpu,l53_a),l53_b_post=ReadBuffer(gpu,l53_b);
    const auto linear_a_post=ReadBuffer(gpu,linear_a.d3d),
               linear_b_post=ReadBuffer(gpu,linear_b.d3d),
               linear_mask_post=ReadBuffer(gpu,linear_mask.d3d);
    std::size_t canaries=prefix_result.canary_mismatched_bytes+
      CanaryMismatches(l51_guarded,subrect::kL51Bytes)+
      CanaryMismatches(l53_a_guarded,256u*144u*8u)+
      CanaryMismatches(l53_b_guarded,256u*144u*8u)+
      CanaryMismatches(l53_a_post,256u*144u*8u)+
      CanaryMismatches(l53_b_post,256u*144u*8u)+
      CanaryMismatches(linear_a_guarded,subrect::kInputABytes)+
      CanaryMismatches(linear_b_guarded,subrect::kInputBBytes)+
      CanaryMismatches(linear_mask_guarded,subrect::kInputCBytes)+
      CanaryMismatches(linear_a_post,subrect::kInputABytes)+
      CanaryMismatches(linear_b_post,subrect::kInputBBytes)+
      CanaryMismatches(linear_mask_post,subrect::kInputCBytes)+
      core_result.canary_mismatched_bytes+
      CanaryMismatches(l83_guarded,subrect::kL83Bytes)+
      CanaryMismatches(l83_after82_guarded,subrect::kL83Bytes);
    for(std::size_t index=0;index<pyramid.size();++index)
      canaries+=CanaryMismatches(pyramid82_guarded[index],subrect::kLevelBytes[index]);
    std::size_t input_mismatches=prefix_result.input_model_mismatched_bytes+
      core_result.input_model_mismatched_bytes+
      Mismatches(l52_depth,ReadTexture(gpu,l52_depth_texture))+
      Mismatches(l52_a_bytes,ReadTexture(gpu,l52_a))+
      Mismatches(l52_b_bytes,ReadTexture(gpu,l52_b))+
      Mismatches(l52_confidence_bytes,ReadTexture(gpu,l52_confidence))+
      Mismatches(l53_external[0],ReadTexture(gpu,l53_flow_a))+
      Mismatches(l53_external[1],ReadTexture(gpu,l53_flow_b))+
      Mismatches(l53_external[2],ReadTexture(gpu,l53_depth_a))+
      Mismatches(l53_external[3],ReadTexture(gpu,l53_depth_b))+
      Mismatches(l53_a_guarded,l53_a_post)+Mismatches(l53_b_guarded,l53_b_post)+
      Mismatches(l54_bytes,ReadTexture(gpu,l54_blend))+
      Mismatches(l55_external.rgba_a,ReadTexture(gpu,rgba_a))+
      Mismatches(l55_external.rgba_b,ReadTexture(gpu,rgba_b))+
      Mismatches(l55_external.motion,ReadTexture(gpu,motion))+
      Mismatches(surface_a_bytes,surface_a_after82)+
      Mismatches(surface_b_bytes,surface_b_after82)+
      Mismatches(linear_a_guarded,linear_a_post)+
      Mismatches(linear_b_guarded,linear_b_post)+
      Mismatches(linear_mask_guarded,linear_mask_post)+
      Mismatches(tail_mask_input,ReadTexture(gpu,tail_mask))+
      Mismatches(l83_bytes,Payload(l83_after82_guarded,subrect::kL83Bytes))+
      Mismatches(canary_texture_before,canary_texture_after);
    std::size_t readonly_tail_mismatches=0;
    for(std::size_t index=0;index<pyramid.size();++index){
      const auto after=ReadBuffer(gpu,pyramid[index]);
      readonly_tail_mismatches+=Mismatches(
          pyramid82[index],Payload(after,subrect::kLevelBytes[index]));
      canaries+=CanaryMismatches(after,subrect::kLevelBytes[index]);
    }
    input_mismatches+=readonly_tail_mismatches;
    std::array<std::string,66> checkpoints{};
    for(std::size_t index=0;index<32;++index)
      checkpoints[index]=prefix_result.checkpoint_sha256[index];
    checkpoints[32]=Sha256(l52_checkpoint);checkpoints[33]=Sha256(l53_checkpoint);
    checkpoints[34]=Sha256(l54_bytes);checkpoints[35]=Sha256(l55_checkpoint);
    for(std::size_t index=0;index<27;++index)
      checkpoints[36+index]=core_result.checkpoint_sha256[index];
    checkpoints[63]=Sha256(l83_bytes);checkpoints[64]=Sha256(checkpoint82);
    checkpoints[65]=Sha256(final_bytes);
    const bool pass=canaries==0&&input_mismatches==0&&readonly_tail_mismatches==0;
    Write(output_directory/"l49.bin",l51_bytes);Write(output_directory/"l50.bin",l52_checkpoint);
    Write(output_directory/"l51.bin",l53_checkpoint);Write(output_directory/"l52.bin",l54_bytes);
    Write(output_directory/"l53.bin",l55_checkpoint);Write(output_directory/"l81.bin",l83_bytes);
    Write(output_directory/"l82.bin",checkpoint82);Write(output_directory/"output-full.bin",final_full);
    Write(output_directory/"output-region.bin",final_bytes);
    std::ofstream receipt(output_directory/"run-receipt.json");
    receipt<<"{\n  \"schema\": \"neuralstudio.fg.connected-subrect-l18-l83-run.v1\",\n"
           <<"  \"backend\": \"rtx-cuda-driver-d3d12-nvapi\",\n"
           <<"  \"run_id\": \""<<run_id<<"\",\n  \"process_id\": "<<GetCurrentProcessId()<<",\n"
           <<"  \"poison_byte\": "<<unsigned(poison)<<",\n  \"region\": [32, 24, 512, 288],\n"
           <<"  \"launch_range\": [18, 83],\n  \"launch_count\": 66,\n"
           <<"  \"checkpoint_sha256\": {";
    for(std::size_t index=0;index<checkpoints.size();++index)
      receipt<<(index?",":"")<<"\n    \"launch-"<<std::setfill('0')<<std::setw(6)<<(18u+index)<<"\": \""<<checkpoints[index]<<"\"";
    receipt<<"\n  },\n  \"input_model_mismatched_bytes\": "<<input_mismatches
           <<",\n  \"canary_mismatched_bytes\": "<<canaries
           <<",\n  \"readonly_tail_mismatched_bytes\": "<<readonly_tail_mismatches
           <<",\n  \"live_device_handoff_count\": 65,\n"
           <<"  \"host_intermediate_ingress\": false,\n"
           <<"  \"expected_or_archived_tensor_ingress\": false,\n"
           <<"  \"address_fields_recorded\": false,\n"
           <<"  \"output_full_bytes\": "<<final_full.size()<<",\n"
           <<"  \"output_full_sha256\": \""<<Sha256(final_full)<<"\",\n"
           <<"  \"output_region_bytes\": "<<final_bytes.size()<<",\n"
           <<"  \"output_region_sha256\": \""<<checkpoints.back()<<"\",\n"
           <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
    std::cout<<"l49_sha256="<<checkpoints[31]<<'\n'
             <<"l50_sha256="<<checkpoints[32]<<'\n'
             <<"l82_sha256="<<checkpoints[64]<<'\n'
             <<"l83_region_sha256="<<checkpoints[65]<<'\n'
             <<"result="<<(pass?"pass":"fail")<<'\n';
    l51.Release();linear_a.Release();linear_b.Release();linear_mask.Release();l83.Release();
    return pass?0:3;'''


def generate(source: str) -> str:
    source = once(
        source,
        '#include "rtx_shared_cuda_buffer.h"',
        '#include "rtx_shared_cuda_buffer.h"\n'
        '#include "../fg_connected_subrect_l3_l83/contract.h"\n'
        '#include "../fg_connected_subrect_l3_l83/rtx_subrect_bridges.h"',
        "subrect includes",
    )
    source = once(source, "namespace l53 = fg_mid_l53;",
                  "namespace l53 = fg_mid_l53;\n" + INJECTION,
                  "helper injection")
    source = source.replace("constexpr std::size_t kL51Bytes = 160u * 96u * 16u * 2u;",
                            "constexpr std::size_t kL51Bytes = subrect::kL51Bytes;")
    source = source.replace("constexpr std::size_t kL52VectorBytes = 320u * 180u * 4u;",
                            "constexpr std::size_t kL52VectorBytes = subrect::kL52VectorBytes;")
    source = source.replace("constexpr std::size_t kL52ScalarBytes = 320u * 180u * 2u;",
                            "constexpr std::size_t kL52ScalarBytes = subrect::kL52ScalarBytes;")
    source = section(source, "constexpr std::size_t kTailMotionBytes", "};",
                     "constexpr std::size_t kTailMotionBytes=subrect::kTailMotionBytes;\n"
                     "constexpr std::size_t kTailMaskBytes=subrect::kTailMaskBytes;\n"
                     "constexpr auto kTailPyramidBytes=subrect::kLevelBytes;",
                     "tail constants")
    source = source.replace("const GeneratedInputs l55_external;",
                            "const SubrectL55InputsRtx l55_external;")
    source = source.replace("const auto l52_depth = BinaryDepth();",
                            "const auto l52_depth = SubrectBinaryDepthRtx();")
    source = source.replace(
        "const std::array<std::vector<std::uint8_t>, 4> l53_external = {\n"
        "        l53::CurrToPrevFlow(), l53::PrevToCurrFlow(false),\n"
        "        l53::CurrDepth(), l53::PrevDepth()};",
        "const auto l53_external=SubrectL53InputsRtx();",
    )
    source = source.replace("const auto candidate_initial = Guarded(l53::InitialOutput());",
                            "const auto candidate_initial=Guarded(SubrectCandidateInitialRtx());")
    source = source.replace("RunFgConnectedPrefixL20L51",
                            "RunFgSubrectPrefixL18L49")
    source = source.replace("FgConnectedPrefixResult", "FgSubrectPrefixResult")
    source = source.replace("RunFgConnectedCoreL56L83Live",
                            "RunFgSubrectCoreL54L80Live")
    source = source.replace("FgConnectedCoreBridgeResult", "FgSubrectCoreResult")

    middle_start=source.index("auto l51 = connected::UploadSharedCudaBuffer")
    middle_end=source.index("const auto tail_mask_input",middle_start)
    middle=source[middle_start:middle_end]
    middle=middle.replace("320, 180","256, 144").replace("320,180","256,144")
    middle=middle.replace("kActiveWidth","512").replace("kActiveHeight","288")
    middle=middle.replace("kCandidateWidth","256").replace("kCandidateHeight","144")
    middle=middle.replace("kSurfaceRgbaBytes","subrect::kInputABytes")
    middle=middle.replace("kSurfaceMaskBytes","subrect::kTailMaskBytes")
    middle=middle.replace("kLinearRgbaBytes","subrect::kInputABytes")
    middle=middle.replace("kLinearMaskBytes","subrect::kInputCBytes")
    middle=middle.replace("fg_connected_custom::kL83Bytes","subrect::kL83Bytes")
    middle=middle.replace("l53::kOutputBytes","256u*144u*8u")
    middle=middle.replace("kRgTextureBytes","subrect::kL52VectorBytes")
    middle=middle.replace("kPhysicalWidth","512").replace("kPhysicalHeight","288")
    middle=middle.replace("kGridX","32").replace("kGridY","18")
    middle=middle.replace("PackU32(&p52, 136, 256);","PackU32(&p52, 136, 256);")
    middle=middle.replace("PackU32(&p52, 144, 160);","PackU32(&p52, 144, 128);")
    middle=middle.replace("PackU32(&p52, 136, 320);","PackU32(&p52, 136, 256);")
    middle=middle.replace("PackU32(&p52, 140, 180);","PackU32(&p52, 140, 144);")
    middle=middle.replace("PackU32(&p52, 160, 3);","PackU32(&p52, 160, 12);")
    middle=middle.replace("PackU32(&p52, 164, 3);","PackU32(&p52, 164, 12);")
    middle=middle.replace("PackU32(&p53, 112, 320);","PackU32(&p53, 112, 256);")
    middle=middle.replace("PackU32(&p53, 116, 180);","PackU32(&p53, 116, 144);")
    middle=middle.replace("PackU32(&p53, 120, 0x414f5c29u);",
                          "PackU32(&p53, 120, 0x4104b5ddu);")
    middle=middle.replace("PackU32(&p54, 0, 320);","PackU32(&p54, 0, 256);")
    middle=middle.replace("PackU32(&p54, 4, 180);","PackU32(&p54, 4, 144);")
    middle=middle.replace("PackU32(&p54, 8, 320);","PackU32(&p54, 8, 256);")
    middle=middle.replace("PackU32(&p54, 12, 180);","PackU32(&p54, 12, 144);")
    middle=middle.replace("{20, 12, 1}, {16, 16, 1}","{16, 9, 1}, {16, 16, 1}")
    middle=middle.replace("{20, 12, 1}, {324, 1, 1}","{16, 9, 1}, {324, 1, 1}")
    source=source[:middle_start]+middle+source[middle_end:]
    # Supply live L55 outputs and selected custom L81 destination.
    source=once(
        source,
        "argv[1], argv[2], poison,\n            static_cast<std::uint64_t>(linear_a.cuda_ptr + kGuardBytes),",
        "argv[1], argv[2], poison,\n            static_cast<std::uint64_t>(linear_a.cuda_ptr + kGuardBytes),",
        "core call anchor",
    )
    source = section(source, "const auto tail_mask_input",
                     "auto final_bytes = ReadTexture(gpu, final_color);",
                     TAIL_EXECUTION, "selected L82-L83")
    source = section(source, "const auto l52_checkpoint", "return pass ? 0 : 3;",
                     VERIFICATION, "tail verification")
    return "// Generated exact selected subrect RTX tail.\n" + source


def main() -> int:
    parser=argparse.ArgumentParser()
    parser.add_argument("source",type=Path);parser.add_argument("output",type=Path)
    args=parser.parse_args()
    if args.output.exists():raise RuntimeError(f"preserve existing output: {args.output}")
    result=generate(args.source.read_text());args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(result);print(f"generated={args.output}");return 0


if __name__=="__main__":raise SystemExit(main())
