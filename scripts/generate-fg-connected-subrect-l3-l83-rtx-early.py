#!/usr/bin/env python3
"""Derive the canonical selected L3-L17 subrect RTX oracle host."""

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


EXECUTION = r'''std::vector<std::pair<std::string,std::string>> checkpoints;
    auto checkpoint=[&](std::uint32_t launch_index,
        std::initializer_list<std::vector<std::uint8_t>> values){
      std::ostringstream label;label<<"launch-"<<std::setfill('0')
        <<std::setw(6)<<launch_index;
      checkpoints.emplace_back(label.str(),Sha256(JoinValues(values)));
    };
    auto encode_m11=[&](std::uint64_t source,std::uint64_t history,
                        std::uint64_t feature,std::uint64_t next_history){
      std::array<std::uint8_t,168> p{};
      Pack64(&p,0,source);Pack32(&p,8,32);Pack32(&p,12,24);
      Pack64(&p,80,history);Pack64(&p,96,feature);
      Pack64(&p,104,shared_metric_surf);Pack64(&p,120,next_history);
      Pack32(&p,128,2);Pack32(&p,136,512);Pack32(&p,140,288);
      Pack32(&p,144,256);Pack32(&p,148,144);
      launch(11,p.data(),p.size(),{32,18,1},{256,1,1});
    };
    encode_m11(source_reset_tex,history_a_tex,reset_feature_surf,
               history_b_surf);
    const auto reset_feature_bytes=snapshot_texture(
        reset_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto reset_metric_bytes=snapshot_texture(
        shared_metric,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto history_b_bytes=snapshot_texture(
        history_b,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(3,{reset_feature_bytes,reset_metric_bytes,
                  connected::HistoryRegion(history_b_bytes,8)});
    Write(output_root/"debug-reset-color.rgba16f",reset_feature_bytes);
    Write(output_root/"debug-reset-metric.r16f",reset_metric_bytes);
    Write(output_root/"debug-reset-history.rgba16f",
          connected::HistoryRegion(history_b_bytes,8));

    auto encode_m15=[&](std::uint64_t vector,std::uint64_t scalar,
                        std::uint64_t depth){
      std::array<std::uint8_t,160> p{};
      Pack64(&p,0,vector);Pack32(&p,8,32);Pack32(&p,12,24);
      Pack32(&p,16,512);Pack32(&p,20,288);
      Pack32(&p,24,0x3acccccd);Pack32(&p,28,0x3b360b61);
      Pack64(&p,32,flow_surf);Pack64(&p,40,scalar);
      Pack32(&p,48,32);Pack32(&p,52,24);
      Pack32(&p,56,512);Pack32(&p,60,288);
      Pack32(&p,64,0x3acccccd);Pack32(&p,68,0x3b360b61);
      Pack32(&p,72,0x42200000);Pack32(&p,76,0x3f800000);
      Pack32(&p,80,0x44160000);p[84]=0;p[85]=127;
      Pack64(&p,88,depth);Pack32(&p,144,256);Pack32(&p,148,144);
      Pack32(&p,152,0x364ccccd);Pack32(&p,156,0x3721d13a);
      launch(15,p.data(),p.size(),{16,9,1},{16,16,1});
    };
    encode_m15(vector_reset_tex,scalar_reset_tex,reset_depth_surf);
    const auto flow_reset_bytes=snapshot_texture(
        flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto reset_depth_bytes=snapshot_texture(
        reset_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(4,{flow_reset_bytes,reset_depth_bytes});

    auto encode_m13=[&](std::uint64_t depth,std::uint64_t mask){
      std::array<std::uint8_t,144> p{};
      Pack64(&p,0,flow_tex);Pack64(&p,8,depth);
      Pack64(&p,16,atomic.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,24,mask);Pack32(&p,40,0x3b000000);
      Pack32(&p,44,0x3b638e39);Pack32(&p,48,256);Pack32(&p,52,144);
      Pack32(&p,56,0x42b851ec);Pack32(&p,60,0x4104b5dd);
      Pack32(&p,64,0x42200000);Pack32(&p,68,0x3f800000);
      Pack32(&p,72,0x44160000);
      const std::array<float,16> identity={
        1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
      std::memcpy(p.data()+80,identity.data(),64);
      launch(13,p.data(),p.size(),{16,9,1},{16,16,1});
    };
    encode_m13(reset_depth_tex,reset_mask_surf);
    const auto atomic_reset_bytes=snapshot_buffer(
        atomic,connected::kEarlyAtomicBytes);
    const auto reset_mask_bytes=snapshot_texture(
        reset_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(5,{atomic_reset_bytes,reset_mask_bytes});

    auto encode_m21=[&](std::uint64_t depth,std::uint64_t mask){
      std::array<std::uint8_t,224> p{};
      Pack64(&p,0,flow_tex);Pack64(&p,8,depth);Pack64(&p,16,mask);
      Pack64(&p,24,atomic.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,32,working_vector_surf);Pack64(&p,40,working_mask_surf);
      for(std::size_t i=0;i<pyramid.size();++i)
        Pack64(&p,48+i*8,pyramid[i].resource->GetGPUVirtualAddress()+
                             kGuardBytes);
      Pack32(&p,128,0x3b000000);Pack32(&p,132,0x3b638e39);
      Pack32(&p,136,256);Pack32(&p,140,144);
      Pack32(&p,144,0x42200000);Pack32(&p,148,0x3f800000);
      Pack32(&p,152,0x44160000);p[156]=0;p[157]=1;
      const std::array<float,16> identity={
        1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
      std::memcpy(p.data()+160,identity.data(),64);
      launch(21,p.data(),p.size(),{16,9,1},{256,1,1});
    };
    encode_m21(generated_depth_tex,generated_mask_tex);
    const auto working_vector_reset=snapshot_texture(
        working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto working_mask_reset=snapshot_texture(
        working_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::array<std::vector<std::uint8_t>,5> pyramid_reset;
    for(std::size_t i=0;i<pyramid.size();++i)
      pyramid_reset[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
    checkpoint(6,{working_vector_reset,working_mask_reset,pyramid_reset[0],
                  pyramid_reset[1],pyramid_reset[2],pyramid_reset[3],
                  pyramid_reset[4]});

    auto encode_m23=[&](){
      std::array<std::uint8_t,72> p{};
      Pack64(&p,8,pyramid[1].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,16,pyramid[2].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,24,pyramid[3].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,32,pyramid[4].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack32(&p,40,256);Pack32(&p,44,144);Pack32(&p,48,1);
      Pack64(&p,56,working_vector_surf);
      Pack64(&p,64,pyramid[0].resource->GetGPUVirtualAddress()+kGuardBytes);
      launch(23,p.data(),p.size(),{16,9,1},{256,1,1});
    };
    encode_m23();
    const auto working_vector_m7=snapshot_texture(
        working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(7,{working_vector_m7});

    auto encode_m25=[&](std::uint64_t scatter){
      std::array<std::uint8_t,48> p{};
      Pack64(&p,0,working_vector_tex);Pack64(&p,8,scatter);
      Pack32(&p,16,0);Pack32(&p,20,256);Pack32(&p,24,144);
      Pack32(&p,28,0x3b800000);Pack32(&p,32,0x3be38e39);
      Pack32(&p,36,256);Pack32(&p,40,144);Pack32(&p,44,0x4104b5dd);
      launch(25,p.data(),p.size(),{16,9,1},{16,16,1});
    };
    encode_m25(scatter_reset_surf);
    const auto scatter_reset_bytes=snapshot_texture(
        scatter_reset,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(8,{scatter_reset_bytes});

    std::array<std::uint8_t,32> p14{};
    Pack64(&p14,0,reset_feature_tex);Pack64(&p14,8,scatter_reset_tex);
    Pack64(&p14,16,reset_projection_surf);Pack32(&p14,24,256);
    Pack32(&p14,28,144);
    launch(14,p14.data(),p14.size(),{16,9,1},{16,16,1});
    const auto reset_projection_bytes=snapshot_texture(
        reset_projection,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(9,{reset_projection_bytes});

    std::array<std::uint8_t,40> p4{};
    Pack64(&p4,0,source_reset_tex);Pack32(&p4,8,32);Pack32(&p4,12,24);
    Pack64(&p4,16,reset_present_surf);Pack32(&p4,24,32);
    Pack32(&p4,28,24);Pack32(&p4,32,512);Pack32(&p4,36,288);
    launch(4,p4.data(),p4.size(),{32,18,1},{16,16,1});
    const auto reset_present_bytes=snapshot_texture(
        reset_present,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    checkpoint(10,{connected::FullRegion(reset_present_bytes,4)});

    Restore(gpu,history_a,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m11(source_generated_tex,history_b_tex,generated_feature_surf,
               history_a_surf);
    const auto generated_feature_bytes=snapshot_texture(
        generated_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto generated_metric_bytes=snapshot_texture(
        shared_metric,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto history_a_bytes=snapshot_texture(
        history_a,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    checkpoint(11,{generated_feature_bytes,generated_metric_bytes,
                   connected::HistoryRegion(history_a_bytes,8)});
    Write(output_root/"debug-generated-color.rgba16f",generated_feature_bytes);
    Write(output_root/"debug-generated-metric.r16f",generated_metric_bytes);
    Write(output_root/"debug-generated-history.rgba16f",
          connected::HistoryRegion(history_a_bytes,8));

    Restore(gpu,flow,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,generated_depth,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m15(vector_generated_tex,scalar_generated_tex,
               generated_depth_surf);
    const auto flow_generated_bytes=snapshot_texture(
        flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    const auto generated_depth_bytes=snapshot_texture(
        generated_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(12,{flow_generated_bytes,generated_depth_bytes});

    OverwriteBuffer(gpu,atomic,Guarded(connected::InitialAtomic()));
    Restore(gpu,generated_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m13(generated_depth_tex,generated_mask_surf);
    const auto atomic_generated_bytes=snapshot_buffer(
        atomic,connected::kEarlyAtomicBytes);
    const auto generated_mask_bytes=snapshot_texture(
        generated_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(13,{atomic_generated_bytes,generated_mask_bytes});

    Restore(gpu,working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_m21(reset_depth_tex,reset_mask_tex);
    const auto working_vector_generated=snapshot_texture(
        working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    const auto working_mask_generated=snapshot_texture(
        working_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    std::array<std::vector<std::uint8_t>,5> pyramid_generated;
    for(std::size_t i=0;i<pyramid.size();++i)
      pyramid_generated[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
    checkpoint(14,{working_vector_generated,working_mask_generated,
                   pyramid_generated[0],pyramid_generated[1],
                   pyramid_generated[2],pyramid_generated[3],
                   pyramid_generated[4]});
    encode_m23();
    const auto working_vector_m15=snapshot_texture(
        working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(15,{working_vector_m15});
    encode_m25(scatter_generated_surf);
    const auto scatter_generated_bytes=snapshot_texture(
        scatter_generated,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoint(16,{scatter_generated_bytes});

    const std::array<std::uint64_t,10> m5_inputs={
      reset_feature_tex,generated_feature_tex,scatter_generated_tex,
      scatter_reset_tex,reset_projection_tex,shared_metric_tex,
      allocation29_tex,flow_tex,working_vector_tex,working_mask_tex};
    std::array<std::uint8_t,192> p5{};
    for(std::size_t i=0;i<m5_inputs.size();++i)Pack64(&p5,i*8,m5_inputs[i]);
    Pack64(&p5,80,model_input.resource->GetGPUVirtualAddress()+kGuardBytes);
    Pack64(&p5,88,auxiliary_mask_surf);Pack64(&p5,96,auxiliary_vector_surf);
    Pack32(&p5,152,256);Pack32(&p5,156,144);
    Pack32(&p5,160,256);Pack32(&p5,164,144);
    Pack32(&p5,168,128);Pack32(&p5,172,96);
    Pack32(&p5,176,0);Pack32(&p5,180,0);
    Pack32(&p5,184,12);Pack32(&p5,188,12);
    launch(5,p5.data(),p5.size(),{16,9,1},{16,16,1});
    const auto model_guarded=ReadBuffer(gpu,model_input);
    const auto model_input_bytes=Payload(model_guarded,connected::kModelInputBytes);
    const auto auxiliary_mask_bytes=ReadTexture(gpu,auxiliary_mask);
    const auto auxiliary_vector_bytes=ReadTexture(gpu,auxiliary_vector);
    checkpoint(17,{model_input_bytes,
                   connected::Module5ActiveRows(auxiliary_mask_bytes,1),
                   connected::Module5ActiveRows(auxiliary_vector_bytes,4)});'''


FINAL = r'''std::size_t input_mismatches=0;
    input_mismatches+=Mismatches(external.source_reset,ReadTexture(gpu,source_reset));
    input_mismatches+=Mismatches(external.source_generated,ReadTexture(gpu,source_generated));
    input_mismatches+=Mismatches(external.vector_reset,ReadTexture(gpu,vector_reset));
    input_mismatches+=Mismatches(external.vector_generated,ReadTexture(gpu,vector_generated));
    input_mismatches+=Mismatches(external.scalar_reset,ReadTexture(gpu,scalar_reset));
    input_mismatches+=Mismatches(external.scalar_generated,ReadTexture(gpu,scalar_generated));
    input_mismatches+=Mismatches(connected::InitialAllocation29(),
                                 ReadTexture(gpu,allocation29));
    const auto history_a_initial=connected::InitialHistory();
    const std::vector<std::uint8_t> history_b_initial(
        connected::kHistoryBytes,poison);
    const std::vector<std::uint8_t> reset_present_initial(
        connected::kSourceBytes,poison);
    const auto outside_mismatches=
        OutsideMismatches(history_a_initial,history_a_bytes,8,0,0)+
        OutsideMismatches(history_b_initial,history_b_bytes,8,0,0)+
        OutsideMismatches(reset_present_initial,reset_present_bytes,4,
                          connected::kOriginX,connected::kOriginY);
    std::size_t canary_mismatches=CanaryMismatches(
        model_guarded,connected::kModelInputBytes);
    auto atomic_guarded=ReadBuffer(gpu,atomic);
    canary_mismatches+=CanaryMismatches(
        atomic_guarded,connected::kEarlyAtomicBytes);
    for(std::size_t i=0;i<pyramid.size();++i){
      const auto guarded=ReadBuffer(gpu,pyramid[i]);
      canary_mismatches+=CanaryMismatches(guarded,kPyramidBytes[i]);
    }
    std::size_t tensor_padding_nonzero_bytes=0;
    for(std::uint32_t y=0;y<connected::kTensorHeight;++y){
      if(y>=12u&&y<72u)continue;
      const auto begin=std::size_t{y}*connected::kTensorWidth*48u*2u;
      const auto end=begin+std::size_t{connected::kTensorWidth}*48u*2u;
      for(std::size_t index=begin;index<end;++index)
        tensor_padding_nonzero_bytes+=model_input_bytes[index]!=0u;
    }
    const std::size_t auxiliary_untouched_row_mismatches=
        connected::Module5UntouchedRowMismatches(
            auxiliary_mask_bytes,poison,1)+
        connected::Module5UntouchedRowMismatches(
            auxiliary_vector_bytes,poison,4);
    for(auto item=loaded.rbegin();item!=loaded.rend();++item){
      RequireNvapi(destroy_function(gpu.device.Get(),item->function),
                   "DestroyFunction");
      RequireNvapi(destroy_module(gpu.device.Get(),item->module),
                   "DestroyModule");
    }
    const bool pass=input_mismatches==0&&outside_mismatches==0&&
                    canary_mismatches==0&&tensor_padding_nonzero_bytes==0&&
                    auxiliary_untouched_row_mismatches==0&&
                    checkpoints.size()==15;
    Write(output_root/"allocation37.fp16",model_input_bytes);
    Write(output_root/"allocation33.r8",auxiliary_mask_bytes);
    Write(output_root/"allocation27.rg16f",auxiliary_vector_bytes);
    Write(output_root/"reset-present-region.rgba8",
          connected::FullRegion(reset_present_bytes,4));
    std::ofstream receipt(output_root/"early-run-receipt.json");
    receipt<<"{\n  \"schema\": \"neuralstudio.fg.connected-subrect-l3-l17-run.v1\",\n"
           <<"  \"backend\": \"rtx-d3d12-nvapi\",\n"
           <<"  \"run_id\": \""<<run_id<<"\",\n"
           <<"  \"process_id\": "<<GetCurrentProcessId()<<",\n"
           <<"  \"poison_byte\": "<<unsigned(poison)<<",\n"
           <<"  \"region\": [32, 24, 512, 288],\n"
           <<"  \"launch_range\": [3, 17],\n"
           <<"  \"launch_count\": 15,\n"
           <<"  \"module22_module24_status_executed\": false,\n"
           <<"  \"checkpoint_sha256\": {";
    for(std::size_t i=0;i<checkpoints.size();++i)
      receipt<<(i?",":"")<<"\n    \""<<checkpoints[i].first
             <<"\": \""<<checkpoints[i].second<<"\"";
    receipt<<"\n  },\n  \"allocation37_bytes\": "<<model_input_bytes.size()
           <<",\n  \"allocation37_sha256\": \""<<Sha256(model_input_bytes)<<"\",\n"
           <<"  \"tensor_padding_nonzero_bytes\": "<<tensor_padding_nonzero_bytes<<",\n"
           <<"  \"tensor_active_rows\": 60,\n"
           <<"  \"auxiliary_active_rows\": 120,\n"
           <<"  \"auxiliary_untouched_row_mismatched_bytes\": "
           <<auxiliary_untouched_row_mismatches<<",\n"
           <<"  \"auxiliary_mask_sha256\": \""<<Sha256(auxiliary_mask_bytes)<<"\",\n"
           <<"  \"auxiliary_vector_sha256\": \""<<Sha256(auxiliary_vector_bytes)<<"\",\n"
           <<"  \"input_mismatched_bytes\": "<<input_mismatches<<",\n"
           <<"  \"canary_mismatched_bytes\": "<<canary_mismatches<<",\n"
           <<"  \"outside_region_mismatched_bytes\": "<<outside_mismatches<<",\n"
           <<"  \"raw_parameter_origin\": [32, 24],\n"
           <<"  \"module23_mode\": 1,\n"
           <<"  \"allocation29_runtime_source\": \"generated-half-0.5\",\n"
           <<"  \"expected_or_archived_intermediate_ingress\": false,\n"
           <<"  \"address_fields_recorded\": false,\n"
           <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
    std::cout<<"allocation37_sha256="<<Sha256(model_input_bytes)<<'\n'
             <<"checkpoint_count="<<checkpoints.size()<<'\n'
             <<"result="<<(pass?"pass":"fail")<<'\n';
    return pass?0:3;'''


OUTSIDE_HELPER = r'''
std::size_t OutsideMismatches(const std::vector<std::uint8_t>& before,
                              const std::vector<std::uint8_t>& after,
                              std::uint32_t bytes_per_pixel,
                              std::uint32_t origin_x,
                              std::uint32_t origin_y){
  if(before.size()!=after.size()||
     before.size()!=std::size_t{connected::kResourceWidth}*
                       connected::kResourceHeight*bytes_per_pixel)
    throw std::runtime_error("outside comparison size drift");
  std::size_t mismatches=0;
  for(std::uint32_t y=0;y<connected::kResourceHeight;++y)
    for(std::uint32_t x=0;x<connected::kResourceWidth;++x){
      if(x>=origin_x&&x<origin_x+connected::kWidth&&
         y>=origin_y&&y<origin_y+connected::kHeight)
        continue;
      const auto offset=(std::size_t{y}*connected::kResourceWidth+x)*
                        bytes_per_pixel;
      for(std::uint32_t lane=0;lane<bytes_per_pixel;++lane)
        mismatches+=before[offset+lane]!=after[offset+lane];
    }
  return mismatches;
}
'''


def generate(source: str) -> str:
    source = once(source, 'namespace connected = fg_connected_l3_l19;',
                  'namespace connected = fg_connected_subrect_l3_l83;',
                  'namespace')
    source = once(source,
                  'constexpr std::array<std::size_t,5> kPyramidBytes={57600,86400,21600,5520,1440};',
                  'constexpr auto kPyramidBytes=connected::kEarlyPyramidBytes;',
                  'pyramid sizes')
    source = once(source, '}  // namespace\n\nint main',
                  OUTSIDE_HELPER + '\n}  // namespace\n\nint main',
                  'outside helper')
    source = source.replace('namespace connected = fg_connected_subrect_l3_l83;',
                            'namespace connected = fg_connected_subrect_l3_l83;')
    source = once(source,
                  'constexpr std::array<ModuleSpec,11> kModules={{\n'
                  '  {11,53920,"06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206"},\n'
                  '  {15,12632,"9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"},\n'
                  '  {13,13088,"fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"},\n'
                  '  {21,38472,"11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"},\n'
                  '  {23,56480,"d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d"},\n'
                  '  {25,6640,"28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"},\n'
                  '  {14,4584,"540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"},\n'
                  '  {27,4024,"c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7"},\n'
                  '  {4,4512,"c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840"},\n'
                  '  {5,14448,"3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"},\n'
                  '  {28,16512,"7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"},\n'
                  '}};',
                  'constexpr std::array<ModuleSpec,9> kModules={{\n'
                  '  {11,53920,"06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206"},\n'
                  '  {15,12632,"9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"},\n'
                  '  {13,13088,"fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"},\n'
                  '  {21,38472,"11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"},\n'
                  '  {23,56480,"d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d"},\n'
                  '  {25,6640,"28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"},\n'
                  '  {14,4584,"540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"},\n'
                  '  {4,4512,"c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840"},\n'
                  '  {5,14448,"3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"},\n'
                  '}};', 'module census')
    source = source.replace('connected::kHalfColorBytes',
                            'connected::kEarlyColorBytes')
    source = source.replace('connected::kHalfScalarBytes',
                            'connected::kEarlyMetricBytes')
    source = source.replace('connected::kDepthBytes*2u',
                            'connected::kEarlyFlowBytes')
    source = source.replace('connected::kDepthBytes',
                            'connected::kEarlyDepthBytes')
    source = source.replace('connected::kMaskBytes',
                            'connected::kEarlyMaskBytes')
    source = source.replace('connected::kVectorBytes',
                            'connected::kEarlyVectorBytes')
    source = source.replace('connected::kAtomicBytes',
                            'connected::kEarlyAtomicBytes')
    source = source.replace('connected::kAuxMaskBytes',
                            'connected::kEarlyMaskBytes')
    source = source.replace('connected::kAuxVectorBytes',
                            'connected::kEarlyVectorBytes')
    for name in ('reset_feature', 'generated_feature', 'shared_metric', 'flow',
                 'reset_depth', 'generated_depth', 'reset_mask',
                 'generated_mask', 'working_vector', 'working_mask',
                 'scatter_reset', 'scatter_generated', 'reset_projection',
                 'allocation29', 'auxiliary_mask', 'auxiliary_vector'):
        source = source.replace(
            f'auto {name}=make_texture(', f'auto {name}=make_texture(')
    source = source.replace(',320,180,8,', ',256,144,8,')
    source = source.replace(',320,180,4,', ',256,144,4,')
    source = source.replace(',320,180,2,', ',256,144,2,')
    source = source.replace(',320,180,1,', ',256,144,1,')
    model_start = '#ifdef FG_CONNECTED_FULL_CHAIN\n    auto model_input_shared='
    model_end = '    ExecuteAndReset(gpu);'
    model_replacement = '''    auto model_input=UploadBuffer(gpu,Guarded(\n        std::vector<std::uint8_t>(connected::kModelInputBytes,0u)));\n    ExecuteAndReset(gpu);'''
    source = section(source, model_start, model_end, model_replacement,
                     'model allocation')
    source = section(
        source,
        '#ifdef FG_CONNECTED_FULL_CHAIN\n    std::array<std::string,17> checkpoints{};',
        '    std::size_t input_mismatches=0;',
        EXECUTION + '\n\n    std::size_t input_mismatches=0;',
        'execution')
    final_start = 'std::size_t input_mismatches=0;'
    final_end = "    std::cout<<\"result=\"<<(pass?\"pass\":\"fail\")<<'\\n';return pass?0:3;"
    source = section(source, final_start, final_end, FINAL, 'final receipt')
    source = source.replace(
        'if(argc!=6){std::cerr<<"usage=fg-connected-l3-l19-rtx ',
        'if(argc!=6){std::cerr<<"usage=fg-connected-subrect-l3-l17-rtx ')
    source = source.replace('const auto external=connected::Generate();',
                            'const auto external=connected::GenerateEarlyExternal();')
    source = source.replace(
        'const fs::path modules=argv[1],model=argv[2],output_root=argv[3];',
        'const fs::path modules=argv[1],model=argv[2],output_root=argv[3];\n'
        '    (void)model;')
    return '// Generated canonical selected subrect RTX early host.\n' + source


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError(f'preserve existing output: {args.output}')
    result = generate(args.source.read_text())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(result)
    print(f'generated={args.output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
