#define main FgMidL55StandaloneEmbeddedMain
#include "../fg_mid_l55/rtx_oracle.cpp"
#undef main

#include "contract.h"

#ifdef FG_CONNECTED_FULL_CHAIN
#include "../fg_connected_l20_l87/rtx_shared_cuda_buffer.h"
#include "rtx_live_prefix_bridge.h"
#include "rtx_shared_texture.h"
#include "rtx_temporal_handoff.h"
int FgConnectedFullTailMain(int argc, char** argv);
#endif

namespace {

namespace connected = fg_production_contract_l3_l87;
constexpr std::array<std::size_t,5> kPyramidBytes={57600,86400,21600,5520,1440};

GpuTexture UploadTypedTexture(GpuContext& gpu,DXGI_FORMAT format,
    std::uint32_t width,std::uint32_t height,std::uint32_t bytes_per_pixel,
    D3D12_RESOURCE_FLAGS flags,D3D12_RESOURCE_STATES final_state,
    const std::vector<std::uint8_t>& bytes){
  const std::size_t source_row=std::size_t{width}*bytes_per_pixel;
  if(bytes.size()!=source_row*height)throw std::runtime_error("typed texture size drift");
  GpuTexture result;result.width=width;result.height=height;
  result.bytes_per_pixel=bytes_per_pixel;result.format=format;result.state=final_state;
  D3D12_HEAP_PROPERTIES heap{};heap.Type=D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC description{};description.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  description.Width=width;description.Height=height;description.DepthOrArraySize=1;
  description.MipLevels=1;description.Format=format;description.SampleDesc.Count=1;
  description.Layout=D3D12_TEXTURE_LAYOUT_UNKNOWN;description.Flags=flags;
  RequireHr(gpu.device->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,
      &description,D3D12_RESOURCE_STATE_COPY_DEST,nullptr,
      IID_PPV_ARGS(&result.resource)),"CreateCommittedResource(typed texture)");
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};UINT rows=0;
  UINT64 row_size=0,upload_size=0;gpu.device->GetCopyableFootprints(
      &description,0,1,0,&footprint,&rows,&row_size,&upload_size);
  auto upload=CreateBuffer(gpu.device.Get(),upload_size,D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped=nullptr;const D3D12_RANGE no_read{0,0};
  RequireHr(upload->Map(0,&no_read,&mapped),"Map(typed texture upload)");
  for(std::uint32_t row=0;row<height;++row)std::memcpy(
      static_cast<std::uint8_t*>(mapped)+std::size_t{row}*footprint.Footprint.RowPitch,
      bytes.data()+std::size_t{row}*source_row,source_row);
  upload->Unmap(0,nullptr);D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource=result.resource.Get();destination.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION source{};source.pResource=upload.Get();
  source.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;source.PlacedFootprint=footprint;
  gpu.list->CopyTextureRegion(&destination,0,0,0,&source,nullptr);
  Transition(gpu.list.Get(),result.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,final_state);
  gpu.pending.push_back(upload);return result;
}

template<std::size_t N>void Pack32(std::array<std::uint8_t,N>* bytes,
                                    std::size_t offset,std::uint32_t value){
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}
template<std::size_t N>void Pack64(std::array<std::uint8_t,N>* bytes,
                                    std::size_t offset,std::uint64_t value){
  std::memcpy(bytes->data()+offset,&value,sizeof(value));
}

std::vector<std::uint8_t> JoinValues(
    std::initializer_list<std::vector<std::uint8_t>> values){
  std::vector<std::uint8_t> result;
  for(const auto& value:values)result.insert(result.end(),value.begin(),value.end());
  return result;
}

void Restore(GpuContext& gpu,GpuTexture& texture,D3D12_RESOURCE_STATES state){
  Transition(gpu.list.Get(),texture.resource.Get(),texture.state,state);
  texture.state=state;
}
void Restore(GpuContext& gpu,GpuBuffer& buffer,D3D12_RESOURCE_STATES state){
  Transition(gpu.list.Get(),buffer.resource.Get(),buffer.state,state);
  buffer.state=state;
}

void OverwriteBuffer(GpuContext& gpu,GpuBuffer& buffer,
                     const std::vector<std::uint8_t>& bytes){
  if(bytes.size()!=buffer.size)throw std::runtime_error("buffer overwrite size drift");
  auto upload=CreateBuffer(gpu.device.Get(),bytes.size(),D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped=nullptr;const D3D12_RANGE no_read{0,0};
  RequireHr(upload->Map(0,&no_read,&mapped),"Map(buffer overwrite)");
  std::memcpy(mapped,bytes.data(),bytes.size());upload->Unmap(0,nullptr);
  Transition(gpu.list.Get(),buffer.resource.Get(),buffer.state,D3D12_RESOURCE_STATE_COPY_DEST);
  gpu.list->CopyBufferRegion(buffer.resource.Get(),0,upload.Get(),0,bytes.size());
  Transition(gpu.list.Get(),buffer.resource.Get(),D3D12_RESOURCE_STATE_COPY_DEST,
             D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
  buffer.state=D3D12_RESOURCE_STATE_UNORDERED_ACCESS;gpu.pending.push_back(upload);
  ExecuteAndReset(gpu);
}

std::string ModuleName(std::uint32_t token){
  std::ostringstream stream;stream<<"cu-module-"<<std::setfill('0')
      <<std::setw(6)<<token<<".bin";return stream.str();
}

struct ModuleSpec{std::uint32_t token;std::size_t bytes;const char* sha;};
constexpr std::array<ModuleSpec,11> kModules={{
  {11,53920,"06eee29ec33202996f97c063a08ccdc767d0167252ad03b5ed457c941cb85206"},
  {15,12632,"9e772df29e9434e4eff1edf1cde32706ea641c2b0156a459a080565a7483f381"},
  {13,13088,"fd0665969da887dd92540b673009e0361a6c1de74fe54badf48f300b1fccbbf1"},
  {21,38472,"11df926493fca0a981d560427b36c4dcf489ccb84d794501a70bda8d414ea8a2"},
  {23,56480,"d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d"},
  {25,6640,"28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824"},
  {14,4584,"540f686ffcd4508212b19296992a69ed81ca95772fd229d0a252905c8015419c"},
  {27,4024,"c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7"},
  {4,4512,"c11cd7474562e20762d272162c94378a17f8f77c7de8206d695e6c469383e840"},
  {5,14448,"3f2308a693730b44c00e064ed25dab3b73a420f2e92d215ef733c4e10a30439c"},
  {28,16512,"7f9c1f110e8ec9347aa968bc9019e5ffc8839057987a2e5676a0f64637c8df2d"},
}};

}  // namespace

int main(int argc,char** argv){
  if(argc!=6){std::cerr<<"usage=fg-production-contract-l3-l87-rtx <modules> <model> <output> <run-id> <poison>\n";return 2;}
  try{
    const fs::path modules=argv[1],model=argv[2],output_root=argv[3];
    fs::create_directories(output_root);
    fs::create_directories(output_root/"generated-1");
    fs::create_directories(output_root/"generated-2");
    const std::string run_id=argv[4];const auto poison=ParsePoison(argv[5]);
    const auto external=connected::Generate();
    GpuContext gpu=CreateGpuContext();
    UniqueLibrary nvapi(LoadLibraryW(L"nvapi64.dll"));
    if(!nvapi)throw std::runtime_error("nvapi64.dll unavailable");
    const auto query=reinterpret_cast<QueryInterface>(
        GetProcAddress(nvapi.get(),"nvapi_QueryInterface"));
    const auto initialize=Resolve<Initialize>(query,kIdInitialize,"Initialize");
    const auto get_texture=Resolve<GetMergedTextureSampler>(query,kIdGetMergedTextureSampler,"GetTexture");
    const auto get_surface=Resolve<GetIndependentDescriptor>(query,kIdGetIndependentDescriptor,"GetSurface");
    const auto create_module=Resolve<CreateCuModule>(query,kIdCreateCuModule,"CreateModule");
    const auto create_function=Resolve<CreateCuFunction>(query,kIdCreateCuFunction,"CreateFunction");
    const auto launch_chain=Resolve<LaunchKernelChain>(query,kIdLaunchKernelChain,"Launch");
    const auto destroy_module=Resolve<DestroyCuModule>(query,kIdDestroyCuModule,"DestroyModule");
    const auto destroy_function=Resolve<DestroyCuFunction>(query,kIdDestroyCuFunction,"DestroyFunction");
    RequireNvapi(initialize(),"NvAPI_Initialize");

    D3D12_DESCRIPTOR_HEAP_DESC resource_description{};
    resource_description.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    resource_description.NumDescriptors=128;
    resource_description.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    resource_description.NodeMask=1;
    ComPtr<ID3D12DescriptorHeap> resource_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&resource_description,
        IID_PPV_ARGS(&resource_heap)),"resource heap");
    D3D12_DESCRIPTOR_HEAP_DESC sampler_description{};
    sampler_description.Type=D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;
    sampler_description.NumDescriptors=3;
    sampler_description.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    sampler_description.NodeMask=1;
    ComPtr<ID3D12DescriptorHeap> sampler_heap;
    RequireHr(gpu.device->CreateDescriptorHeap(&sampler_description,
        IID_PPV_ARGS(&sampler_heap)),"sampler heap");
    const auto resource_start=resource_heap->GetCPUDescriptorHandleForHeapStart();
    const auto sampler_start=sampler_heap->GetCPUDescriptorHandleForHeapStart();
    const UINT resource_increment=gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    const UINT sampler_increment=gpu.device->GetDescriptorHandleIncrementSize(
        D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER);
    auto resource_slot=[&](UINT i){return D3D12_CPU_DESCRIPTOR_HANDLE{
        resource_start.ptr+std::uint64_t{i}*resource_increment};};
    auto sampler_slot=[&](UINT i){return D3D12_CPU_DESCRIPTOR_HANDLE{
        sampler_start.ptr+std::uint64_t{i}*sampler_increment};};
    D3D12_SAMPLER_DESC point_border{};point_border.Filter=D3D12_FILTER_MIN_MAG_MIP_POINT;
    point_border.AddressU=point_border.AddressV=point_border.AddressW=D3D12_TEXTURE_ADDRESS_MODE_BORDER;
    gpu.device->CreateSampler(&point_border,sampler_slot(0));
    D3D12_SAMPLER_DESC point_clamp=point_border;point_clamp.AddressU=point_clamp.AddressV=point_clamp.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;
    gpu.device->CreateSampler(&point_clamp,sampler_slot(1));
    D3D12_SAMPLER_DESC linear_clamp=point_clamp;linear_clamp.Filter=D3D12_FILTER_MIN_MAG_MIP_LINEAR;linear_clamp.MaxAnisotropy=1;
    gpu.device->CreateSampler(&linear_clamp,sampler_slot(2));
    UINT descriptor_index=0;
    auto texture_handle=[&](GpuTexture& texture,UINT sampler_index){
      const UINT slot=descriptor_index++;
      D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=texture.format;
      view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;
      view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
      view.Texture2D.MipLevels=1;gpu.device->CreateShaderResourceView(
          texture.resource.Get(),&view,resource_slot(slot));
      MergedTextureSamplerParams request{};request.struct_size_in=request.struct_size_out=sizeof(request);
      request.device=gpu.device.Get();request.texture_descriptor.ptr=resource_slot(slot).ptr;
      request.sampler_descriptor.ptr=sampler_slot(sampler_index).ptr;
      RequireNvapi(get_texture(&request),"texture handle");return request.texture_handle;
    };
    auto surface_handle=[&](GpuTexture& texture){
      const UINT slot=descriptor_index++;
      D3D12_UNORDERED_ACCESS_VIEW_DESC view{};view.Format=texture.format;
      view.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;
      gpu.device->CreateUnorderedAccessView(texture.resource.Get(),nullptr,&view,resource_slot(slot));
      IndependentDescriptorParams request{};request.struct_size_in=request.struct_size_out=sizeof(request);
      request.device=gpu.device.Get();request.type=IndependentDescriptorType::kSurface;
      request.descriptor.ptr=resource_slot(slot).ptr;RequireNvapi(get_surface(&request),"surface handle");
      return request.handle;
    };

    auto make_texture=[&](DXGI_FORMAT format,std::uint32_t w,std::uint32_t h,
                          std::uint32_t bpp,D3D12_RESOURCE_STATES state,
                          const std::vector<std::uint8_t>& bytes){
      return UploadTypedTexture(gpu,format,w,h,bpp,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,state,bytes);
    };
    auto source_reset=UploadTypedTexture(gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
        D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.source_reset);
    auto source_generated=UploadTypedTexture(gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
        D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.source_generated);
    auto source_generated2=UploadTypedTexture(gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,
        D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.source_generated2);
    auto history_a=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialHistory());
    auto history_b=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,640,360,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHistoryBytes,poison));
    auto reset_feature=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,320,180,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHalfColorBytes,poison));
    auto generated_feature=make_texture(DXGI_FORMAT_R16G16B16A16_FLOAT,320,180,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHalfColorBytes,poison));
    auto shared_metric=make_texture(DXGI_FORMAT_R16_FLOAT,320,180,2,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kHalfScalarBytes,poison));
    auto vector_reset=make_texture(DXGI_FORMAT_R16G16_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.vector_reset);
    auto vector_generated=make_texture(DXGI_FORMAT_R16G16_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.vector_generated);
    auto vector_generated2=make_texture(DXGI_FORMAT_R16G16_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.vector_generated2);
    auto scalar_reset=make_texture(DXGI_FORMAT_R32_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.scalar_reset);
    auto scalar_generated=make_texture(DXGI_FORMAT_R32_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.scalar_generated);
    auto scalar_generated2=make_texture(DXGI_FORMAT_R32_FLOAT,640,360,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,external.scalar_generated2);
    std::vector<connected::RenderedFrameInput> additional_frames;
    std::vector<GpuTexture> additional_source;
    std::vector<GpuTexture> additional_motion;
    std::vector<GpuTexture> additional_depth;
    additional_frames.reserve(13);additional_source.reserve(13);
    additional_motion.reserve(13);additional_depth.reserve(13);
    for(std::uint32_t frame=3;frame<16;++frame){
      additional_frames.push_back(connected::GenerateRenderedFrame(frame));
      const auto& input=additional_frames.back();
      additional_source.push_back(UploadTypedTexture(
          gpu,DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,input.source));
      additional_motion.push_back(make_texture(
          DXGI_FORMAT_R16G16_FLOAT,640,360,4,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,input.motion));
      additional_depth.push_back(make_texture(
          DXGI_FORMAT_R32_FLOAT,640,360,4,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,input.depth));
    }
    auto flow=make_texture(DXGI_FORMAT_R32G32_FLOAT,320,180,8,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes*2u,poison));
    auto reset_depth=make_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kDepthBytes,poison));
    auto generated_depth=make_texture(DXGI_FORMAT_R32_UINT,320,180,4,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialGeneratedDepth());
    auto reset_mask=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kMaskBytes,poison));
    auto generated_mask=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialGeneratedMask());
    auto working_vector=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));
    auto working_mask=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kMaskBytes,poison));
    auto scatter_reset=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kMaskBytes,255));
    auto scatter_generated=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kMaskBytes,255));
    auto reset_projection=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kVectorBytes,poison));
#ifdef FG_CONNECTED_FULL_CHAIN
    auto allocation29_shared=connected::UploadSharedTexture(
        gpu,DXGI_FORMAT_R16_FLOAT,320,180,2,
        D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
        connected::InitialAllocation29());
    GpuTexture& allocation29=allocation29_shared.d3d;
#else
    auto allocation29=make_texture(DXGI_FORMAT_R16_FLOAT,320,180,2,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,connected::InitialAllocation29());
#endif
    auto auxiliary_mask=make_texture(DXGI_FORMAT_R8_UNORM,320,180,1,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxMaskBytes,poison));
    auto auxiliary_vector=make_texture(DXGI_FORMAT_R16G16_FLOAT,320,180,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kAuxVectorBytes,poison));
    auto reset_present=make_texture(DXGI_FORMAT_R8G8B8A8_UNORM,640,360,4,D3D12_RESOURCE_STATE_UNORDERED_ACCESS,std::vector<std::uint8_t>(connected::kSourceBytes,poison));
    auto atomic=UploadBuffer(gpu,Guarded(connected::InitialAtomic()));
    std::array<GpuBuffer,5> pyramid;
    for(std::size_t i=0;i<pyramid.size();++i)pyramid[i]=UploadBuffer(gpu,Guarded(std::vector<std::uint8_t>(kPyramidBytes[i],poison)));
#ifdef FG_CONNECTED_FULL_CHAIN
    auto model_input_shared=fg_connected_l20_l87::UploadSharedCudaBuffer(
        gpu,Guarded(std::vector<std::uint8_t>(connected::kModelInputBytes,poison)));
    GpuBuffer& model_input=model_input_shared.d3d;
#else
    auto model_input_storage=UploadBuffer(gpu,Guarded(
        std::vector<std::uint8_t>(connected::kModelInputBytes,poison)));
    GpuBuffer& model_input=model_input_storage;
#endif
    auto status_buffer=UploadBuffer(gpu,Guarded(std::vector<std::uint8_t>(4,poison)));
    const auto l20_weight_bytes=Read(model/"default-token-000063.bin");
    if(l20_weight_bytes.size()!=56320||Sha256(l20_weight_bytes)!=
       "44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9")
      throw std::runtime_error("L20 weight identity mismatch");
    auto l20_weight=UploadBuffer(gpu,Guarded(l20_weight_bytes));
    auto l20_output=UploadBuffer(gpu,Guarded(
        std::vector<std::uint8_t>(connected::kL20OutputBytes,poison)));
    ExecuteAndReset(gpu);

    const auto source_reset_tex=texture_handle(source_reset,2);
    const auto source_generated_tex=texture_handle(source_generated,0);
    const auto source_generated2_tex=texture_handle(source_generated2,0);
    const auto history_a_tex=texture_handle(history_a,0);
    const auto history_a_surf=surface_handle(history_a);
    const auto history_b_tex=texture_handle(history_b,0);
    const auto history_b_surf=surface_handle(history_b);
    const auto reset_feature_tex=texture_handle(reset_feature,0);
    const auto reset_feature_surf=surface_handle(reset_feature);
    const auto generated_feature_tex=texture_handle(generated_feature,0);
    const auto generated_feature_surf=surface_handle(generated_feature);
    const auto shared_metric_tex=texture_handle(shared_metric,0);
    const auto shared_metric_surf=surface_handle(shared_metric);
    const auto vector_reset_tex=texture_handle(vector_reset,1);
    const auto vector_generated_tex=texture_handle(vector_generated,1);
    const auto vector_generated2_tex=texture_handle(vector_generated2,1);
    const auto scalar_reset_tex=texture_handle(scalar_reset,0);
    const auto scalar_generated_tex=texture_handle(scalar_generated,0);
    const auto scalar_generated2_tex=texture_handle(scalar_generated2,0);
    std::array<std::uint64_t,16> frame_source_handle{};
    std::array<std::uint64_t,16> frame_motion_handle{};
    std::array<std::uint64_t,16> frame_depth_handle{};
    frame_source_handle[0]=source_reset_tex;
    frame_source_handle[1]=source_generated_tex;
    frame_source_handle[2]=source_generated2_tex;
    frame_motion_handle[0]=vector_reset_tex;
    frame_motion_handle[1]=vector_generated_tex;
    frame_motion_handle[2]=vector_generated2_tex;
    frame_depth_handle[0]=scalar_reset_tex;
    frame_depth_handle[1]=scalar_generated_tex;
    frame_depth_handle[2]=scalar_generated2_tex;
    for(std::size_t i=0;i<additional_frames.size();++i){
      const auto frame=i+3u;
      frame_source_handle[frame]=texture_handle(additional_source[i],0);
      frame_motion_handle[frame]=texture_handle(additional_motion[i],1);
      frame_depth_handle[frame]=texture_handle(additional_depth[i],0);
    }
    const auto flow_tex=texture_handle(flow,0);
    const auto flow_surf=surface_handle(flow);
    const auto reset_depth_tex=texture_handle(reset_depth,0);
    const auto reset_depth_surf=surface_handle(reset_depth);
    const auto generated_depth_tex=texture_handle(generated_depth,0);
    const auto generated_depth_surf=surface_handle(generated_depth);
    const auto reset_mask_tex=texture_handle(reset_mask,0);
    const auto reset_mask_surf=surface_handle(reset_mask);
    const auto generated_mask_tex=texture_handle(generated_mask,0);
    const auto generated_mask_surf=surface_handle(generated_mask);
    const auto working_vector_tex=texture_handle(working_vector,0);
    const auto working_vector_surf=surface_handle(working_vector);
    const auto working_mask_tex=texture_handle(working_mask,0);
    const auto working_mask_surf=surface_handle(working_mask);
    const auto scatter_reset_tex=texture_handle(scatter_reset,0);
    const auto scatter_reset_surf=surface_handle(scatter_reset);
    const auto scatter_generated_tex=texture_handle(scatter_generated,0);
    const auto scatter_generated_surf=surface_handle(scatter_generated);
    const auto reset_projection_tex=texture_handle(reset_projection,0);
    const auto reset_projection_surf=surface_handle(reset_projection);
    const auto allocation29_tex=texture_handle(allocation29,0);
    const auto auxiliary_mask_surf=surface_handle(auxiliary_mask);
    const auto auxiliary_vector_tex=texture_handle(auxiliary_vector,0);
    const auto auxiliary_vector_surf=surface_handle(auxiliary_vector);
    const auto reset_present_surf=surface_handle(reset_present);

    struct Loaded{NvObjectHandle module=nullptr,function=nullptr;};
    std::array<Loaded,kModules.size()> loaded{};
    for(std::size_t i=0;i<kModules.size();++i){
      const auto bytes=Read(modules/ModuleName(kModules[i].token));
      if(bytes.size()!=kModules[i].bytes||Sha256(bytes)!=kModules[i].sha)
        throw std::runtime_error("early module identity mismatch");
      RequireNvapi(create_module(gpu.device.Get(),bytes.data(),
          static_cast<std::uint32_t>(bytes.size()),&loaded[i].module),"CreateModule");
      const char* function_name=kModules[i].token==28?
          "k_conv_fp16_nhwc":"main_kernel";
      RequireNvapi(create_function(gpu.device.Get(),loaded[i].module,
          function_name,&loaded[i].function),"CreateFunction");
    }
    auto function_for=[&](std::uint32_t token){
      for(std::size_t i=0;i<kModules.size();++i)
        if(kModules[i].token==token)return loaded[i].function;
      throw std::runtime_error("module function missing");
    };
    auto launch=[&](std::uint32_t token,const void* parameters,std::size_t bytes,
                    Dim3 grid,Dim3 block){
      const KernelLaunchParams descriptor{function_for(token),grid,block,0,parameters,
                                           static_cast<std::uint32_t>(bytes)};
      RequireNvapi(launch_chain(gpu.list.Get(),&descriptor,1),"LaunchKernelChain");
      ExecuteAndReset(gpu);
    };
    auto snapshot_texture=[&](GpuTexture& texture,D3D12_RESOURCE_STATES next){
      auto bytes=ReadTexture(gpu,texture);Restore(gpu,texture,next);ExecuteAndReset(gpu);
      return bytes;
    };
    auto snapshot_buffer=[&](GpuBuffer& buffer,std::size_t bytes){
      auto guarded=ReadBuffer(gpu,buffer);Restore(gpu,buffer,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);return Payload(guarded,bytes);
    };
#ifdef FG_CONNECTED_FULL_CHAIN
    std::array<std::string,129> checkpoints{};
#else
    std::array<std::string,18> checkpoints{};
#endif
    auto encode_l3=[&](std::uint64_t source,std::uint64_t history,
                       std::uint64_t feature,std::uint64_t next_history){
      std::array<std::uint8_t,168> p{};Pack64(&p,0,source);Pack64(&p,80,history);
      Pack64(&p,96,feature);Pack64(&p,104,shared_metric_surf);
      Pack64(&p,120,next_history);Pack32(&p,128,2);Pack32(&p,136,640);
      Pack32(&p,140,360);Pack32(&p,144,320);Pack32(&p,148,180);
      launch(11,p.data(),p.size(),{40,23,1},{256,1,1});
    };
    encode_l3(source_reset_tex,history_a_tex,reset_feature_surf,history_b_surf);
    auto reset_feature_bytes=snapshot_texture(reset_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto reset_metric_bytes=snapshot_texture(shared_metric,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto history_b_bytes=snapshot_texture(history_b,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[0]=Sha256(JoinValues({reset_feature_bytes,reset_metric_bytes,history_b_bytes}));

    auto encode_l4=[&](bool reset,std::uint64_t vector,std::uint64_t scalar,
                       std::uint64_t depth){
      std::array<std::uint8_t,160> p{};Pack64(&p,0,vector);Pack32(&p,16,640);
      Pack32(&p,20,360);Pack32(&p,24,0x3acccccd);Pack32(&p,28,0x3b360b61);
      Pack64(&p,32,flow_surf);Pack64(&p,40,scalar);Pack32(&p,56,640);
      Pack32(&p,60,360);Pack32(&p,64,0x3acccccd);Pack32(&p,68,0x3b360b61);
      Pack32(&p,72,0x42200000);Pack32(&p,76,0x3f800000);Pack32(&p,80,0x44160000);
      p[84]=reset?127:0;Pack64(&p,88,depth);Pack32(&p,144,320);
      Pack32(&p,148,180);Pack32(&p,152,0x3623d70a);Pack32(&p,156,0x3701742e);
      launch(15,p.data(),p.size(),{20,12,1},{16,16,1});
    };
    encode_l4(true,vector_reset_tex,scalar_reset_tex,reset_depth_surf);
    auto flow_reset_bytes=snapshot_texture(flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto reset_depth_bytes=snapshot_texture(reset_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[1]=Sha256(JoinValues({flow_reset_bytes,reset_depth_bytes}));

    auto encode_l5=[&](std::uint64_t depth,std::uint64_t mask){
      std::array<std::uint8_t,144> p{};Pack64(&p,0,flow_tex);Pack64(&p,8,depth);
      Pack64(&p,16,atomic.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,24,mask);Pack32(&p,40,0x3acccccd);Pack32(&p,44,0x3b360b61);
      Pack32(&p,48,320);Pack32(&p,52,180);Pack32(&p,56,0x43100000);
      Pack32(&p,60,0x414f5c29);Pack32(&p,64,0x42200000);Pack32(&p,68,0x3f800000);
      Pack32(&p,72,0x44160000);const std::array<float,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
      std::memcpy(p.data()+80,identity.data(),64);
      launch(13,p.data(),p.size(),{20,12,1},{16,16,1});
    };
    encode_l5(reset_depth_tex,reset_mask_surf);
    auto atomic_reset_bytes=snapshot_buffer(atomic,connected::kAtomicBytes);
    auto reset_mask_bytes=snapshot_texture(reset_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[2]=Sha256(JoinValues({atomic_reset_bytes,reset_mask_bytes}));

    auto encode_l6=[&](std::uint64_t depth,std::uint64_t mask){
      std::array<std::uint8_t,224> p{};Pack64(&p,0,flow_tex);Pack64(&p,8,depth);
      Pack64(&p,16,mask);Pack64(&p,24,atomic.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,32,working_vector_surf);Pack64(&p,40,working_mask_surf);
      for(std::size_t i=0;i<pyramid.size();++i)
        Pack64(&p,48+i*8,pyramid[i].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack32(&p,128,std::bit_cast<std::uint32_t>(1.0f/640.0f));
      Pack32(&p,132,std::bit_cast<std::uint32_t>(1.0f/360.0f));
      Pack32(&p,136,320);Pack32(&p,140,180);Pack32(&p,144,0x42200000);
      Pack32(&p,148,0x3f800000);Pack32(&p,152,0x44160000);
      const std::array<float,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
      std::memcpy(p.data()+160,identity.data(),64);
      launch(21,p.data(),p.size(),{20,12,1},{256,1,1});
    };
    encode_l6(generated_depth_tex,generated_mask_tex);
    auto working_vector_l6=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto working_mask_l6=snapshot_texture(working_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    std::array<std::vector<std::uint8_t>,5> pyramid_l6;
    for(std::size_t i=0;i<pyramid.size();++i)pyramid_l6[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
    checkpoints[3]=Sha256(JoinValues({working_vector_l6,working_mask_l6,
        pyramid_l6[0],pyramid_l6[1],pyramid_l6[2],pyramid_l6[3],pyramid_l6[4]}));
    auto encode_l7=[&]{
      std::array<std::uint8_t,72> p{};Pack64(&p,8,pyramid[1].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,16,pyramid[2].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,24,pyramid[3].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&p,32,pyramid[4].resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack32(&p,40,320);Pack32(&p,44,180);Pack32(&p,48,1);
      Pack64(&p,56,working_vector_surf);
      Pack64(&p,64,pyramid[0].resource->GetGPUVirtualAddress()+kGuardBytes);
      launch(23,p.data(),p.size(),{20,12,1},{256,1,1});
    };
    encode_l7();
    auto working_vector_l7=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[4]=Sha256(working_vector_l7);
    auto encode_l8=[&](std::uint64_t scatter){
      std::array<std::uint8_t,48> p{};Pack64(&p,0,working_vector_tex);Pack64(&p,8,scatter);
      Pack32(&p,16,0);Pack32(&p,20,320);Pack32(&p,24,180);
      Pack32(&p,28,0x3b4ccccd);Pack32(&p,32,0x3bb60b61);
      Pack32(&p,36,320);Pack32(&p,40,180);Pack32(&p,44,0x414f5c29);
      launch(25,p.data(),p.size(),{20,12,1},{16,16,1});
    };
    encode_l8(scatter_reset_surf);
    auto scatter_reset_bytes=snapshot_texture(scatter_reset,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[5]=Sha256(scatter_reset_bytes);
    std::array<std::uint8_t,32> p9{};Pack64(&p9,0,reset_feature_tex);
    Pack64(&p9,8,scatter_reset_tex);Pack64(&p9,16,reset_projection_surf);
    Pack32(&p9,24,320);Pack32(&p9,28,180);
    launch(14,p9.data(),p9.size(),{20,12,1},{16,16,1});
    auto reset_projection_bytes=snapshot_texture(reset_projection,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[6]=Sha256(reset_projection_bytes);
    auto encode_status=[&](std::uint32_t value){
      std::array<std::uint8_t,16> p{};
      Pack64(&p,0,status_buffer.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack32(&p,8,1);Pack32(&p,12,value);
      launch(27,p.data(),p.size(),{1,1,1},{64,1,1});
    };
    encode_status(1);checkpoints[7]=Sha256(snapshot_buffer(status_buffer,4));
    std::array<std::uint8_t,40> p11{};Pack64(&p11,0,source_reset_tex);
    Pack64(&p11,16,reset_present_surf);Pack32(&p11,32,640);Pack32(&p11,36,360);
    launch(4,p11.data(),p11.size(),{40,23,1},{16,16,1});
    auto reset_present_bytes=snapshot_texture(reset_present,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    checkpoints[8]=Sha256(reset_present_bytes);
    encode_status(0);checkpoints[9]=Sha256(snapshot_buffer(status_buffer,4));
    Restore(gpu,history_a,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l3(source_generated_tex,history_b_tex,generated_feature_surf,history_a_surf);
    auto generated_feature_bytes=snapshot_texture(generated_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto generated_metric_bytes=snapshot_texture(shared_metric,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto history_a_bytes=snapshot_texture(history_a,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    checkpoints[10]=Sha256(JoinValues({generated_feature_bytes,generated_metric_bytes,history_a_bytes}));
    Restore(gpu,flow,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,generated_depth,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l4(false,vector_generated_tex,scalar_generated_tex,generated_depth_surf);
    auto flow_generated_bytes=snapshot_texture(flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto generated_depth_bytes=snapshot_texture(generated_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[11]=Sha256(JoinValues({flow_generated_bytes,generated_depth_bytes}));
    Write(output_root/"generated-1"/"l14-flow.rg32f",flow_generated_bytes);
    Write(output_root/"generated-1"/"l14-depth.r32u",generated_depth_bytes);
    OverwriteBuffer(gpu,atomic,Guarded(connected::InitialAtomic()));
    Restore(gpu,generated_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l5(generated_depth_tex,generated_mask_surf);
    auto atomic_generated_bytes=snapshot_buffer(atomic,connected::kAtomicBytes);
    auto generated_mask_bytes=snapshot_texture(generated_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[12]=Sha256(JoinValues({atomic_generated_bytes,generated_mask_bytes}));
    Restore(gpu,working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l6(reset_depth_tex,reset_mask_tex);
    auto working_vector_l16=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto working_mask_l16=snapshot_texture(working_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    std::array<std::vector<std::uint8_t>,5> pyramid_l16;
    for(std::size_t i=0;i<pyramid.size();++i)pyramid_l16[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
    checkpoints[13]=Sha256(JoinValues({working_vector_l16,working_mask_l16,
        pyramid_l16[0],pyramid_l16[1],pyramid_l16[2],pyramid_l16[3],pyramid_l16[4]}));
    encode_l7();
    auto working_vector_l17=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[14]=Sha256(working_vector_l17);
    encode_l8(scatter_generated_surf);
    auto scatter_generated_bytes=snapshot_texture(scatter_generated,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[15]=Sha256(scatter_generated_bytes);
    auto encode_l19=[&](const std::array<std::uint64_t,10>& inputs,
                        std::uint64_t vector_surface){
      std::array<std::uint8_t,192> parameters{};
      for(std::size_t i=0;i<inputs.size();++i)
        Pack64(&parameters,i*8,inputs[i]);
      Pack64(&parameters,80,model_input.resource->GetGPUVirtualAddress()+kGuardBytes);
      Pack64(&parameters,88,auxiliary_mask_surf);
      Pack64(&parameters,96,vector_surface);
      Pack32(&parameters,152,320);Pack32(&parameters,156,180);
      Pack32(&parameters,160,320);Pack32(&parameters,164,180);
      Pack32(&parameters,168,160);Pack32(&parameters,172,96);
      Pack32(&parameters,184,3);Pack32(&parameters,188,3);
      launch(5,parameters.data(),parameters.size(),{20,12,1},{16,16,1});
    };
    const std::array<std::uint64_t,10> l19_inputs_frame1={
      reset_feature_tex,generated_feature_tex,scatter_generated_tex,
      scatter_reset_tex,reset_projection_tex,shared_metric_tex,
      allocation29_tex,flow_tex,working_vector_tex,working_mask_tex};
    encode_l19(l19_inputs_frame1,auxiliary_vector_surf);
    auto model_frame1_guarded=ReadBuffer(gpu,model_input);
    auto model_input_frame1_bytes=Payload(model_frame1_guarded,connected::kModelInputBytes);
    auto auxiliary_mask_frame1_bytes=ReadTexture(gpu,auxiliary_mask);
    auto auxiliary_vector_frame1_bytes=ReadTexture(gpu,auxiliary_vector);
    checkpoints[16]=Sha256(JoinValues({model_input_frame1_bytes,
        auxiliary_mask_frame1_bytes,auxiliary_vector_frame1_bytes}));
#ifdef FG_CONNECTED_FULL_CHAIN
    std::string tail_program="fg-connected-l20-l87-live";
    std::string modules_argument=modules.string();
    std::string model_argument=model.string();
    std::string poison_argument=argv[5];
    auto run_tail=[&](const std::string& frame_name){
      const fs::path tail_output=output_root/frame_name/"l20-l87";
      std::string tail_output_argument=tail_output.string();
      std::string l22_debug_argument=(tail_output/"l22-debug.bin").string();
      std::string frame_run_id=run_id+"-"+frame_name;
      std::array<char*,6> tail_argv={
          tail_program.data(),modules_argument.data(),model_argument.data(),
          tail_output_argument.data(),frame_run_id.data(),poison_argument.data()};
      FgConnectedSetRtxLivePrefixInput(static_cast<std::uint64_t>(
          model_input_shared.cuda_ptr+kGuardBytes));
      FgConnectedSetRtxLivePrefixDebugPath(l22_debug_argument.c_str());
      FgConnectedSetRtxAllocation29Handle(reinterpret_cast<std::uint64_t>(
          allocation29_shared.shared_handle));
      const int status=FgConnectedFullTailMain(
          static_cast<int>(tail_argv.size()),tail_argv.data());
      FgConnectedSetRtxLivePrefixInput(0);
      FgConnectedSetRtxLivePrefixDebugPath(nullptr);
      FgConnectedSetRtxAllocation29Handle(0);
      if(status!=0)throw std::runtime_error(
          std::string("connected L20-L87 tail failed: ")+frame_name);
      allocation29.state=D3D12_RESOURCE_STATE_COPY_SOURCE;
    };
    run_tail("generated-1");
    auto allocation29_after_frame1=ReadTexture(gpu,allocation29);

    encode_status(0);checkpoints[17]=Sha256(snapshot_buffer(status_buffer,4));
    Restore(gpu,history_b,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,reset_feature,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,shared_metric,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l3(source_generated2_tex,history_a_tex,reset_feature_surf,history_b_surf);
    auto reset_feature_frame2=snapshot_texture(reset_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto metric_frame2=snapshot_texture(shared_metric,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto history_b_frame2=snapshot_texture(history_b,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[18]=Sha256(JoinValues({reset_feature_frame2,metric_frame2,history_b_frame2}));
    Restore(gpu,flow,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,reset_depth,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l4(false,vector_generated2_tex,scalar_generated2_tex,reset_depth_surf);
    auto flow_frame2=snapshot_texture(flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    auto reset_depth_frame2=snapshot_texture(reset_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[19]=Sha256(JoinValues({flow_frame2,reset_depth_frame2}));
    Write(output_root/"generated-2"/"l14-flow.rg32f",flow_frame2);
    Write(output_root/"generated-2"/"l14-depth.r32u",reset_depth_frame2);
    OverwriteBuffer(gpu,atomic,Guarded(connected::InitialAtomic()));
    Restore(gpu,reset_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l5(reset_depth_tex,reset_mask_surf);
    auto atomic_frame2=snapshot_buffer(atomic,connected::kAtomicBytes);
    auto reset_mask_frame2=snapshot_texture(reset_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[20]=Sha256(JoinValues({atomic_frame2,reset_mask_frame2}));
    Restore(gpu,working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l6(generated_depth_tex,generated_mask_tex);
    auto working_vector_frame2=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    auto working_mask_frame2=snapshot_texture(working_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    std::array<std::vector<std::uint8_t>,5> pyramid_frame2;
    for(std::size_t i=0;i<pyramid.size();++i)
      pyramid_frame2[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
    checkpoints[21]=Sha256(JoinValues({working_vector_frame2,working_mask_frame2,
        pyramid_frame2[0],pyramid_frame2[1],pyramid_frame2[2],
        pyramid_frame2[3],pyramid_frame2[4]}));
    encode_l7();
    auto working_vector_l17_frame2=snapshot_texture(
        working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[22]=Sha256(working_vector_l17_frame2);
    Restore(gpu,scatter_reset,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    encode_l8(scatter_reset_surf);
    auto scatter_reset_frame2=snapshot_texture(
        scatter_reset,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    checkpoints[23]=Sha256(scatter_reset_frame2);
    Restore(gpu,auxiliary_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,reset_projection,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,auxiliary_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    Restore(gpu,model_input,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    Restore(gpu,allocation29,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    ExecuteAndReset(gpu);
    const std::array<std::uint64_t,10> l19_inputs_frame2={
      generated_feature_tex,reset_feature_tex,scatter_reset_tex,
      scatter_generated_tex,auxiliary_vector_tex,shared_metric_tex,
      allocation29_tex,flow_tex,working_vector_tex,working_mask_tex};
    encode_l19(l19_inputs_frame2,reset_projection_surf);
    auto model_frame2_guarded=ReadBuffer(gpu,model_input);
    auto model_input_frame2_bytes=Payload(model_frame2_guarded,connected::kModelInputBytes);
    auto auxiliary_mask_frame2_bytes=ReadTexture(gpu,auxiliary_mask);
    auto auxiliary_vector_frame2_bytes=ReadTexture(gpu,reset_projection);
    checkpoints[24]=Sha256(JoinValues({model_input_frame2_bytes,
        auxiliary_mask_frame2_bytes,auxiliary_vector_frame2_bytes}));
    run_tail("generated-2");
    auto allocation29_after_frame2=ReadTexture(gpu,allocation29);
    std::vector<std::string> allocation37_sha256={
        Sha256(model_input_frame1_bytes),Sha256(model_input_frame2_bytes)};
    std::vector<std::string> allocation29_sha256={
        Sha256(allocation29_after_frame1),Sha256(allocation29_after_frame2)};
    std::vector<std::uint8_t> final_model_before_tail=model_input_frame2_bytes;
    std::size_t extra_model_poison_halfwords=0;
    std::size_t extra_state_poison_halfwords=0;
    std::size_t extra_canary_mismatches=0;
    for(std::uint32_t frame=3;frame<16;++frame){
      const bool odd=(frame&1u)!=0u;
      const std::size_t base=9u+std::size_t{frame-1u}*8u;
      encode_status(0);checkpoints[base]=Sha256(snapshot_buffer(status_buffer,4));
      GpuTexture& previous_history=odd?history_b:history_a;
      GpuTexture& next_history=odd?history_a:history_b;
      GpuTexture& current_feature=odd?generated_feature:reset_feature;
      GpuTexture& previous_feature=odd?reset_feature:generated_feature;
      const auto previous_history_tex=odd?history_b_tex:history_a_tex;
      const auto next_history_surf=odd?history_a_surf:history_b_surf;
      const auto current_feature_surf=odd?generated_feature_surf:reset_feature_surf;
      Restore(gpu,previous_history,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      Restore(gpu,next_history,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,current_feature,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,shared_metric,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);
      encode_l3(frame_source_handle[frame],previous_history_tex,
                current_feature_surf,next_history_surf);
      auto feature_bytes=snapshot_texture(current_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      auto metric_bytes=snapshot_texture(shared_metric,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      auto history_bytes=snapshot_texture(next_history,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      checkpoints[base+1u]=Sha256(JoinValues({feature_bytes,metric_bytes,history_bytes}));
      GpuTexture& current_depth=odd?generated_depth:reset_depth;
      GpuTexture& previous_depth=odd?reset_depth:generated_depth;
      GpuTexture& current_mask=odd?generated_mask:reset_mask;
      GpuTexture& previous_mask=odd?reset_mask:generated_mask;
      const auto current_depth_surf=odd?generated_depth_surf:reset_depth_surf;
      const auto previous_depth_tex=odd?reset_depth_tex:generated_depth_tex;
      const auto current_mask_surf=odd?generated_mask_surf:reset_mask_surf;
      const auto previous_mask_tex=odd?reset_mask_tex:generated_mask_tex;
      Restore(gpu,flow,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,current_depth,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);
      encode_l4(false,frame_motion_handle[frame],frame_depth_handle[frame],
                current_depth_surf);
      auto frame_flow=snapshot_texture(flow,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      auto frame_depth_bytes=snapshot_texture(current_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      checkpoints[base+2u]=Sha256(JoinValues({frame_flow,frame_depth_bytes}));
      const fs::path frame_root=output_root/("generated-"+std::to_string(frame));
      fs::create_directories(frame_root);
      Write(frame_root/"l14-flow.rg32f",frame_flow);
      Write(frame_root/"l14-depth.r32u",frame_depth_bytes);
      OverwriteBuffer(gpu,atomic,Guarded(connected::InitialAtomic()));
      Restore(gpu,current_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);
      encode_l5(current_depth_surf,current_mask_surf);
      auto atomic_bytes=snapshot_buffer(atomic,connected::kAtomicBytes);
      auto mask_bytes=snapshot_texture(current_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      checkpoints[base+3u]=Sha256(JoinValues({atomic_bytes,mask_bytes}));
      Restore(gpu,previous_depth,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      Restore(gpu,previous_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      Restore(gpu,working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,working_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);
      encode_l6(previous_depth_tex,previous_mask_tex);
      auto vector_bytes=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      auto working_mask_bytes=snapshot_texture(working_mask,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      std::array<std::vector<std::uint8_t>,5> frame_pyramid;
      for(std::size_t i=0;i<pyramid.size();++i)
        frame_pyramid[i]=snapshot_buffer(pyramid[i],kPyramidBytes[i]);
      checkpoints[base+4u]=Sha256(JoinValues({vector_bytes,working_mask_bytes,
          frame_pyramid[0],frame_pyramid[1],frame_pyramid[2],
          frame_pyramid[3],frame_pyramid[4]}));
      encode_l7();
      auto repaired=snapshot_texture(working_vector,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      checkpoints[base+5u]=Sha256(repaired);
      GpuTexture& current_scatter=odd?scatter_generated:scatter_reset;
      const auto current_scatter_surf=odd?scatter_generated_surf:scatter_reset_surf;
      const auto current_scatter_tex=odd?scatter_generated_tex:scatter_reset_tex;
      const auto previous_scatter_tex=odd?scatter_reset_tex:scatter_generated_tex;
      Restore(gpu,current_scatter,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      ExecuteAndReset(gpu);encode_l8(current_scatter_surf);
      auto scatter_bytes=snapshot_texture(current_scatter,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      checkpoints[base+6u]=Sha256(scatter_bytes);
      GpuTexture& previous_projection=odd?reset_projection:auxiliary_vector;
      GpuTexture& current_projection=odd?auxiliary_vector:reset_projection;
      const auto previous_projection_tex=odd?reset_projection_tex:auxiliary_vector_tex;
      const auto current_projection_surf=odd?auxiliary_vector_surf:reset_projection_surf;
      const auto previous_feature_tex=odd?reset_feature_tex:generated_feature_tex;
      const auto current_feature_tex=odd?generated_feature_tex:reset_feature_tex;
      Restore(gpu,previous_feature,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      Restore(gpu,previous_projection,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      Restore(gpu,current_projection,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,auxiliary_mask,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,model_input,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
      Restore(gpu,allocation29,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
      ExecuteAndReset(gpu);
      const std::array<std::uint64_t,10> inputs={
          previous_feature_tex,current_feature_tex,current_scatter_tex,
          previous_scatter_tex,previous_projection_tex,shared_metric_tex,
          allocation29_tex,flow_tex,working_vector_tex,working_mask_tex};
      encode_l19(inputs,current_projection_surf);
      auto frame_model_guarded=ReadBuffer(gpu,model_input);
      auto frame_model=Payload(frame_model_guarded,connected::kModelInputBytes);
      extra_canary_mismatches+=CanaryMismatches(
          frame_model_guarded,connected::kModelInputBytes);
      auto aux_mask_bytes=ReadTexture(gpu,auxiliary_mask);
      auto projection_bytes=ReadTexture(gpu,current_projection);
      checkpoints[base+7u]=Sha256(JoinValues({frame_model,aux_mask_bytes,
                                              projection_bytes}));
      final_model_before_tail=frame_model;
      run_tail("generated-"+std::to_string(frame));
      auto frame_state=ReadTexture(gpu,allocation29);
      Write(frame_root/"allocation37.fp16",frame_model);
      Write(frame_root/"allocation29-post-l52.r16f",frame_state);
      allocation37_sha256.push_back(Sha256(frame_model));
      allocation29_sha256.push_back(Sha256(frame_state));
      for(std::size_t i=0;i<frame_model.size();i+=2)
        extra_model_poison_halfwords+=frame_model[i]==poison&&
                                      frame_model[i+1]==poison;
      for(std::size_t i=0;i<frame_state.size();i+=2)
        extra_state_poison_halfwords+=frame_state[i]==poison&&
                                      frame_state[i+1]==poison;
    }
    auto model_input_after_tail_guarded=ReadBuffer(gpu,model_input);
    auto model_input_after_tail=Payload(model_input_after_tail_guarded,
                                        connected::kModelInputBytes);
#else
    Restore(gpu,model_input,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    ExecuteAndReset(gpu);
    std::array<std::uint8_t,48> p20{};
    Pack64(&p20,0,l20_weight.resource->GetGPUVirtualAddress()+kGuardBytes);
    Pack64(&p20,8,model_input.resource->GetGPUVirtualAddress()+kGuardBytes);
    Pack32(&p20,16,160);Pack32(&p20,20,96);Pack32(&p20,24,162);
    Pack32(&p20,28,98);
    Pack64(&p20,32,l20_output.resource->GetGPUVirtualAddress()+kGuardBytes);
    Pack32(&p20,40,160);Pack32(&p20,44,96);
    launch(28,p20.data(),p20.size(),{10,8,1},{32,4,1});
    auto l20_output_guarded=ReadBuffer(gpu,l20_output);
    auto l20_output_bytes=Payload(l20_output_guarded,connected::kL20OutputBytes);
    auto model_input_after_l20_guarded=ReadBuffer(gpu,model_input);
    auto model_input_after_l20=Payload(model_input_after_l20_guarded,
                                       connected::kModelInputBytes);
    auto l20_weight_after_guarded=ReadBuffer(gpu,l20_weight);
    auto l20_weight_after=Payload(l20_weight_after_guarded,l20_weight_bytes.size());
    const auto l20_checkpoint_sha=Sha256(l20_output_bytes);
#endif
    std::size_t input_mismatches=0;
    input_mismatches+=Mismatches(external.source_reset,ReadTexture(gpu,source_reset));
    input_mismatches+=Mismatches(external.source_generated,ReadTexture(gpu,source_generated));
    input_mismatches+=Mismatches(external.source_generated2,ReadTexture(gpu,source_generated2));
    input_mismatches+=Mismatches(external.vector_reset,ReadTexture(gpu,vector_reset));
    input_mismatches+=Mismatches(external.vector_generated,ReadTexture(gpu,vector_generated));
    input_mismatches+=Mismatches(external.vector_generated2,ReadTexture(gpu,vector_generated2));
    input_mismatches+=Mismatches(external.scalar_reset,ReadTexture(gpu,scalar_reset));
    input_mismatches+=Mismatches(external.scalar_generated,ReadTexture(gpu,scalar_generated));
    input_mismatches+=Mismatches(external.scalar_generated2,ReadTexture(gpu,scalar_generated2));
    for(std::size_t i=0;i<additional_frames.size();++i){
      input_mismatches+=Mismatches(additional_frames[i].source,
                                   ReadTexture(gpu,additional_source[i]));
      input_mismatches+=Mismatches(additional_frames[i].motion,
                                   ReadTexture(gpu,additional_motion[i]));
      input_mismatches+=Mismatches(additional_frames[i].depth,
                                   ReadTexture(gpu,additional_depth[i]));
    }
#ifdef FG_CONNECTED_FULL_CHAIN
    input_mismatches+=Mismatches(final_model_before_tail,model_input_after_tail);
#else
    input_mismatches+=Mismatches(model_input_bytes,model_input_after_l20);
    input_mismatches+=Mismatches(l20_weight_bytes,l20_weight_after);
#endif
    std::size_t canary_mismatches=CanaryMismatches(model_frame1_guarded,connected::kModelInputBytes);
    canary_mismatches+=CanaryMismatches(model_frame2_guarded,connected::kModelInputBytes);
    canary_mismatches+=extra_canary_mismatches;
#ifdef FG_CONNECTED_FULL_CHAIN
    canary_mismatches+=CanaryMismatches(model_input_after_tail_guarded,
                                        connected::kModelInputBytes);
#else
    canary_mismatches+=CanaryMismatches(model_input_after_l20_guarded,
                                        connected::kModelInputBytes);
    canary_mismatches+=CanaryMismatches(l20_output_guarded,
                                        connected::kL20OutputBytes);
    canary_mismatches+=CanaryMismatches(l20_weight_after_guarded,
                                        l20_weight_bytes.size());
#endif
    auto atomic_guarded=ReadBuffer(gpu,atomic);canary_mismatches+=CanaryMismatches(atomic_guarded,connected::kAtomicBytes);
    auto status_guarded=ReadBuffer(gpu,status_buffer);canary_mismatches+=CanaryMismatches(status_guarded,4);
    for(std::size_t i=0;i<pyramid.size();++i){auto guarded=ReadBuffer(gpu,pyramid[i]);canary_mismatches+=CanaryMismatches(guarded,kPyramidBytes[i]);}
    std::size_t poison_halfwords=0;
    for(const auto* bytes:{&model_input_frame1_bytes,&model_input_frame2_bytes})
      for(std::size_t i=0;i<bytes->size();i+=2)
        poison_halfwords+=(*bytes)[i]==poison&&(*bytes)[i+1]==poison;
    poison_halfwords+=extra_model_poison_halfwords;
    std::size_t allocation29_poison_halfwords=0;
    for(const auto* bytes:{&allocation29_after_frame1,&allocation29_after_frame2})
      for(std::size_t i=0;i<bytes->size();i+=2)
        allocation29_poison_halfwords+=(*bytes)[i]==poison&&
                                           (*bytes)[i+1]==poison;
    allocation29_poison_halfwords+=extra_state_poison_halfwords;
#ifndef FG_CONNECTED_FULL_CHAIN
    std::size_t l20_poison_halfwords=0;
    for(std::size_t i=0;i<l20_output_bytes.size();i+=2)
      l20_poison_halfwords+=l20_output_bytes[i]==poison&&
                            l20_output_bytes[i+1]==poison;
#endif
    for(auto item=loaded.rbegin();item!=loaded.rend();++item){
      RequireNvapi(destroy_function(gpu.device.Get(),item->function),"DestroyFunction");
      RequireNvapi(destroy_module(gpu.device.Get(),item->module),"DestroyModule");
    }
#ifdef FG_CONNECTED_FULL_CHAIN
    const bool pass=input_mismatches==0&&canary_mismatches==0&&
                    poison_halfwords==0&&allocation29_poison_halfwords==0&&
                    allocation29_after_frame1!=connected::InitialAllocation29();
#else
    const bool pass=input_mismatches==0&&canary_mismatches==0&&
                    poison_halfwords==0&&l20_poison_halfwords==0;
#endif
    Write(output_root/"generated-1"/"allocation37.fp16",model_input_frame1_bytes);
    Write(output_root/"generated-1"/"allocation33.r8",auxiliary_mask_frame1_bytes);
    Write(output_root/"generated-1"/"allocation27.rg16f",auxiliary_vector_frame1_bytes);
    Write(output_root/"generated-1"/"allocation29-post-l52.r16f",allocation29_after_frame1);
    Write(output_root/"generated-2"/"allocation37.fp16",model_input_frame2_bytes);
    Write(output_root/"generated-2"/"allocation33.r8",auxiliary_mask_frame2_bytes);
    Write(output_root/"generated-2"/"allocation26.rg16f",auxiliary_vector_frame2_bytes);
    Write(output_root/"generated-2"/"allocation29-post-l52.r16f",allocation29_after_frame2);
#ifndef FG_CONNECTED_FULL_CHAIN
    Write(output_root/"l20-output.fp16",l20_output_bytes);
#endif
    std::ofstream receipt(output_root/"run-receipt.json");
    receipt<<"{\n  \"schema\": \"neuralstudio.fg.production-contract-early-run.v1\",\n"
           <<"  \"backend\": \"rtx-d3d12-nvapi\",\n"
           <<"  \"run_id\": \""<<run_id<<"\",\n"
           <<"  \"process_id\": "<<GetCurrentProcessId()<<",\n"
           <<"  \"poison_byte\": "<<unsigned(poison)<<",\n"
#ifdef FG_CONNECTED_FULL_CHAIN
           <<"  \"rendered_frame_count\": 16,\n"
           <<"  \"phase_launch_counts\": [9, 120],\n"
           <<"  \"launch_count\": 129,\n"
#else
           <<"  \"launch_range\": [3, 20],\n"
           <<"  \"launch_count\": 18,\n"
#endif
           <<"  \"checkpoint_sha256\": {";
    for(std::size_t i=0;i<9;++i)
      receipt<<(i?",":"")<<"\n    \"reset-launch-"<<std::setfill('0')
             <<std::setw(6)<<(3+i)<<"\": \""<<checkpoints[i]<<"\"";
    for(std::size_t frame=1;frame<16;++frame)
      for(std::size_t i=0;i<8;++i)
        receipt<<",\n    \"generated-"<<frame<<"-launch-"<<std::setfill('0')
               <<std::setw(6)<<(12+i)<<"\": \""
               <<checkpoints[9+(frame-1)*8+i]<<"\"";
    receipt<<"\n  },\n  \"allocation29_runtime_source\": \"generated-half-0.5-then-live-L52\",\n"
           <<"  \"allocation29_capture_bytes_consumed\": false,\n"
           <<"  \"input_mismatched_bytes\": "<<input_mismatches<<",\n"
           <<"  \"canary_mismatched_bytes\": "<<canary_mismatches<<",\n"
           <<"  \"allocation37_poison_halfwords\": "<<poison_halfwords<<",\n"
           <<"  \"allocation29_poison_halfwords\": "<<allocation29_poison_halfwords<<",\n"
#ifndef FG_CONNECTED_FULL_CHAIN
           <<"  \"l20_output_poison_halfwords\": "<<l20_poison_halfwords<<",\n"
#endif
           <<"  \"expected_or_archived_intermediate_ingress\": false,\n"
           <<"  \"address_fields_recorded\": false,\n"
           <<"  \"allocation37_sha256\": [";
    for(std::size_t i=0;i<allocation37_sha256.size();++i)
      receipt<<(i?", ":"")<<"\""<<allocation37_sha256[i]<<"\"";
    receipt<<"],\n  \"allocation29_sha256\": [";
    for(std::size_t i=0;i<allocation29_sha256.size();++i)
      receipt<<(i?", ":"")<<"\""<<allocation29_sha256[i]<<"\"";
    receipt<<"],\n"
           <<"  \"l20_live_handoff\": true,\n"
#ifdef FG_CONNECTED_FULL_CHAIN
           <<"  \"connected_tail_receipt_pattern\": \"generated-N/l20-l87/run-receipt.json\",\n"
#else
           <<"  \"l20_output_bytes\": "<<l20_output_bytes.size()<<",\n"
           <<"  \"l20_output_sha256\": \""<<l20_checkpoint_sha<<"\",\n"
#endif
           <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
    std::cout<<"allocation37_frame1_sha256="<<Sha256(model_input_frame1_bytes)<<'\n'
             <<"allocation37_frame2_sha256="<<Sha256(model_input_frame2_bytes)<<'\n'
             <<"allocation29_frame1_sha256="<<Sha256(allocation29_after_frame1)<<'\n'
             <<"allocation29_frame2_sha256="<<Sha256(allocation29_after_frame2)<<'\n';
#ifndef FG_CONNECTED_FULL_CHAIN
    std::cout<<"l20_output_sha256="<<l20_checkpoint_sha<<'\n';
#else
    model_input_shared.Release();
    allocation29_shared.Release();
#endif
    std::cout<<"result="<<(pass?"pass":"fail")<<'\n';return pass?0:3;
  }catch(const std::exception& error){std::cerr<<"error="<<error.what()<<'\n';return 1;}
}
