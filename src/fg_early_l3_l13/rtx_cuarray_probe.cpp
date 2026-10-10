// Preserved diagnostic only. The accepted oracle uses true D3D12/NvAPI
// typed resources in rtx_oracle.cpp.
#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/rtx_oracle.cpp"

#include "contract.h"

namespace {

namespace contract = fg_early_l3_l13;

struct ArrayResource {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::uint32_t width = 0, height = 0, bytes_per_pixel = 0;

  void Allocate(std::uint32_t w, std::uint32_t h, CUarray_format format,
                unsigned channels, std::uint32_t bpp, bool writable) {
    width = w;
    height = h;
    bytes_per_pixel = bpp;
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    descriptor.Flags = writable ? CUDA_ARRAY3D_SURFACE_LDST : 0u;
    Check(cuArray3DCreate(&array, &descriptor), "cuArray3DCreate");
  }

  void MakeTexture() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    Check(cuTexObjectCreate(&texture, &resource, &descriptor, nullptr),
          "cuTexObjectCreate");
  }

  void MakeSurface() {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    Check(cuSurfObjectCreate(&surface, &resource), "cuSurfObjectCreate");
  }

  void Upload(const std::vector<std::uint8_t>& bytes) const {
    if (bytes.size() != std::size_t{width} * height * bytes_per_pixel)
      throw std::runtime_error("CUarray upload size drift");
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = std::size_t{width} * bytes_per_pixel;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = std::size_t{width} * bytes_per_pixel;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(HtoA)");
  }

  std::vector<std::uint8_t> Readback() const {
    std::vector<std::uint8_t> bytes(
        std::size_t{width} * height * bytes_per_pixel);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = std::size_t{width} * bytes_per_pixel;
    copy.WidthInBytes = std::size_t{width} * bytes_per_pixel;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(AtoH)");
    return bytes;
  }
};

template <std::size_t N>
void Pack64(std::array<std::uint8_t, N>* bytes, std::size_t offset,
            std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t N>
void Pack32(std::array<std::uint8_t, N>* bytes, std::size_t offset,
            std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

std::vector<std::uint8_t> Join(
    std::initializer_list<const std::vector<std::uint8_t>*> values) {
  std::vector<std::uint8_t> result;
  for (const auto* value : values)
    result.insert(result.end(), value->begin(), value->end());
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::cerr << "usage=fg-early-l3-l13-cuarray <module> <output-dir> "
                 "<poison-hex> <fixture>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != contract::kModuleBytes ||
        Sha256(module_bytes) != contract::kModuleSha256)
      throw std::runtime_error("module11 identity mismatch");
    const auto poison_value = std::stoul(argv[3], nullptr, 16);
    if (poison_value > 255u) throw std::runtime_error("invalid poison");
    const auto poison = static_cast<std::uint8_t>(poison_value);
    const std::string fixture = argv[4];
    const auto source_before = contract::GenerateSource();
    const auto history_before = contract::GenerateHistory(fixture);
    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "retain context");
    Check(cuCtxSetCurrent(context), "set context");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "load module11");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "get module11 function");
    ArrayResource source, history, half_color, half_metric, next_history;
    source.Allocate(640,360,CU_AD_FORMAT_UNSIGNED_INT8,4,4,false);
    history.Allocate(640,360,CU_AD_FORMAT_HALF,4,8,false);
    half_color.Allocate(320,180,CU_AD_FORMAT_HALF,4,8,true);
    half_metric.Allocate(320,180,CU_AD_FORMAT_HALF,1,2,true);
    next_history.Allocate(640,360,CU_AD_FORMAT_HALF,4,8,true);
    source.Upload(source_before);
    history.Upload(history_before);
    half_color.Upload(contract::Poison(contract::kHalfColorBytes,poison));
    half_metric.Upload(contract::Poison(contract::kHalfMetricBytes,poison));
    next_history.Upload(contract::Poison(contract::kHistoryBytes,poison));
    source.MakeTexture();
    history.MakeTexture();
    half_color.MakeSurface();
    half_metric.MakeSurface();
    next_history.MakeSurface();
    std::array<std::uint8_t,168> packed{};
    Pack64(&packed,0,source.texture);
    Pack64(&packed,80,history.texture);
    Pack64(&packed,96,half_color.surface);
    Pack64(&packed,104,half_metric.surface);
    Pack64(&packed,120,next_history.surface);
    Pack32(&packed,128,2);
    Pack32(&packed,136,640);
    Pack32(&packed,140,360);
    Pack32(&packed,144,320);
    Pack32(&packed,148,180);
    std::size_t parameter_bytes=packed.size();
    void* extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,packed.data(),
                   CU_LAUNCH_PARAM_BUFFER_SIZE,&parameter_bytes,
                   CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function,40,23,1,256,1,1,0,nullptr,nullptr,extra),
          "launch module11 with generic CUarrays");
    Check(cuCtxSynchronize(),"synchronize CUarray probe");
    const auto half_color_after=half_color.Readback();
    const auto half_metric_after=half_metric.Readback();
    const auto next_history_after=next_history.Readback();
    const auto combined=Join(
        {&half_color_after,&half_metric_after,&next_history_after});
    std::filesystem::create_directories(argv[2]);
    Write(std::filesystem::path(argv[2])/"output.bin",combined);
    std::ofstream receipt(std::filesystem::path(argv[2])/"run-receipt.json");
    receipt << "{\n  \"schema\": \"neuralstudio.fg.early-l3-l13-cuarray-diagnostic.v1\",\n"
            << "  \"fixture\": \""<<fixture<<"\",\n"
            << "  \"poison_byte\": "<<unsigned(poison)<<",\n"
            << "  \"combined_output_sha256\": \""<<Sha256(combined)<<"\",\n"
            << "  \"generic_cuarray_transport\": true,\n"
            << "  \"authoritative_oracle\": false,\n"
            << "  \"expected_or_archived_tensor_ingress\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"diagnostic-only\"\n}\n";
    std::cout<<"combined_output_sha256="<<Sha256(combined)<<"\nresult=diagnostic-only\n";
    return 0;
  } catch(const std::exception& error) {
    std::cerr<<"error="<<error.what()<<'\n';
    return 1;
  }
}
