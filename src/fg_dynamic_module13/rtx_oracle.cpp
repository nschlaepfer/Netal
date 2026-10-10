#define main FgDynamicModule13EmbeddedMain
#include "../fg_early_l5_l15/rtx_oracle.cpp"
#undef main

#include "contract.h"

namespace {
namespace dynamic_contract = fg_dynamic_module13;

struct DynamicArray {
  CUarray array = nullptr;
  CUtexObject texture = 0;
  CUsurfObject surface = 0;
  std::size_t row_bytes = 0;
  std::uint32_t rows = 0;

  void Allocate(std::uint32_t width, std::uint32_t height,
                CUarray_format format, unsigned channels,
                std::size_t component_bytes, bool surface_load_store) {
    rows = height;
    row_bytes = std::size_t{width} * channels * component_bytes;
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    descriptor.Width = width;
    descriptor.Height = height;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    descriptor.Flags = surface_load_store ? CUDA_ARRAY3D_SURFACE_LDST : 0u;
    Check(cuArray3DCreate(&array, &descriptor), "cuArray3DCreate");
  }
  void MakeTexture(bool normalized, bool integer) {
    CUDA_RESOURCE_DESC resource{};
    resource.resType = CU_RESOURCE_TYPE_ARRAY;
    resource.res.array.hArray = array;
    CUDA_TEXTURE_DESC descriptor{};
    descriptor.addressMode[0] = descriptor.addressMode[1] =
        CU_TR_ADDRESS_MODE_BORDER;
    descriptor.filterMode = CU_TR_FILTER_MODE_POINT;
    descriptor.flags = (normalized ? CU_TRSF_NORMALIZED_COORDINATES : 0u) |
                       (integer ? CU_TRSF_READ_AS_INTEGER : 0u);
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
    if (bytes.size() != row_bytes * rows) throw std::runtime_error("upload size");
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = row_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = row_bytes;
    copy.Height = rows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D upload");
  }
  std::vector<std::uint8_t> Download() const {
    std::vector<std::uint8_t> result(row_bytes * rows);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = result.data();
    copy.dstPitch = row_bytes;
    copy.WidthInBytes = row_bytes;
    copy.Height = rows;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D download");
    return result;
  }
  ~DynamicArray() {
    if (texture) cuTexObjectDestroy(texture);
    if (surface) cuSurfObjectDestroy(surface);
    if (array) cuArrayDestroy(array);
  }
};

struct alignas(16) DynamicParameters {
  CUtexObject flow_texture;
  CUtexObject depth_texture;
  CUdeviceptr atomic_output;
  CUsurfObject mask_surface;
  std::uint32_t dead32;
  std::uint32_t dead36;
  float input_scale_x;
  float input_scale_y;
  std::uint32_t width;
  std::uint32_t height;
  float motion_divisor;
  float neighbor_divisor;
  float dead64;
  float depth_scale;
  float dead72;
  std::uint8_t invert_depth;
  std::uint8_t padding77;
  std::uint16_t padding78;
  std::array<float, 16> transform;
};
static_assert(sizeof(DynamicParameters) == 144);

struct DynamicAtomicClass {
  std::size_t sentinel = 0, candidate = 0, negative_high = 0;
  std::set<std::uint32_t> high_words, low_words;
};

DynamicAtomicClass DynamicClassifyAtomic(
    const std::vector<std::uint8_t>& bytes) {
  DynamicAtomicClass result;
  for (std::size_t offset = 0; offset < bytes.size(); offset += 8u) {
    std::uint64_t value = 0;
    std::memcpy(&value, bytes.data() + offset, 8);
    if (value == dynamic_contract::kAtomicSentinel) { ++result.sentinel; continue; }
    ++result.candidate;
    const auto high = static_cast<std::uint32_t>(value >> 32u);
    const auto low = static_cast<std::uint32_t>(value);
    result.negative_high += (high & 0x80000000u) != 0u;
    result.high_words.insert(high); result.low_words.insert(low);
  }
  return result;
}

struct DynamicMaskClass { std::size_t zero=0, one=0, poison=0, other=0; };
DynamicMaskClass DynamicClassifyMask(const std::vector<std::uint8_t>& bytes,
                                     std::uint8_t poison) {
  DynamicMaskClass result;
  for (auto value : bytes) {
    if (value == 0) ++result.zero; else if (value == 255) ++result.one;
    else if (value == poison) ++result.poison; else ++result.other;
  }
  return result;
}

std::vector<std::uint8_t> DynamicBytes(const DynamicParameters& value) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(&value);
  return std::vector<std::uint8_t>(begin, begin + sizeof(value));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 9) {
    std::cerr << "usage=fg-dynamic-module13-rtx <module> <out> <run> <poison> "
                 "<resolution> <phase> <scenario> <full|hash-only>\n";
    return 2;
  }
  try {
    const auto module_bytes = Read(argv[1]);
    if (module_bytes.size() != dynamic_contract::kModuleBytes ||
        Sha256(module_bytes) != dynamic_contract::kModuleSha256)
      throw std::runtime_error("module13 identity");
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    char* end = nullptr; const auto parsed = std::strtoul(argv[4], &end, 16);
    if (end == argv[4] || *end || parsed > 255) throw std::runtime_error("poison");
    const auto poison = static_cast<std::uint8_t>(parsed);
    const auto& resolution = dynamic_contract::ParseResolution(argv[5]);
    const auto phase = dynamic_contract::ParsePhase(argv[6]);
    const auto scenario = dynamic_contract::ParseScenario(argv[7]);
    const std::string artifact_mode = argv[8];
    if (artifact_mode != "full" && artifact_mode != "hash-only")
      throw std::runtime_error("artifact mode");
    const bool artifacts = artifact_mode == "full";
    std::filesystem::create_directories(output_root);
    const auto flow_before = dynamic_contract::Flow(resolution, phase, scenario);
    const auto depth_before = dynamic_contract::Depth(resolution, phase, scenario);
    const auto atomic_before = dynamic_contract::InitialAtomic(resolution);
    std::vector<std::uint8_t> mask_before(
        dynamic_contract::Pixels(resolution) + resolution.width, poison);

    Check(cuInit(0), "cuInit"); CUdevice device=0; Check(cuDeviceGet(&device,0),"device");
    CUcontext context=nullptr; Check(cuDevicePrimaryCtxRetain(&context,device),"retain");
    Check(cuCtxSetCurrent(context),"current"); CUmodule module=nullptr;
    Check(cuModuleLoadData(&module,module_bytes.data()),"module"); CUfunction function=nullptr;
    Check(cuModuleGetFunction(&function,module,"main_kernel"),"function");
    DynamicArray flow, depth, mask;
    flow.Allocate(resolution.width,resolution.height,CU_AD_FORMAT_FLOAT,2,4,false);
    depth.Allocate(resolution.width,resolution.height,CU_AD_FORMAT_UNSIGNED_INT32,1,4,false);
    mask.Allocate(resolution.width,resolution.height+1u,CU_AD_FORMAT_UNSIGNED_INT8,1,1,true);
    flow.Upload(flow_before); depth.Upload(depth_before); mask.Upload(mask_before);
    flow.MakeTexture(true,false); depth.MakeTexture(false,true); mask.MakeSurface();
    GuardedDeviceBuffer atomic; atomic.Allocate(atomic_before.size());
    Check(cuMemcpyHtoD(atomic.payload(),atomic_before.data(),atomic_before.size()),"atomic upload");
    const std::array<float,16> identity={1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1};
    DynamicParameters parameters{flow.texture,depth.texture,atomic.payload(),mask.surface,
      0,0,std::bit_cast<float>(resolution.input_scale_x_bits),
      std::bit_cast<float>(resolution.input_scale_y_bits),resolution.width,resolution.height,
      std::bit_cast<float>(resolution.motion_divisor_bits),
      std::bit_cast<float>(resolution.neighbor_divisor_bits),40.0f,1.0f,600.0f,0,0,0,identity};
    auto address_free=parameters;address_free.flow_texture=0;address_free.depth_texture=0;
    address_free.atomic_output=0;address_free.mask_surface=0;
    const auto normalized_sha=Sha256(DynamicBytes(address_free));
    if(normalized_sha!=resolution.address_free_sha256)throw std::runtime_error("parameter hash");
    std::size_t parameter_bytes=sizeof(parameters);void*extra[]={CU_LAUNCH_PARAM_BUFFER_POINTER,&parameters,CU_LAUNCH_PARAM_BUFFER_SIZE,&parameter_bytes,CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function,resolution.grid_x,resolution.grid_y,1,16,16,1,0,nullptr,nullptr,extra),"launch");
    Check(cuCtxSynchronize(),"sync");
    const auto flow_after=flow.Download(),depth_after=depth.Download(),guarded_mask=mask.Download();
    const auto guarded_atomic=atomic.ReadAll();
    const auto atomic_output=Payload(guarded_atomic,atomic_before.size());
    const std::vector<std::uint8_t> mask_output(guarded_mask.begin(),guarded_mask.begin()+dynamic_contract::Pixels(resolution));
    const auto atomic_class=DynamicClassifyAtomic(atomic_output);
    const auto mask_class=DynamicClassifyMask(mask_output,poison);
    const std::size_t input_mismatch=Mismatches(flow_before,flow_after)+Mismatches(depth_before,depth_after);
    const std::size_t atomic_canary=CanaryMismatches(guarded_atomic,atomic_before.size());
    std::size_t mask_canary=0;for(std::size_t i=mask_output.size();i<guarded_mask.size();++i)mask_canary+=guarded_mask[i]!=poison;
    const bool classification=atomic_class.candidate>0&&atomic_class.negative_high==0&&atomic_class.low_words.size()>1&&mask_class.zero>0&&mask_class.one>0&&mask_class.poison==0&&mask_class.other==0;
    const bool pass=input_mismatch==0&&atomic_canary==0&&mask_canary==0&&classification;
    if(artifacts){Write(output_root/"atomic-output.bin",atomic_output);Write(output_root/"mask-output.bin",mask_output);}
    std::ofstream receipt(output_root/"run-receipt.json");
    receipt<<"{\n  \"schema\": \"neuralstudio.fg.dynamic-module13-run.v1\",\n  \"backend\": \"rtx-cuda-driver-typed-arrays\",\n"
      <<"  \"run_id\": \""<<run_id<<"\",\n  \"process_id\": "<<GetCurrentProcessId()<<",\n  \"resolution\": \""<<resolution.name<<"\",\n"
      <<"  \"phase\": \""<<dynamic_contract::PhaseName(phase)<<"\",\n  \"scenario\": \""<<dynamic_contract::ScenarioName(scenario)<<"\",\n"
      <<"  \"poison_byte\": "<<unsigned(poison)<<",\n  \"module_sha256\": \""<<dynamic_contract::kModuleSha256<<"\",\n"
      <<"  \"large_trace_launch\": "<<(phase==dynamic_contract::Phase::kReset?5:17)<<",\n  \"packed_parameter_bytes\": 144,\n"
      <<"  \"extent\": ["<<resolution.width<<", "<<resolution.height<<"],\n  \"grid\": ["<<resolution.grid_x<<", "<<resolution.grid_y<<", 1],\n  \"block\": [16, 16, 1],\n"
      <<"  \"trace_raw_parameter_sha256\": \""<<(phase==dynamic_contract::Phase::kReset?resolution.reset_raw_sha256:resolution.generated_raw_sha256)<<"\",\n"
      <<"  \"address_free_parameter_sha256\": \""<<normalized_sha<<"\",\n  \"input_generator\": \""<<dynamic_contract::kGenerator<<"\",\n"
      <<"  \"flow_pre_sha256\": \""<<Sha256(flow_before)<<"\",\n  \"flow_post_sha256\": \""<<Sha256(flow_after)<<"\",\n"
      <<"  \"depth_pre_sha256\": \""<<Sha256(depth_before)<<"\",\n  \"depth_post_sha256\": \""<<Sha256(depth_after)<<"\",\n"
      <<"  \"atomic_output_bytes\": "<<atomic_output.size()<<",\n  \"atomic_output_sha256\": \""<<Sha256(atomic_output)<<"\",\n"
      <<"  \"mask_output_bytes\": "<<mask_output.size()<<",\n  \"mask_output_sha256\": \""<<Sha256(mask_output)<<"\",\n"
      <<"  \"atomic_sentinel_u64\": "<<atomic_class.sentinel<<",\n  \"atomic_candidate_u64\": "<<atomic_class.candidate<<",\n"
      <<"  \"atomic_unique_high_words\": "<<atomic_class.high_words.size()<<",\n  \"atomic_unique_low_words\": "<<atomic_class.low_words.size()<<",\n"
      <<"  \"mask_zero_pixels\": "<<mask_class.zero<<",\n  \"mask_one_pixels\": "<<mask_class.one<<",\n"
      <<"  \"input_mismatched_bytes\": "<<input_mismatch<<",\n  \"atomic_canary_mismatched_bytes\": "<<atomic_canary<<",\n  \"mask_guard_row_mismatched_bytes\": "<<mask_canary<<",\n"
      <<"  \"artifacts_written\": "<<(artifacts?"true":"false")<<",\n  \"expected_or_captured_output_ingress\": false,\n  \"archived_intermediate_ingress\": false,\n  \"address_fields_recorded\": false,\n"
      <<"  \"result\": \""<<(pass?"pass":"fail")<<"\"\n}\n";
    std::cout<<"atomic_output_sha256="<<Sha256(atomic_output)<<"\nmask_output_sha256="<<Sha256(mask_output)<<"\nresult="<<(pass?"pass":"fail")<<'\n';
    return pass?0:3;
  }catch(const std::exception&e){std::cerr<<"error="<<e.what()<<'\n';return 1;}
}
