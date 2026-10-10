#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace {

// nvngx_dlssnr.dll uses MSVC's overload-sorted vtable for
// NVSDK_NGX_Parameter. MinGW follows declaration order instead, so a normal
// C++ subclass is ABI-incompatible even though the source declarations match.
// Keep this raw vtable pinned to a clang-generated x64 MSVC layout.
class ParameterStore final {
 public:
  ParameterStore() : vtable_(MsvcVtable()) {}

  NVSDK_NGX_Parameter* AsNgx() {
    return reinterpret_cast<NVSDK_NGX_Parameter*>(this);
  }

  const NVSDK_NGX_Parameter* AsNgx() const {
    return reinterpret_cast<const NVSDK_NGX_Parameter*>(this);
  }

  void SetU32(const char* name, unsigned int value) {
    SetUnsigned(this, name, value);
  }

  void SetI32(const char* name, int value) { SetInt(this, name, value); }

  void SetF32(const char* name, float value) { SetFloat(this, name, value); }

  void SetPointerValue(const char* name, void* value) {
    SetPointer(this, name, value);
  }

  NVSDK_NGX_Result GetU64(const char* name,
                          unsigned long long* output) const {
    return GetUnsignedLongLong(const_cast<ParameterStore*>(this), name,
                               output);
  }

  NVSDK_NGX_Result GetPointerValue(const char* name, void** output) const {
    return GetPointer(const_cast<ParameterStore*>(this), name, output);
  }

 private:
  using Slot = void*;

  template <typename Function>
  static Slot SlotFor(Function function) {
    Slot slot = nullptr;
    static_assert(sizeof(slot) == sizeof(function));
    std::memcpy(&slot, &function, sizeof(slot));
    return slot;
  }

  static Slot* MsvcVtable() {
    static Slot slots[] = {
        SlotFor(&SetPointer),
        SlotFor(&SetD3D12),
        SlotFor(&SetD3D11),
        SlotFor(&SetInt),
        SlotFor(&SetUnsigned),
        SlotFor(&SetDouble),
        SlotFor(&SetFloat),
        SlotFor(&SetUnsignedLongLong),
        SlotFor(&GetPointer),
        SlotFor(&GetD3D12),
        SlotFor(&GetD3D11),
        SlotFor(&GetInt),
        SlotFor(&GetUnsigned),
        SlotFor(&GetDouble),
        SlotFor(&GetFloat),
        SlotFor(&GetUnsignedLongLong),
        SlotFor(&Reset),
    };
    return slots;
  }

  static void WINAPI SetPointer(ParameterStore* self, const char* name,
                                void* value) {
    self->pointers_[name] = value;
    LogSetValue(name, "pointer", value);
  }
  static void WINAPI SetD3D12(ParameterStore* self, const char* name,
                              ID3D12Resource* value) {
    self->d3d12_[name] = value;
    LogSetValue(name, "d3d12", value);
  }
  static void WINAPI SetD3D11(ParameterStore* self, const char* name,
                              ID3D11Resource* value) {
    self->d3d11_[name] = value;
    LogSetValue(name, "d3d11", value);
  }
  static void WINAPI SetInt(ParameterStore* self, const char* name, int value) {
    self->i32_[name] = value;
    LogSetValue(name, "i32", value);
  }
  static void WINAPI SetUnsigned(ParameterStore* self, const char* name,
                                 unsigned int value) {
    self->u32_[name] = value;
    LogSetValue(name, "u32", value);
  }
  static void WINAPI SetDouble(ParameterStore* self, const char* name,
                               double value) {
    self->f64_[name] = value;
    LogSetValue(name, "f64", value);
  }
  static void WINAPI SetFloat(ParameterStore* self, const char* name,
                              float value) {
    self->f32_[name] = value;
    LogSetValue(name, "f32", value);
  }
  static void WINAPI SetUnsignedLongLong(ParameterStore* self,
                                         const char* name,
                                         unsigned long long value) {
    self->u64_[name] = value;
    LogSetValue(name, "u64", value);
  }

  static NVSDK_NGX_Result WINAPI GetPointer(ParameterStore* self,
                                             const char* name, void** output) {
    return GetValue(self->pointers_, name, output, "pointer");
  }
  static NVSDK_NGX_Result WINAPI GetD3D12(ParameterStore* self,
                                           const char* name,
                                           ID3D12Resource** output) {
    return GetValue(self->d3d12_, name, output, "d3d12");
  }
  static NVSDK_NGX_Result WINAPI GetD3D11(ParameterStore* self,
                                           const char* name,
                                           ID3D11Resource** output) {
    return GetValue(self->d3d11_, name, output, "d3d11");
  }
  static NVSDK_NGX_Result WINAPI GetInt(ParameterStore* self,
                                         const char* name, int* output) {
    return GetValue(self->i32_, name, output, "i32");
  }
  static NVSDK_NGX_Result WINAPI GetUnsigned(ParameterStore* self,
                                              const char* name,
                                              unsigned int* output) {
    return GetValue(self->u32_, name, output, "u32");
  }
  static NVSDK_NGX_Result WINAPI GetDouble(ParameterStore* self,
                                            const char* name, double* output) {
    return GetValue(self->f64_, name, output, "f64");
  }
  static NVSDK_NGX_Result WINAPI GetFloat(ParameterStore* self,
                                           const char* name, float* output) {
    return GetValue(self->f32_, name, output, "f32");
  }
  static NVSDK_NGX_Result WINAPI GetUnsignedLongLong(
      ParameterStore* self, const char* name, unsigned long long* output) {
    return GetValue(self->u64_, name, output, "u64");
  }

  static void WINAPI Reset(ParameterStore* self) {
    self->u64_.clear();
    self->f32_.clear();
    self->f64_.clear();
    self->u32_.clear();
    self->i32_.clear();
    self->d3d11_.clear();
    self->d3d12_.clear();
    self->pointers_.clear();
  }

  template <typename Map, typename Value>
  static NVSDK_NGX_Result GetValue(const Map& values, const char* name,
                                   Value* output, const char* type) {
    if (name == nullptr || output == nullptr) {
      return NVSDK_NGX_Result_FAIL_InvalidParameter;
    }
    const auto found = values.find(name);
    if (found == values.end()) {
      *output = Value{};
      std::cout << "parameter.get name=\"" << name << "\" type=" << type
                << " hit=0\n";
      return NVSDK_NGX_Result_FAIL_MissingInput;
    }
    *output = found->second;
    std::cout << "parameter.get name=\"" << name << "\" type=" << type
              << " hit=1 ";
    PrintValue(*output);
    std::cout << '\n';
    return NVSDK_NGX_Result_Success;
  }

  template <typename Value>
  static void LogSetValue(const char* name, const char* type, Value value) {
    std::cout << "parameter.set name=\"" << (name != nullptr ? name : "")
              << "\" type=" << type << ' ';
    PrintValue(value);
    std::cout << '\n';
  }

  template <typename Value>
  static void PrintValue(Value value) {
    if constexpr (std::is_pointer_v<Value>) {
      std::cout << "value=" << (value != nullptr ? "present" : "null");
    } else if constexpr (std::is_same_v<Value, float>) {
      uint32_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      std::cout << "value_bits=0x" << std::hex << bits << std::dec;
    } else if constexpr (std::is_same_v<Value, double>) {
      uint64_t bits = 0;
      std::memcpy(&bits, &value, sizeof(bits));
      std::cout << "value_bits=0x" << std::hex << bits << std::dec;
    } else {
      std::cout << "value=" << value;
    }
  }

  Slot* vtable_;
  std::unordered_map<std::string, unsigned long long> u64_;
  std::unordered_map<std::string, float> f32_;
  std::unordered_map<std::string, double> f64_;
  std::unordered_map<std::string, unsigned int> u32_;
  std::unordered_map<std::string, int> i32_;
  std::unordered_map<std::string, ID3D11Resource*> d3d11_;
  std::unordered_map<std::string, ID3D12Resource*> d3d12_;
  std::unordered_map<std::string, void*> pointers_;
};

template <typename Function>
Function LoadFunction(HMODULE module, const char* name) {
  const FARPROC raw = GetProcAddress(module, name);
  Function function = nullptr;
  static_assert(sizeof(function) == sizeof(raw));
  std::memcpy(&function, &raw, sizeof(function));
  return function;
}

template <typename Function>
Function LoadVersionedFunction(HMODULE module, const char* versioned_name,
                               const char* fallback_name) {
  Function function = LoadFunction<Function>(module, versioned_name);
  return function != nullptr ? function
                             : LoadFunction<Function>(module, fallback_name);
}

using GetRuntimeU32 = uint32_t(NVSDK_CONV*)();
using Populate = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using Init = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long, const wchar_t*, NVSDK_NGX_CUDADevice*,
    NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using Create = NVSDK_NGX_Result(NVSDK_CONV*)(
    NVSDK_NGX_CUDADevice*, NVSDK_NGX_Feature,
    const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_CUDADevice*);
using Requirements = NVSDK_NGX_Result(NVSDK_CONV*)(
    int, const NVSDK_NGX_FeatureDiscoveryInfo*,
    NVSDK_NGX_FeatureRequirement*);
using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
    const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback);

using ThunkPopulate = NVSDK_NGX_Result(NVSDK_CONV*)(Populate,
                                                    NVSDK_NGX_Parameter*);
using ThunkInit = NVSDK_NGX_Result(NVSDK_CONV*)(
    Init, unsigned long long, const wchar_t*, NVSDK_NGX_CUDADevice*,
    NVSDK_NGX_Version, const NVSDK_NGX_Parameter*);
using ThunkCreate = NVSDK_NGX_Result(NVSDK_CONV*)(
    Create, NVSDK_NGX_CUDADevice*, NVSDK_NGX_Feature,
    const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using ThunkRelease = NVSDK_NGX_Result(NVSDK_CONV*)(Release,
                                                   NVSDK_NGX_Handle*);
using ThunkShutdown = NVSDK_NGX_Result(NVSDK_CONV*)(Shutdown,
                                                    NVSDK_NGX_CUDADevice*);
// The pre-device forms, on runtimes that predate the "1" variants. Each of those inserts an
// NVSDK_NGX_CUDADevice* in a different place, so a symbol-level fallback cannot express this and
// LoadVersionedFunction - which returns one Function type - is the wrong shape for it.
using InitLegacy = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long, const wchar_t*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using CreateLegacy = NVSDK_NGX_Result(NVSDK_CONV*)(
    NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using ShutdownLegacy = NVSDK_NGX_Result(NVSDK_CONV*)();
using ThunkInitLegacy = NVSDK_NGX_Result(NVSDK_CONV*)(
    InitLegacy, unsigned long long, const wchar_t*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using ThunkCreateLegacy = NVSDK_NGX_Result(NVSDK_CONV*)(
    CreateLegacy, NVSDK_NGX_Feature, const NVSDK_NGX_Parameter*,
    NVSDK_NGX_Handle**);
using ThunkShutdownLegacy = NVSDK_NGX_Result(NVSDK_CONV*)(ShutdownLegacy);
using ThunkRequirements = NVSDK_NGX_Result(NVSDK_CONV*)(
    Requirements, int, const NVSDK_NGX_FeatureDiscoveryInfo*,
    NVSDK_NGX_FeatureRequirement*);
using ThunkEvaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
    Evaluate, const NVSDK_NGX_Handle*, const NVSDK_NGX_Parameter*,
    PFN_NVSDK_NGX_ProgressCallback);

struct CudaArrayDescriptor {
  size_t width;
  size_t height;
  int format;
  unsigned int channels;
};

struct CudaArray3DDescriptor {
  size_t width;
  size_t height;
  size_t depth;
  int format;
  unsigned int channels;
  unsigned int flags;
};

struct CudaResourceDesc {
  int resource_type;
  union {
    struct {
      void* array;
    } array;
    struct {
      uint64_t device_pointer;
      int format;
      unsigned int channels;
      size_t size_bytes;
    } linear;
    struct {
      uint64_t device_pointer;
      int format;
      unsigned int channels;
      size_t width;
      size_t height;
      size_t pitch_bytes;
    } pitch2d;
    int reserved[32];
  } resource;
  unsigned int flags;
};

struct CudaTextureDesc {
  int address_mode[3];
  int filter_mode;
  unsigned int flags;
  unsigned int max_anisotropy;
  int mipmap_filter_mode;
  float mipmap_level_bias;
  float min_mipmap_level_clamp;
  float max_mipmap_level_clamp;
  float border_color[4];
  int reserved[12];
};

enum CudaMemoryType {
  kCudaMemoryTypeHost = 1,
  kCudaMemoryTypeArray = 3,
};

struct CudaMemcpy2D {
  size_t source_x_bytes;
  size_t source_y;
  int source_memory_type;
  const void* source_host;
  uint64_t source_device;
  void* source_array;
  size_t source_pitch;
  size_t destination_x_bytes;
  size_t destination_y;
  int destination_memory_type;
  void* destination_host;
  uint64_t destination_device;
  void* destination_array;
  size_t destination_pitch;
  size_t width_bytes;
  size_t height;
};

using CuArrayCreate = int(NVSDK_CONV*)(void**, const CudaArrayDescriptor*);
using CuArray3DCreate = int(NVSDK_CONV*)(void**, const CudaArray3DDescriptor*);
using CuArrayDestroy = int(NVSDK_CONV*)(void*);
using CuTexObjectCreate = int(NVSDK_CONV*)(uint64_t*, const CudaResourceDesc*,
                                          const void*, const void*);
using CuTexObjectDestroy = int(NVSDK_CONV*)(uint64_t);
using CuSurfObjectCreate = int(NVSDK_CONV*)(uint64_t*, const CudaResourceDesc*);
using CuSurfObjectDestroy = int(NVSDK_CONV*)(uint64_t);
using CuMemcpy2DFunction = int(NVSDK_CONV*)(const CudaMemcpy2D*);
using CuCtxSynchronize = int(NVSDK_CONV*)();
using SetTracePhase = void(NVSDK_CONV*)(const char*);
using CuInit = int(NVSDK_CONV*)(unsigned int);
using CuDeviceGet = int(NVSDK_CONV*)(int*, int);
using CuDevicePrimaryCtxRetain = int(NVSDK_CONV*)(void**, int);
using CuDevicePrimaryCtxRelease = int(NVSDK_CONV*)(int);
using CuCtxSetCurrent = int(NVSDK_CONV*)(void*);

void NVSDK_CONV NoopTracePhase(const char*) {}

bool EnvironmentEnabled(const char* name) {
  char value[8]{};
  const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
  return length > 0 && length < sizeof(value) && value[0] == '1';
}

bool WriteIoArtifact(const char* filename, const void* data, size_t bytes) {
  char directory[MAX_PATH]{};
  const DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_CUDA_IO_DIR", directory,
      static_cast<DWORD>(sizeof(directory)));
  if (length == 0) return true;
  if (length >= sizeof(directory) || data == nullptr || bytes == 0) return false;
  CreateDirectoryA(directory, nullptr);
  const std::string path = std::string(directory) + "\\" + filename;
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) return false;
  output.write(static_cast<const char*>(data),
               static_cast<std::streamsize>(bytes));
  return output.good();
}

bool CopyHostToArray(CuMemcpy2DFunction copy, void* array, const void* host,
                     size_t row_bytes, size_t height) {
  CudaMemcpy2D descriptor{};
  descriptor.source_memory_type = kCudaMemoryTypeHost;
  descriptor.source_host = host;
  descriptor.source_pitch = row_bytes;
  descriptor.destination_memory_type = kCudaMemoryTypeArray;
  descriptor.destination_array = array;
  descriptor.width_bytes = row_bytes;
  descriptor.height = height;
  return copy(&descriptor) == 0;
}

bool CopyArrayToHost(CuMemcpy2DFunction copy, void* array, void* host,
                     size_t row_bytes, size_t height) {
  CudaMemcpy2D descriptor{};
  descriptor.source_memory_type = kCudaMemoryTypeArray;
  descriptor.source_array = array;
  descriptor.destination_memory_type = kCudaMemoryTypeHost;
  descriptor.destination_host = host;
  descriptor.destination_pitch = row_bytes;
  descriptor.width_bytes = row_bytes;
  descriptor.height = height;
  return copy(&descriptor) == 0;
}

void PopulateDeterministicInputs(std::vector<uint16_t>* color,
                                 std::vector<uint32_t>* depth,
                                 std::vector<uint16_t>* motion,
                                 unsigned int width, unsigned int height) {
  constexpr uint16_t kHalfPalette[] = {
      0x0000,  // 0.0
      0x3000,  // 0.125
      0x3400,  // 0.25
      0x3800,  // 0.5
      0x3A00,  // 0.75
      0x3C00,  // 1.0
  };
  constexpr uint32_t kDepthPalette[] = {
      0x3E800000,  // 0.25f
      0x3F000000,  // 0.5f
      0x3F400000,  // 0.75f
      0x3F800000,  // 1.0f
  };
  color->resize(static_cast<size_t>(width) * height * 4);
  depth->resize(static_cast<size_t>(width) * height);
  motion->assign(static_cast<size_t>(width) * height * 2, 0);
  for (unsigned int y = 0; y < height; ++y) {
    for (unsigned int x = 0; x < width; ++x) {
      const size_t pixel = static_cast<size_t>(y) * width + x;
      (*color)[pixel * 4 + 0] = kHalfPalette[(x / 64) % 6];
      (*color)[pixel * 4 + 1] = kHalfPalette[(y / 64) % 6];
      (*color)[pixel * 4 + 2] = kHalfPalette[((x + y) / 64) % 6];
      (*color)[pixel * 4 + 3] = kHalfPalette[5];
      (*depth)[pixel] = kDepthPalette[(y / 128) % 4];
    }
  }
}

struct HalfOutputStatistics {
  size_t nonzero_elements = 0;
  size_t nonfinite_elements = 0;
  size_t rgb_nonzero_elements = 0;
  size_t alpha_nonzero_elements = 0;
};

HalfOutputStatistics AnalyzeHalfOutput(const std::vector<uint16_t>& output) {
  HalfOutputStatistics statistics{};
  for (size_t index = 0; index < output.size(); ++index) {
    const uint16_t value = output[index];
    if ((value & 0x7FFFu) != 0) ++statistics.nonzero_elements;
    if ((value & 0x7FFFu) != 0) {
      if (index % 4 == 3) {
        ++statistics.alpha_nonzero_elements;
      } else {
        ++statistics.rgb_nonzero_elements;
      }
    }
    if ((value & 0x7C00u) == 0x7C00u) ++statistics.nonfinite_elements;
  }
  return statistics;
}

void PrintResult(const char* name, NVSDK_NGX_Result result) {
  std::cout << name << "=0x" << std::hex << std::uppercase
            << static_cast<uint32_t>(result) << std::dec << '\n';
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  if (argc != 3) {
    std::wcerr << L"usage=dlss-sr-cuda-capture.exe <nvngx_dlss.dll> "
                  L"<quality|performance>\n";
    return 2;
  }
  const bool quality_scenario = wcscmp(argv[2], L"quality") == 0;
  const bool performance_scenario = wcscmp(argv[2], L"performance") == 0;
  if (!quality_scenario && !performance_scenario) {
    std::cerr << "error=invalid_scenario\n";
    return 2;
  }
  const int perf_quality = quality_scenario
                               ? NVSDK_NGX_PerfQuality_Value_MaxQuality
                               : NVSDK_NGX_PerfQuality_Value_MaxPerf;
  const unsigned int render_width = quality_scenario ? 1280u : 960u;
  const unsigned int render_height = quality_scenario ? 720u : 540u;
  constexpr unsigned int kOutputWidth = 1920;
  constexpr unsigned int kOutputHeight = 1080;
  const char* scenario_id = quality_scenario
                                ? "quality-k-1280x720-to-1920x1080-reset0"
                                : "performance-m-960x540-to-1920x1080-reset0";
  HMODULE thunk_module = LoadLibraryW(L"nvngx.dll");
  HMODULE cuda_driver = LoadLibraryW(L"nvcuda.dll");
  HMODULE nvapi_arch = LoadLibraryW(L"nvapi64.dll");
  HMODULE runtime = LoadLibraryW(argv[1]);
  if (thunk_module == nullptr || cuda_driver == nullptr ||
      nvapi_arch == nullptr || runtime == nullptr) {
    std::cerr << "error=module_load_failed win32=" << GetLastError() << '\n';
    return 3;
  }
  const bool real_cuda = EnvironmentEnabled("NEURALSTUDIO_REAL_CUDA");
  std::cout << "runtime.scenario_id=" << scenario_id << '\n';

  const auto get_application_id =
      LoadFunction<GetRuntimeU32>(runtime, "NVSDK_NGX_GetApplicationId");
  const auto get_api_version =
      LoadFunction<GetRuntimeU32>(runtime, "NVSDK_NGX_GetAPIVersion");
  const auto populate = LoadFunction<Populate>(
      runtime, "NVSDK_NGX_CUDA_PopulateParameters_Impl");
  const auto init =
      LoadFunction<Init>(runtime, "NVSDK_NGX_CUDA_Init_Ext1");
  const auto create =
      LoadFunction<Create>(runtime, "NVSDK_NGX_CUDA_CreateFeature1");
  const auto release =
      LoadFunction<Release>(runtime, "NVSDK_NGX_CUDA_ReleaseFeature");
  const auto shutdown =
      LoadFunction<Shutdown>(runtime, "NVSDK_NGX_CUDA_Shutdown1");
  const auto init_legacy =
      LoadFunction<InitLegacy>(runtime, "NVSDK_NGX_CUDA_Init_Ext");
  const auto create_legacy =
      LoadFunction<CreateLegacy>(runtime, "NVSDK_NGX_CUDA_CreateFeature");
  const auto shutdown_legacy =
      LoadFunction<ShutdownLegacy>(runtime, "NVSDK_NGX_CUDA_Shutdown");
  // All three or none: mixing the two ABIs would hand the runtime a device pointer where it wants
  // an SDK version, which does not crash - it records the wrong bytes.
  const bool modern_abi =
      init != nullptr && create != nullptr && shutdown != nullptr;
  const bool legacy_abi = !modern_abi && init_legacy != nullptr &&
                          create_legacy != nullptr &&
                          shutdown_legacy != nullptr;
  std::cout << "runtime.ngx_abi="
            << (modern_abi ? "device" : legacy_abi ? "legacy" : "none") << '\n';
  const auto requirements = LoadFunction<Requirements>(
      runtime, "NVSDK_NGX_CUDA_GetFeatureRequirements");
  const auto evaluate =
      LoadFunction<Evaluate>(runtime, "NVSDK_NGX_CUDA_EvaluateFeature");
  const auto thunk_populate = LoadFunction<ThunkPopulate>(
      thunk_module, "NeuralStudioPopulateCUDA");
  const auto thunk_init =
      LoadFunction<ThunkInit>(thunk_module, "NeuralStudioInitCUDA");
  const auto thunk_create =
      LoadFunction<ThunkCreate>(thunk_module, "NeuralStudioCreateCUDA");
  const auto thunk_release =
      LoadFunction<ThunkRelease>(thunk_module, "NeuralStudioReleaseCUDA");
  const auto thunk_shutdown =
      LoadFunction<ThunkShutdown>(thunk_module, "NeuralStudioShutdownCUDA");
  const auto thunk_init_legacy = LoadFunction<ThunkInitLegacy>(
      thunk_module, "NeuralStudioInitCUDALegacy");
  const auto thunk_create_legacy = LoadFunction<ThunkCreateLegacy>(
      thunk_module, "NeuralStudioCreateCUDALegacy");
  const auto thunk_shutdown_legacy = LoadFunction<ThunkShutdownLegacy>(
      thunk_module, "NeuralStudioShutdownCUDALegacy");
  const auto thunk_requirements = LoadFunction<ThunkRequirements>(
      thunk_module, "NeuralStudioRequirementsCUDA");
  const auto thunk_evaluate = LoadFunction<ThunkEvaluate>(
      thunk_module, "NeuralStudioEvaluateCUDA");
  const auto cu_array_create =
      LoadVersionedFunction<CuArrayCreate>(cuda_driver, "cuArrayCreate_v2",
                                           "cuArrayCreate");
  const auto cu_array_destroy =
      LoadFunction<CuArrayDestroy>(cuda_driver, "cuArrayDestroy");
  const auto cu_array_3d_create = LoadVersionedFunction<CuArray3DCreate>(
      cuda_driver, "cuArray3DCreate_v2", "cuArray3DCreate");
  const auto cu_tex_create =
      LoadFunction<CuTexObjectCreate>(cuda_driver, "cuTexObjectCreate");
  const auto cu_tex_destroy =
      LoadFunction<CuTexObjectDestroy>(cuda_driver, "cuTexObjectDestroy");
  const auto cu_surf_create =
      LoadFunction<CuSurfObjectCreate>(cuda_driver, "cuSurfObjectCreate");
  const auto cu_surf_destroy =
      LoadFunction<CuSurfObjectDestroy>(cuda_driver, "cuSurfObjectDestroy");
  const auto cu_memcpy_2d =
      LoadVersionedFunction<CuMemcpy2DFunction>(cuda_driver, "cuMemcpy2D_v2",
                                                "cuMemcpy2D");
  const auto cu_ctx_synchronize =
      LoadFunction<CuCtxSynchronize>(cuda_driver, "cuCtxSynchronize");
  const auto loaded_trace_phase = LoadFunction<SetTracePhase>(
      cuda_driver, "NeuralStudioCudaTraceSetPhase");
  const SetTracePhase set_trace_phase =
      loaded_trace_phase != nullptr ? loaded_trace_phase : &NoopTracePhase;
  const auto cu_init = LoadFunction<CuInit>(cuda_driver, "cuInit");
  const auto cu_device_get =
      LoadFunction<CuDeviceGet>(cuda_driver, "cuDeviceGet");
  const auto cu_primary_retain = LoadFunction<CuDevicePrimaryCtxRetain>(
      cuda_driver, "cuDevicePrimaryCtxRetain");
  const auto cu_primary_release = LoadFunction<CuDevicePrimaryCtxRelease>(
      cuda_driver, "cuDevicePrimaryCtxRelease");
  const auto cu_ctx_set_current =
      LoadFunction<CuCtxSetCurrent>(cuda_driver, "cuCtxSetCurrent");
  // Named, not lumped. A runtime that is missing one export used to fail identically to one
  // missing twenty, which turned every version mismatch into a bisect against the DLL's export
  // table. Reporting each one by name makes an unfamiliar build diagnosable in a single run.
  struct RequiredExport {
    const char* name;
    bool present;
  };
  const RequiredExport required[] = {
      {"NVSDK_NGX_GetApplicationId", get_application_id != nullptr},
      {"NVSDK_NGX_GetAPIVersion", get_api_version != nullptr},
      {"NVSDK_NGX_CUDA_PopulateParameters_Impl", populate != nullptr},
      {"NVSDK_NGX_CUDA_Init_Ext1 or NVSDK_NGX_CUDA_Init_Ext",
       modern_abi || legacy_abi},
      {"NVSDK_NGX_CUDA_ReleaseFeature", release != nullptr},
      {"NVSDK_NGX_CUDA_EvaluateFeature", evaluate != nullptr},
      {"NeuralStudioPopulateCUDA", thunk_populate != nullptr},
      {"NeuralStudioReleaseCUDA", thunk_release != nullptr},
      {"NeuralStudioEvaluateCUDA", thunk_evaluate != nullptr},
      {"NeuralStudioInitCUDA (device ABI)",
       !modern_abi || thunk_init != nullptr},
      {"NeuralStudioCreateCUDA (device ABI)",
       !modern_abi || thunk_create != nullptr},
      {"NeuralStudioShutdownCUDA (device ABI)",
       !modern_abi || thunk_shutdown != nullptr},
      {"NeuralStudioInitCUDALegacy", !legacy_abi || thunk_init_legacy != nullptr},
      {"NeuralStudioCreateCUDALegacy",
       !legacy_abi || thunk_create_legacy != nullptr},
      {"NeuralStudioShutdownCUDALegacy",
       !legacy_abi || thunk_shutdown_legacy != nullptr},
      {"cuArrayCreate", cu_array_create != nullptr},
      {"cuArray3DCreate", cu_array_3d_create != nullptr},
      {"cuArrayDestroy", cu_array_destroy != nullptr},
      {"cuTexObjectCreate", cu_tex_create != nullptr},
      {"cuTexObjectDestroy", cu_tex_destroy != nullptr},
      {"cuSurfObjectCreate", cu_surf_create != nullptr},
      {"cuSurfObjectDestroy", cu_surf_destroy != nullptr},
      {"cuMemcpy2D", !real_cuda || cu_memcpy_2d != nullptr},
      {"cuCtxSynchronize", !real_cuda || cu_ctx_synchronize != nullptr},
      {"cuInit", !real_cuda || cu_init != nullptr},
      {"cuDeviceGet", !real_cuda || cu_device_get != nullptr},
      {"cuDevicePrimaryCtxRetain", !real_cuda || cu_primary_retain != nullptr},
      {"cuDevicePrimaryCtxRelease", !real_cuda || cu_primary_release != nullptr},
      {"cuCtxSetCurrent", !real_cuda || cu_ctx_set_current != nullptr},
  };
  bool any_missing = false;
  for (const RequiredExport& entry : required) {
    if (!entry.present) {
      std::cerr << "error=missing_export name=" << entry.name << '\n';
      any_missing = true;
    }
  }
  if (any_missing) {
    std::cerr << "error=required_export_missing\n";
    return 4;
  }

  int cuda_device = 0;
  void* cuda_context =
      reinterpret_cast<void*>(static_cast<uintptr_t>(0x1200));
  if (real_cuda) {
    set_trace_phase("harness_setup");
    if (cu_init(0) != 0 || cu_device_get(&cuda_device, 0) != 0 ||
        cu_primary_retain(&cuda_context, cuda_device) != 0 ||
        cu_ctx_set_current(cuda_context) != 0) {
      std::cerr << "error=real_cuda_context_setup_failed\n";
      return 4;
    }
    std::cout << "runtime.real_cuda_context=retained\n";
  }

  set_trace_phase("populate");
  ParameterStore parameters;
  const NVSDK_NGX_Result populate_result =
      thunk_populate(populate, parameters.AsNgx());
  PrintResult("runtime.populate_result", populate_result);
  if (NVSDK_NGX_FAILED(populate_result)) return 5;

  NVSDK_NGX_FeatureDiscoveryInfo discovery{};
  discovery.SDKVersion =
      static_cast<NVSDK_NGX_Version>(get_api_version());
  discovery.FeatureID = NVSDK_NGX_Feature_SuperSampling;
  discovery.Identifier.IdentifierType =
      NVSDK_NGX_Application_Identifier_Type_Application_Id;
  discovery.Identifier.v.ApplicationId = get_application_id();
  discovery.ApplicationDataPath = L"ngx-data";
  NVSDK_NGX_FeatureRequirement requirement{};
  // Optional on purpose: this is a capability probe whose result is printed and never read again,
  // and older runtimes do not export it at all. Requiring it would refuse to record a DLL that is
  // otherwise perfectly readable.
  if (requirements != nullptr && thunk_requirements != nullptr) {
    set_trace_phase("requirements");
    const NVSDK_NGX_Result requirements_result = thunk_requirements(
        requirements, 0, &discovery, &requirement);
    PrintResult("runtime.requirements_result", requirements_result);
    std::cout << "runtime.requirements_min_arch=0x" << std::hex
              << std::uppercase << requirement.MinHWArchitecture << std::dec
              << '\n';
  } else {
    std::cout << "runtime.requirements_result=unavailable\n";
  }

  CreateDirectoryW(L"ngx-data", nullptr);
  NVSDK_NGX_CUDADevice device{};
  device.cudaContext = cuda_context;
  const auto api_version = static_cast<NVSDK_NGX_Version>(get_api_version());
  set_trace_phase("init");
  const NVSDK_NGX_Result init_result =
      modern_abi ? thunk_init(init, get_application_id(), L"ngx-data", &device,
                              api_version, parameters.AsNgx())
                 : thunk_init_legacy(init_legacy, get_application_id(),
                                     L"ngx-data", api_version,
                                     parameters.AsNgx());
  PrintResult("runtime.init_result", init_result);
  if (NVSDK_NGX_FAILED(init_result)) return 6;

  std::cout << "parameter.ledger.begin stage=create\n";
  parameters.SetI32(NVSDK_NGX_Parameter_Width,
                    static_cast<int>(render_width));
  parameters.SetI32(NVSDK_NGX_Parameter_Height,
                    static_cast<int>(render_height));
  parameters.SetI32(NVSDK_NGX_Parameter_OutWidth,
                    static_cast<int>(kOutputWidth));
  parameters.SetI32(NVSDK_NGX_Parameter_OutHeight,
                    static_cast<int>(kOutputHeight));
  parameters.SetI32(NVSDK_NGX_Parameter_PerfQualityValue, perf_quality);
  std::cout << "runtime.perf_quality=" << perf_quality << '\n';
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                    NVSDK_NGX_DLSS_Feature_Flags_IsHDR |
                        NVSDK_NGX_DLSS_Feature_Flags_MVLowRes);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects, 0);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_DLAA,
                    NVSDK_NGX_DLSS_Hint_Render_Preset_K);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Quality,
                    NVSDK_NGX_DLSS_Hint_Render_Preset_K);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Balanced,
                    NVSDK_NGX_DLSS_Hint_Render_Preset_K);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_Performance,
                    NVSDK_NGX_DLSS_Hint_Render_Preset_M);
  parameters.SetI32(
      NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraPerformance,
      NVSDK_NGX_DLSS_Hint_Render_Preset_L);
  parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Hint_Render_Preset_UltraQuality,
                    NVSDK_NGX_DLSS_Hint_Render_Preset_K);
  parameters.SetI32(NVSDK_NGX_Parameter_CreationNodeMask, 1);
  parameters.SetI32(NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
  parameters.SetI32(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);
  std::cout << "parameter.ledger.end stage=create\n";

  NVSDK_NGX_Handle* handle = nullptr;
  set_trace_phase("create");
  const NVSDK_NGX_Result create_result =
      modern_abi
          ? thunk_create(create, &device, NVSDK_NGX_Feature_SuperSampling,
                         parameters.AsNgx(), &handle)
          : thunk_create_legacy(create_legacy,
                                NVSDK_NGX_Feature_SuperSampling,
                                parameters.AsNgx(), &handle);
  PrintResult("runtime.create_result", create_result);
  std::cout << "runtime.feature_handle="
            << (handle != nullptr ? "acquired" : "null") << '\n';

  if (NVSDK_NGX_SUCCEED(create_result) && handle != nullptr) {
    if (real_cuda && cu_ctx_set_current(cuda_context) != 0) {
      std::cerr << "error=restore_cuda_context_before_resources_failed\n";
      return 7;
    }
    set_trace_phase("input_setup");
    CudaArrayDescriptor color_rgba16{render_width, render_height, 16, 4};
    CudaArray3DDescriptor output_rgba16{
        kOutputWidth, kOutputHeight, 0, 16, 4, 2};
    CudaArrayDescriptor depth_r32{render_width, render_height, 32, 1};
    CudaArrayDescriptor motion_rg16{render_width, render_height, 16, 2};
    CudaArrayDescriptor exposure_r32{1, 1, 32, 1};
    void* color = nullptr;
    void* output = nullptr;
    void* depth = nullptr;
    void* motion = nullptr;
    void* exposure = nullptr;
    if (cu_array_create(&color, &color_rgba16) != 0 ||
        cu_array_3d_create(&output, &output_rgba16) != 0 ||
        cu_array_create(&depth, &depth_r32) != 0 ||
        cu_array_create(&motion, &motion_rg16) != 0 ||
        cu_array_create(&exposure, &exposure_r32) != 0) {
      std::cerr << "error=cuda_array_creation_failed\n";
      return 7;
    }
    std::vector<uint16_t> color_host;
    std::vector<uint32_t> depth_host;
    std::vector<uint16_t> motion_host;
    std::vector<uint16_t> output_host;
    const uint32_t exposure_host = 0x3F800000u;
    if (real_cuda) {
      PopulateDeterministicInputs(&color_host, &depth_host, &motion_host,
                                  render_width, render_height);
      output_host.assign(
          static_cast<size_t>(kOutputWidth) * kOutputHeight * 4, 0);
      if (!WriteIoArtifact("input-color.rgba16le", color_host.data(),
                           color_host.size() * sizeof(color_host[0])) ||
          !WriteIoArtifact("input-depth.r32le", depth_host.data(),
                           depth_host.size() * sizeof(depth_host[0])) ||
          !WriteIoArtifact("input-motion.rg16le", motion_host.data(),
                           motion_host.size() * sizeof(motion_host[0])) ||
          !WriteIoArtifact("input-exposure.r32le", &exposure_host,
                           sizeof(exposure_host)) ||
          !CopyHostToArray(cu_memcpy_2d, color, color_host.data(),
                           static_cast<size_t>(render_width) * 4 *
                               sizeof(uint16_t),
                           render_height) ||
          !CopyHostToArray(cu_memcpy_2d, depth, depth_host.data(),
                           static_cast<size_t>(render_width) * sizeof(uint32_t),
                           render_height) ||
          !CopyHostToArray(cu_memcpy_2d, motion, motion_host.data(),
                           static_cast<size_t>(render_width) * 2 *
                               sizeof(uint16_t),
                           render_height) ||
          !CopyHostToArray(cu_memcpy_2d, exposure, &exposure_host,
                           sizeof(exposure_host), 1) ||
          !CopyHostToArray(cu_memcpy_2d, output, output_host.data(),
                           static_cast<size_t>(kOutputWidth) * 4 *
                               sizeof(uint16_t),
                           kOutputHeight)) {
        std::cerr << "error=deterministic_cuda_array_setup_failed\n";
        return 8;
      }
      std::cout << "runtime.deterministic_inputs="
                   "rgba16-tile-v1_depth-bands-v1_motion-zero_exposure-one\n";
      std::cout << "runtime.deterministic_input_bytes="
                << color_host.size() * sizeof(color_host[0]) +
                       depth_host.size() * sizeof(depth_host[0]) +
                       motion_host.size() * sizeof(motion_host[0]) +
                       sizeof(exposure_host)
                << '\n';
    }
    CudaResourceDesc color_resource{};
    CudaResourceDesc output_resource{};
    CudaResourceDesc depth_resource{};
    CudaResourceDesc motion_resource{};
    CudaResourceDesc exposure_resource{};
    color_resource.resource.array.array = color;
    output_resource.resource.array.array = output;
    depth_resource.resource.array.array = depth;
    motion_resource.resource.array.array = motion;
    exposure_resource.resource.array.array = exposure;
    uint64_t color_texture = 0;
    uint64_t output_surface = 0;
    uint64_t depth_texture = 0;
    uint64_t motion_texture = 0;
    uint64_t exposure_texture = 0;
    CudaTextureDesc texture_desc{};
    texture_desc.address_mode[0] = 1;
    texture_desc.address_mode[1] = 1;
    texture_desc.address_mode[2] = 1;
    texture_desc.filter_mode = 1;
    texture_desc.flags = 2;
    if (cu_tex_create(&color_texture, &color_resource, &texture_desc, nullptr) != 0 ||
        cu_surf_create(&output_surface, &output_resource) != 0 ||
        cu_tex_create(&depth_texture, &depth_resource, &texture_desc, nullptr) != 0 ||
        cu_tex_create(&motion_texture, &motion_resource, &texture_desc, nullptr) != 0 ||
        cu_tex_create(&exposure_texture, &exposure_resource, &texture_desc,
                      nullptr) != 0) {
      std::cerr << "error=cuda_texture_creation_failed\n";
      return 8;
    }
    std::cout << "parameter.ledger.begin stage=evaluate\n";
    parameters.SetPointerValue(NVSDK_NGX_Parameter_Color, &color_texture);
    parameters.SetPointerValue(NVSDK_NGX_Parameter_Output, &output_surface);
    parameters.SetPointerValue(NVSDK_NGX_Parameter_Depth, &depth_texture);
    parameters.SetPointerValue(NVSDK_NGX_Parameter_MotionVectors,
                               &motion_texture);
    parameters.SetPointerValue(NVSDK_NGX_Parameter_ExposureTexture,
                               &exposure_texture);
    parameters.SetPointerValue(NVSDK_NGX_Parameter_TransparencyMask, nullptr);
    parameters.SetPointerValue(
        NVSDK_NGX_Parameter_DLSS_Input_Bias_Current_Color_Mask, nullptr);
    parameters.SetF32(NVSDK_NGX_Parameter_Jitter_Offset_X, 0.0f);
    parameters.SetF32(NVSDK_NGX_Parameter_Jitter_Offset_Y, 0.0f);
    parameters.SetF32(NVSDK_NGX_Parameter_Sharpness, 0.0f);
    parameters.SetI32(NVSDK_NGX_Parameter_Reset, 1);
    parameters.SetF32(NVSDK_NGX_Parameter_MV_Scale_X, 1.0f);
    parameters.SetF32(NVSDK_NGX_Parameter_MV_Scale_Y, 1.0f);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                      render_width);
    parameters.SetU32(
        NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
        render_height);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_X, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_Color_Subrect_Base_Y, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_X, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_Depth_Subrect_Base_Y, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_X, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Input_MV_SubrectBase_Y, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_X, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_DLSS_Output_Subrect_Base_Y, 0);
    parameters.SetF32(NVSDK_NGX_Parameter_DLSS_Pre_Exposure, 1.0f);
    parameters.SetF32(NVSDK_NGX_Parameter_DLSS_Exposure_Scale, 1.0f);
    parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_X_Axis, 0);
    parameters.SetI32(NVSDK_NGX_Parameter_DLSS_Indicator_Invert_Y_Axis, 0);
    std::cout << "resource.bind Color=input_color\n"
                 "resource.bind Output=output_color\n"
                 "resource.bind Depth=input_depth\n"
                 "resource.bind MotionVectors=input_motion\n"
                 "resource.bind ExposureTexture=input_exposure\n";
    std::cout << "parameter.ledger.end stage=evaluate\n";

    set_trace_phase("evaluate");
    const NVSDK_NGX_Result evaluate_result = thunk_evaluate(
        evaluate, handle, parameters.AsNgx(), nullptr);
    PrintResult("runtime.evaluate_result", evaluate_result);
    bool output_readback_valid = !real_cuda;
    if (real_cuda && NVSDK_NGX_SUCCEED(evaluate_result)) {
      if (cu_ctx_set_current(cuda_context) == 0 &&
          cu_ctx_synchronize() == 0 &&
          CopyArrayToHost(cu_memcpy_2d, output, output_host.data(),
                          static_cast<size_t>(kOutputWidth) * 4 *
                              sizeof(uint16_t),
                          kOutputHeight) &&
          WriteIoArtifact("output.rgba16le", output_host.data(),
                          output_host.size() * sizeof(output_host[0]))) {
        const HalfOutputStatistics statistics = AnalyzeHalfOutput(output_host);
        std::cout << "runtime.output_readback_bytes="
                  << output_host.size() * sizeof(output_host[0]) << '\n';
        std::cout << "runtime.output_nonzero_elements="
                  << statistics.nonzero_elements << '\n';
        std::cout << "runtime.output_nonfinite_elements="
                  << statistics.nonfinite_elements << '\n';
        std::cout << "runtime.output_rgb_nonzero_elements="
                  << statistics.rgb_nonzero_elements << '\n';
        std::cout << "runtime.output_alpha_nonzero_elements="
                  << statistics.alpha_nonzero_elements << '\n';
        output_readback_valid = statistics.nonfinite_elements == 0;
      }
    }
    set_trace_phase("release");
    if (real_cuda) cu_ctx_set_current(cuda_context);
    cu_tex_destroy(exposure_texture);
    cu_tex_destroy(motion_texture);
    cu_tex_destroy(depth_texture);
    cu_surf_destroy(output_surface);
    cu_tex_destroy(color_texture);
    cu_array_destroy(exposure);
    cu_array_destroy(motion);
    cu_array_destroy(depth);
    cu_array_destroy(output);
    cu_array_destroy(color);
    PrintResult("runtime.release_result", thunk_release(release, handle));
    if (NVSDK_NGX_FAILED(evaluate_result)) {
      std::cerr << "error=dlss_sr_evaluation_failed\n";
      return 9;
    }
    if (!output_readback_valid) {
      std::cerr << "error=dlss_sr_output_readback_invalid\n";
      return 10;
    }
  }
  set_trace_phase("shutdown");
  PrintResult("runtime.shutdown_result",
               modern_abi ? thunk_shutdown(shutdown, &device)
                          : thunk_shutdown_legacy(shutdown_legacy));
  if (real_cuda) {
    cu_ctx_set_current(nullptr);
    const int release_result = cu_primary_release(cuda_device);
    std::cout << "runtime.cuda_primary_release=" << release_result << '\n';
  }
  std::cout << "probe=dlss_sr_cuda_capture\n";
  std::cout << "result=observed\n";

  FreeLibrary(runtime);
  FreeLibrary(nvapi_arch);
  FreeLibrary(cuda_driver);
  FreeLibrary(thunk_module);
  return 0;
}
