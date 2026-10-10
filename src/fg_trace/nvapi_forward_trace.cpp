#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>

#include "d3d12_descriptor_trace.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <limits>
#include <string>
#include <type_traits>
#include <vector>

#pragma comment(lib, "bcrypt.lib")

// This proxy intentionally carries a private, minimal transcription of the
// pinned NvAPI D3D12 CUDA ABI. Including the full SDK header would couple the
// capture DLL to unrelated SDK revisions and preprocessor state.
using NvStatus = std::int32_t;
using NvObjectHandle = void*;

constexpr NvStatus kNvapiOk = 0;
constexpr NvStatus kNvapiError = -1;
constexpr std::uint32_t kMaxEnumeratedFunctions = 65536;
constexpr std::uint32_t kMaxKernelChainLength = 65536;
constexpr std::size_t kMaxTrackedModules = 1024;
constexpr std::size_t kMaxTrackedFunctions = 16384;
constexpr std::size_t kMaxTrackedShaders = 4096;
constexpr std::size_t kMaxTrackedResources = 16384;

struct CpuDescriptorHandle {
  std::uintptr_t ptr;
};

struct CubinShaderParams {
  std::size_t struct_size_in;
  std::size_t struct_size_out;
  void* device;
  const void* cubin;
  std::uint32_t size;
  std::uint32_t block_x;
  std::uint32_t block_y;
  std::uint32_t block_z;
  std::uint32_t dynamic_shared_bytes;
  const char* shader_name;
  std::uint32_t flags;
  NvObjectHandle shader;
};

struct MergedTextureSamplerParams {
  std::size_t struct_size_in;
  std::size_t struct_size_out;
  void* device;
  CpuDescriptorHandle texture_descriptor;
  CpuDescriptorHandle sampler_descriptor;
  std::uint64_t texture_handle;
};

enum class IndependentDescriptorType : std::int32_t {
  kSurface = 0,
  kTexture = 1,
  kSampler = 2,
};

struct IndependentDescriptorParams {
  std::size_t struct_size_in;
  std::size_t struct_size_out;
  void* device;
  IndependentDescriptorType type;
  CpuDescriptorHandle descriptor;
  std::uint64_t handle;
};

struct Dim3 {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t z;
};

struct KernelLaunchParams {
  NvObjectHandle function;
  Dim3 grid;
  Dim3 block;
  std::uint32_t dynamic_shared_bytes;
  const void* params;
  std::uint32_t param_size;
};

struct KernelLaunchParamsEx {
  NvObjectHandle function;
  Dim3 grid;
  Dim3 block;
  std::uint32_t dynamic_shared_bytes;
  const void* params;
  std::uint32_t param_size;
  void** kernel_params;
};

struct UavInfoV2 {
  std::uint32_t version;
  std::uint32_t surface_handle;
  std::uint64_t gpu_va_start;
  std::uint64_t gpu_va_size;
  std::uint64_t out_flags;
};

static_assert(sizeof(CpuDescriptorHandle) == sizeof(std::uintptr_t));
static_assert(sizeof(NvObjectHandle) == sizeof(void*));
static_assert(sizeof(bool) == 1);
static_assert(sizeof(CubinShaderParams) == 80);
static_assert(sizeof(MergedTextureSamplerParams) == 48);
static_assert(sizeof(IndependentDescriptorParams) == 48);
static_assert(sizeof(Dim3) == 12);
static_assert(sizeof(KernelLaunchParams) == 56);
static_assert(sizeof(KernelLaunchParamsEx) == 64);
static_assert(sizeof(UavInfoV2) == 32);

extern "C" {
__declspec(dllexport) void* __cdecl nvapi_QueryInterface(std::uint32_t id);
__declspec(dllexport) void* __cdecl nvapi_Direct_GetMethod(std::uint32_t id);
__declspec(dllexport) void WINAPI NeuralStudioFgNvapiTraceSetPhase(
    const char* phase);
}

namespace {

constexpr std::uint32_t kIdCreateCubinExV2 = 0x299f5fdc;
constexpr std::uint32_t kIdCreateCubin = 0x2a2c79e8;
constexpr std::uint32_t kIdCreateCubinEx = 0x3151211b;
constexpr std::uint32_t kIdCreateCubinWithName = 0x1dc7261f;
constexpr std::uint32_t kIdLaunchCubin = 0x5c52bb86;
constexpr std::uint32_t kIdDestroyCubin = 0x7fb785ba;
constexpr std::uint32_t kIdGetMergedTextureSampler = 0x329fe6e0;
constexpr std::uint32_t kIdGetIndependentDescriptor = 0x0ddac234;
constexpr std::uint32_t kIdGetTextureObject = 0x80403fc9;
constexpr std::uint32_t kIdGetSurfaceObject = 0x48f5b2ee;
constexpr std::uint32_t kIdIsFatbinPtxSupported = 0x70c07832;
constexpr std::uint32_t kIdCreateCuModule = 0xad1a677d;
constexpr std::uint32_t kIdEnumFunctions = 0x7ab88d88;
constexpr std::uint32_t kIdCreateCuFunction = 0xe2436e22;
constexpr std::uint32_t kIdLaunchKernelChain = 0x24973538;
constexpr std::uint32_t kIdLaunchKernelChainEx = 0x846a9bf0;
constexpr std::uint32_t kIdDestroyCuModule = 0x41c65285;
constexpr std::uint32_t kIdDestroyCuFunction = 0xdf295ea6;
constexpr std::uint32_t kIdCaptureUavInfo = 0x6e5ea9db;
// This resolved ID is absent from the pinned public NvAPI interface table.
// Keep it opaque: the trace records whether the returned function is invoked
// without guessing its signature or semantics.
constexpr std::uint32_t kIdOpaqueAd298d3f = 0xad298d3f;

NvStatus __cdecl TraceCreateCubinExV2(CubinShaderParams* params);
NvStatus __cdecl TraceCreateCubin(void* device, const void* cubin,
                                  std::uint32_t size, std::uint32_t block_x,
                                  std::uint32_t block_y,
                                  std::uint32_t block_z,
                                  NvObjectHandle* shader);
NvStatus __cdecl TraceCreateCubinEx(
    void* device, const void* cubin, std::uint32_t size,
    std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
    std::uint32_t dynamic_shared_bytes, const char* shader_name,
    NvObjectHandle* shader);
NvStatus __cdecl TraceCreateCubinWithName(
    void* device, const void* cubin, std::uint32_t size,
    std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
    const char* shader_name, NvObjectHandle* shader);
NvStatus __cdecl TraceLaunchCubin(void* command_list, NvObjectHandle shader,
                                  std::uint32_t grid_x,
                                  std::uint32_t grid_y,
                                  std::uint32_t grid_z, const void* params,
                                  std::uint32_t param_size);
NvStatus __cdecl TraceDestroyCubin(void* device, NvObjectHandle shader);
NvStatus __cdecl TraceGetMergedTextureSampler(
    MergedTextureSamplerParams* params);
NvStatus __cdecl TraceGetIndependentDescriptor(
    IndependentDescriptorParams* params);
NvStatus __cdecl TraceGetTextureObject(
    void* device, CpuDescriptorHandle texture_descriptor,
    CpuDescriptorHandle sampler_descriptor, std::uint32_t* texture_handle);
NvStatus __cdecl TraceGetSurfaceObject(
    void* device, CpuDescriptorHandle uav_descriptor,
    std::uint32_t* surface_handle);
NvStatus __cdecl TraceIsFatbinPtxSupported(void* device, bool* supported);
NvStatus __cdecl TraceCreateCuModule(void* device, const void* blob,
                                     std::uint32_t size,
                                     NvObjectHandle* module);
NvStatus __cdecl TraceEnumFunctions(void* device, NvObjectHandle module,
                                    std::uint32_t* array_size,
                                    const char** const function_names);
NvStatus __cdecl TraceCreateCuFunction(void* device, NvObjectHandle module,
                                       const char* name,
                                       NvObjectHandle* function);
NvStatus __cdecl TraceLaunchKernelChain(void* command_list,
                                        const KernelLaunchParams* kernels,
                                        std::uint32_t kernel_count);
NvStatus __cdecl TraceLaunchKernelChainEx(void* command_list,
                                          const KernelLaunchParamsEx* kernels,
                                          std::uint32_t kernel_count);
NvStatus __cdecl TraceDestroyCuModule(void* device, NvObjectHandle module);
NvStatus __cdecl TraceDestroyCuFunction(void* device,
                                        NvObjectHandle function);
NvStatus __cdecl TraceCaptureUavInfo(void* device, UavInfoV2* info);

template <typename Destination, typename Source>
Destination PointerCast(Source source) {
  static_assert(sizeof(Destination) == sizeof(Source));
  Destination destination{};
  std::memcpy(&destination, &source, sizeof(destination));
  return destination;
}

struct InterfaceBinding {
  std::uint32_t id;
  const char* name;
  void* wrapper;
  void* volatile real;
};

#define INTERFACE_BINDING(id, name, wrapper) \
  {id, name, PointerCast<void*>(&wrapper), nullptr}
InterfaceBinding g_bindings[] = {
    INTERFACE_BINDING(kIdCreateCubinExV2,
                      "NvAPI_D3D12_CreateCubinComputeShaderExV2",
                      TraceCreateCubinExV2),
    INTERFACE_BINDING(kIdCreateCubin,
                      "NvAPI_D3D12_CreateCubinComputeShader",
                      TraceCreateCubin),
    INTERFACE_BINDING(kIdCreateCubinEx,
                      "NvAPI_D3D12_CreateCubinComputeShaderEx",
                      TraceCreateCubinEx),
    INTERFACE_BINDING(kIdCreateCubinWithName,
                      "NvAPI_D3D12_CreateCubinComputeShaderWithName",
                      TraceCreateCubinWithName),
    INTERFACE_BINDING(kIdLaunchCubin, "NvAPI_D3D12_LaunchCubinShader",
                      TraceLaunchCubin),
    INTERFACE_BINDING(kIdDestroyCubin,
                      "NvAPI_D3D12_DestroyCubinComputeShader",
                      TraceDestroyCubin),
    INTERFACE_BINDING(kIdGetMergedTextureSampler,
                      "NvAPI_D3D12_GetCudaMergedTextureSamplerObject",
                      TraceGetMergedTextureSampler),
    INTERFACE_BINDING(kIdGetIndependentDescriptor,
                      "NvAPI_D3D12_GetCudaIndependentDescriptorObject",
                      TraceGetIndependentDescriptor),
    INTERFACE_BINDING(kIdGetTextureObject,
                      "NvAPI_D3D12_GetCudaTextureObject",
                      TraceGetTextureObject),
    INTERFACE_BINDING(kIdGetSurfaceObject,
                      "NvAPI_D3D12_GetCudaSurfaceObject",
                      TraceGetSurfaceObject),
    INTERFACE_BINDING(kIdIsFatbinPtxSupported,
                      "NvAPI_D3D12_IsFatbinPTXSupported",
                      TraceIsFatbinPtxSupported),
    INTERFACE_BINDING(kIdCreateCuModule, "NvAPI_D3D12_CreateCuModule",
                      TraceCreateCuModule),
    INTERFACE_BINDING(kIdEnumFunctions,
                      "NvAPI_D3D12_EnumFunctionsInModule",
                      TraceEnumFunctions),
    INTERFACE_BINDING(kIdCreateCuFunction,
                      "NvAPI_D3D12_CreateCuFunction",
                      TraceCreateCuFunction),
    INTERFACE_BINDING(kIdLaunchKernelChain,
                      "NvAPI_D3D12_LaunchCuKernelChain",
                      TraceLaunchKernelChain),
    INTERFACE_BINDING(kIdLaunchKernelChainEx,
                      "NvAPI_D3D12_LaunchCuKernelChainEx",
                      TraceLaunchKernelChainEx),
    INTERFACE_BINDING(kIdDestroyCuModule, "NvAPI_D3D12_DestroyCuModule",
                      TraceDestroyCuModule),
    INTERFACE_BINDING(kIdDestroyCuFunction,
                      "NvAPI_D3D12_DestroyCuFunction",
                      TraceDestroyCuFunction),
    INTERFACE_BINDING(kIdCaptureUavInfo, "NvAPI_D3D12_CaptureUAVInfo",
                      TraceCaptureUavInfo),
};
#undef INTERFACE_BINDING

struct KnownPassthroughInterface {
  std::uint32_t id;
  const char* name;
};

// These names are copied from the same pinned nvapi_interface.h used to pin
// the wrapped CUDA IDs. They are classification only: their real pointers are
// returned byte-for-byte unchanged.
constexpr KnownPassthroughInterface kKnownPassthroughInterfaces[] = {
    {0x0150e828, "NvAPI_Initialize"},
    {0xd22bdd7e, "NvAPI_Unload"},
    {0x6c2d048c, "NvAPI_GetErrorMessage"},
    {0x01053fa5, "NvAPI_GetInterfaceVersionString"},
    {0xe5ac921f, "NvAPI_EnumPhysicalGPUs"},
    {0xadd604d1, "NvAPI_GetLogicalGPUFromPhysicalGPU"},
    {0x5380ad1a, "NvAPI_GetPhysicalGPUFromGPUID"},
    {0x842b066e, "NvAPI_GPU_GetLogicalGpuInfo"},
    {0xd8265d24, "NvAPI_GPU_GetArchInfo"},
    {0x01e87354, "NvAPI_D3D12_GetGraphicsCapabilities"},
    {0x0694d52e, "NvAPI_DRS_CreateSession"},
    {0xdad9cff8, "NvAPI_DRS_DestroySession"},
    {0x375dbd6b, "NvAPI_DRS_LoadSettings"},
    {0xfcbc7e14, "NvAPI_DRS_SaveSettings"},
    {0xd3ede889, "NvAPI_DRS_LoadSettingsFromFile"},
    {0x2be25df8, "NvAPI_DRS_SaveSettingsToFile"},
    {0xcc176068, "NvAPI_DRS_CreateProfile"},
    {0x17093206, "NvAPI_DRS_DeleteProfile"},
    {0x1c89c5df, "NvAPI_DRS_SetCurrentGlobalProfile"},
    {0x617bff9f, "NvAPI_DRS_GetCurrentGlobalProfile"},
    {0x61cd6fd6, "NvAPI_DRS_GetProfileInfo"},
    {0x16abd3a9, "NvAPI_DRS_SetProfileInfo"},
    {0x7e4a9a0b, "NvAPI_DRS_FindProfileByName"},
    {0xbc371ee0, "NvAPI_DRS_EnumProfiles"},
    {0x1dae4fbc, "NvAPI_DRS_GetNumProfiles"},
    {0x4347a9de, "NvAPI_DRS_CreateApplication"},
    {0xc5ea85a1, "NvAPI_DRS_DeleteApplicationEx"},
    {0x2c694bc6, "NvAPI_DRS_DeleteApplication"},
    {0xed1f8c69, "NvAPI_DRS_GetApplicationInfo"},
    {0x7fa2173a, "NvAPI_DRS_EnumApplications"},
    {0xeee566b2, "NvAPI_DRS_FindApplicationByName"},
    {0x577dd202, "NvAPI_DRS_SetSetting"},
    {0x73bf8338, "NvAPI_DRS_GetSetting"},
    {0xae3039da, "NvAPI_DRS_EnumSettings"},
    {0xf020614a, "NvAPI_DRS_EnumAvailableSettingIds"},
    {0x2ec39f90, "NvAPI_DRS_EnumAvailableSettingValues"},
    {0xcb7309cd, "NvAPI_DRS_GetSettingIdFromName"},
    {0xd61cbe6e, "NvAPI_DRS_GetSettingNameFromId"},
    {0xe4a26362, "NvAPI_DRS_DeleteProfileSetting"},
    {0x5927b094, "NvAPI_DRS_RestoreAllDefaults"},
    {0xfa5f6134, "NvAPI_DRS_RestoreProfileDefault"},
    {0x53f0381e, "NvAPI_DRS_RestoreProfileDefaultSetting"},
    {0xda8466a0, "NvAPI_DRS_GetBaseProfile"},
};

using Dispatcher = void*(__cdecl*)(std::uint32_t);

struct ModuleRecord {
  NvObjectHandle handle;
  std::uint64_t token;
  bool active;
};

struct FunctionRecord {
  NvObjectHandle handle;
  std::uint64_t token;
  std::uint64_t module_token;
  char name[1024];
  bool active;
};

struct ShaderRecord {
  NvObjectHandle handle;
  std::uint64_t token;
  Dim3 block;
  std::uint32_t dynamic_shared_bytes;
  char name[1024];
  bool active;
};

enum class ResourceKind : std::uint32_t {
  kMergedTextureSampler = 1,
  kIndependentSurface = 2,
  kIndependentTexture = 3,
  kIndependentSampler = 4,
  kTexture = 5,
  kSurface = 6,
};

struct ResourceRecord {
  std::uint64_t handle;
  std::uint64_t token;
  ResourceKind kind;
  bool active;
};

struct ObjectInfo {
  bool tracked;
  std::uint64_t token;
  std::uint64_t parent_token;
  Dim3 block;
  std::uint32_t dynamic_shared_bytes;
  char name[1024];
};

struct DumpInfo {
  bool captured;
  std::uint64_t bytes;
  char file[160];
  char sha256[65];
};

HMODULE g_self = nullptr;
HMODULE g_real_nvapi = nullptr;
Dispatcher g_real_query_interface = nullptr;
Dispatcher g_real_direct_get_method = nullptr;
INIT_ONCE g_runtime_once = INIT_ONCE_STATIC_INIT;
bool g_runtime_ready = false;
std::wstring g_trace_path;
std::wstring g_volatile_trace_path;
std::wstring g_dump_directory;
char g_real_nvapi_sha256[65]{};
char g_current_phase[129] = "unmarked";
SRWLOCK g_trace_lock = SRWLOCK_INIT;
SRWLOCK g_state_lock = SRWLOCK_INIT;
ModuleRecord g_modules[kMaxTrackedModules]{};
FunctionRecord g_functions[kMaxTrackedFunctions]{};
ShaderRecord g_shaders[kMaxTrackedShaders]{};
ResourceRecord g_resources[kMaxTrackedResources]{};
volatile LONG64 g_next_module_token = 0;
volatile LONG64 g_next_function_token = 0;
volatile LONG64 g_next_shader_token = 0;
volatile LONG64 g_next_resource_token = 0;
volatile LONG64 g_next_launch_token = 0;
volatile LONG64 g_next_uav_info_token = 0;
PVOID volatile g_opaque_ad298d3f_thunk = nullptr;
SIZE_T g_opaque_ad298d3f_page_size = 0;
volatile LONG64 g_opaque_ad298d3f_flushed_count = 0;

std::uint64_t NextToken(volatile LONG64* counter) {
  return static_cast<std::uint64_t>(InterlockedIncrement64(counter));
}

InterfaceBinding* FindBinding(std::uint32_t id) {
  for (auto& binding : g_bindings) {
    if (binding.id == id) return &binding;
  }
  return nullptr;
}

const char* FindKnownPassthroughName(std::uint32_t id) {
  for (const auto& known : kKnownPassthroughInterfaces) {
    if (known.id == id) return known.name;
  }
  return nullptr;
}

std::wstring GetEnvironmentString(const wchar_t* name) {
  const DWORD required = GetEnvironmentVariableW(name, nullptr, 0);
  if (required == 0) return {};
  std::vector<wchar_t> buffer(required);
  const DWORD copied = GetEnvironmentVariableW(name, buffer.data(), required);
  if (copied == 0 || copied >= required) return {};
  return std::wstring(buffer.data(), copied);
}

bool IsAbsoluteWindowsPath(const std::wstring& path) {
  if (path.size() >= 3 && ((path[0] >= L'A' && path[0] <= L'Z') ||
                           (path[0] >= L'a' && path[0] <= L'z')) &&
      path[1] == L':' && (path[2] == L'\\' || path[2] == L'/')) {
    return true;
  }
  return path.size() >= 3 && path[0] == L'\\' && path[1] == L'\\' &&
         path[2] != L'\\';
}

std::wstring ParentPath(const std::wstring& path) {
  const std::size_t separator = path.find_last_of(L"\\/");
  if (separator == std::wstring::npos) return {};
  if (separator == 2 && path.size() >= 3 && path[1] == L':') {
    return path.substr(0, 3);
  }
  return path.substr(0, separator);
}

std::wstring BaseName(const std::wstring& path) {
  const std::size_t separator = path.find_last_of(L"\\/");
  return separator == std::wstring::npos ? path : path.substr(separator + 1);
}

bool IsDirectory(const std::wstring& path) {
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool IsFile(const std::wstring& path) {
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) == 0;
}

bool BytesToHex(const unsigned char* bytes, std::size_t size, char* output,
                std::size_t output_size) {
  static constexpr char kHex[] = "0123456789ABCDEF";
  if (bytes == nullptr || output == nullptr || output_size < size * 2 + 1) {
    return false;
  }
  for (std::size_t index = 0; index < size; ++index) {
    output[index * 2] = kHex[bytes[index] >> 4];
    output[index * 2 + 1] = kHex[bytes[index] & 0x0f];
  }
  output[size * 2] = '\0';
  return true;
}

bool HashBuffer(const void* data, std::size_t size, char output[65]) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0;
  DWORD hash_bytes = 0;
  DWORD result_bytes = 0;
  bool success = false;
  std::vector<unsigned char> object;
  std::vector<unsigned char> digest;

  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                        reinterpret_cast<PUCHAR>(&object_bytes),
                        sizeof(object_bytes), &result_bytes, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                        reinterpret_cast<PUCHAR>(&hash_bytes),
                        sizeof(hash_bytes), &result_bytes, 0) < 0 ||
      hash_bytes != 32) {
    goto cleanup;
  }

  object.resize(object_bytes);
  digest.resize(hash_bytes);
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr,
                       0, 0) < 0) {
    goto cleanup;
  }

  {
    const unsigned char* cursor = static_cast<const unsigned char*>(data);
    std::size_t remaining = size;
    while (remaining > 0) {
      constexpr ULONG kMaxHashChunk = (std::numeric_limits<ULONG>::max)();
      const ULONG chunk = remaining > kMaxHashChunk
                              ? kMaxHashChunk
                              : static_cast<ULONG>(remaining);
      if (BCryptHashData(hash, const_cast<PUCHAR>(cursor), chunk, 0) < 0) {
        goto cleanup;
      }
      cursor += chunk;
      remaining -= chunk;
    }
  }
  if (BCryptFinishHash(hash, digest.data(), hash_bytes, 0) < 0 ||
      !BytesToHex(digest.data(), digest.size(), output, 65)) {
    goto cleanup;
  }
  success = true;

cleanup:
  if (hash != nullptr) BCryptDestroyHash(hash);
  if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
  return success;
}

bool HashFile(const std::wstring& path, char output[65]) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_READ,
                            FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                            OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;

  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0;
  DWORD hash_bytes = 0;
  DWORD result_bytes = 0;
  bool success = false;
  std::vector<unsigned char> object;
  std::vector<unsigned char> digest;
  std::vector<unsigned char> buffer(1024 * 1024);

  if (BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM, nullptr,
                                  0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                        reinterpret_cast<PUCHAR>(&object_bytes),
                        sizeof(object_bytes), &result_bytes, 0) < 0 ||
      BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                        reinterpret_cast<PUCHAR>(&hash_bytes),
                        sizeof(hash_bytes), &result_bytes, 0) < 0 ||
      hash_bytes != 32) {
    goto cleanup;
  }

  object.resize(object_bytes);
  digest.resize(hash_bytes);
  if (BCryptCreateHash(algorithm, &hash, object.data(), object_bytes, nullptr,
                       0, 0) < 0) {
    goto cleanup;
  }

  while (true) {
    DWORD bytes_read = 0;
    if (!ReadFile(file, buffer.data(), static_cast<DWORD>(buffer.size()),
                  &bytes_read, nullptr)) {
      goto cleanup;
    }
    if (bytes_read == 0) break;
    if (BCryptHashData(hash, buffer.data(), bytes_read, 0) < 0) goto cleanup;
  }
  if (BCryptFinishHash(hash, digest.data(), hash_bytes, 0) < 0 ||
      !BytesToHex(digest.data(), digest.size(), output, 65)) {
    goto cleanup;
  }
  success = true;

cleanup:
  if (hash != nullptr) BCryptDestroyHash(hash);
  if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
  CloseHandle(file);
  return success;
}

bool ExpectedHashMatches(const std::wstring& expected, const char actual[65]) {
  if (expected.size() != 64) return false;
  for (std::size_t index = 0; index < expected.size(); ++index) {
    wchar_t value = expected[index];
    if (value >= L'a' && value <= L'f') value -= L'a' - L'A';
    const bool valid = (value >= L'0' && value <= L'9') ||
                       (value >= L'A' && value <= L'F');
    if (!valid || value != static_cast<unsigned char>(actual[index])) {
      return false;
    }
  }
  return true;
}

std::string JsonQuote(const char* value) {
  if (value == nullptr) return "null";
  std::string result;
  result.push_back('"');
  for (std::size_t index = 0; value[index] != '\0' && index < 16384;
       ++index) {
    const unsigned char character =
        static_cast<unsigned char>(value[index]);
    switch (character) {
      case '"': result += "\\\""; break;
      case '\\': result += "\\\\"; break;
      case '\b': result += "\\b"; break;
      case '\f': result += "\\f"; break;
      case '\n': result += "\\n"; break;
      case '\r': result += "\\r"; break;
      case '\t': result += "\\t"; break;
      default:
        if (character < 0x20) {
          char encoded[7]{};
          sprintf_s(encoded, "\\u%04X", character);
          result += encoded;
        } else {
          result.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  result.push_back('"');
  return result;
}

void AppendString(std::string& json, const char* key, const char* value) {
  json += ",\"";
  json += key;
  json += "\":";
  json += JsonQuote(value);
}

void AppendUnsigned(std::string& json, const char* key,
                    std::uint64_t value) {
  char number[32]{};
  sprintf_s(number, "%llu", static_cast<unsigned long long>(value));
  json += ",\"";
  json += key;
  json += "\":";
  json += number;
}

void AppendHex(std::string& json, const char* key, std::uint64_t value,
               unsigned int digits) {
  char formatted[32]{};
  if (digits == 8) {
    sprintf_s(formatted, "0x%08X", static_cast<unsigned int>(value));
  } else {
    sprintf_s(formatted, "0x%016llX",
              static_cast<unsigned long long>(value));
  }
  AppendString(json, key, formatted);
}

void AppendSigned(std::string& json, const char* key, std::int64_t value) {
  char number[32]{};
  sprintf_s(number, "%lld", static_cast<long long>(value));
  json += ",\"";
  json += key;
  json += "\":";
  json += number;
}

void AppendBoolean(std::string& json, const char* key, bool value) {
  json += ",\"";
  json += key;
  json += "\":";
  json += value ? "true" : "false";
}

void AppendDim3(std::string& json, const char* key, const Dim3& value) {
  char dimensions[96]{};
  sprintf_s(dimensions, ",\"%s\":[%u,%u,%u]", key, value.x, value.y,
            value.z);
  json += dimensions;
}

void FinishJson(std::string& json) { json.push_back('}'); }

bool WriteJsonLine(const std::wstring& path, const std::string& json) {
  if (path.empty()) return false;
  std::string line = json;
  line.push_back('\n');
  AcquireSRWLockExclusive(&g_trace_lock);
  HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA,
                            FILE_SHARE_READ, nullptr, OPEN_EXISTING,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  bool success = false;
  if (file != INVALID_HANDLE_VALUE) {
    DWORD written = 0;
    success = line.size() <= MAXDWORD &&
              WriteFile(file, line.data(), static_cast<DWORD>(line.size()),
                        &written, nullptr) &&
              written == static_cast<DWORD>(line.size()) &&
              FlushFileBuffers(file);
    CloseHandle(file);
  }
  ReleaseSRWLockExclusive(&g_trace_lock);
  return success;
}

bool WriteTrace(const std::string& json) {
  return WriteJsonLine(g_trace_path, json);
}

bool WriteVolatileTrace(const std::string& json) {
  return WriteJsonLine(g_volatile_trace_path, json);
}

void SanitizePhase(const char* phase, char output[129], bool* truncated) {
  constexpr std::size_t kMaxPhaseBytes = 128;
  *truncated = false;
  std::size_t output_index = 0;
  if (phase != nullptr) {
    for (std::size_t input_index = 0; phase[input_index] != '\0';
         ++input_index) {
      if (output_index == kMaxPhaseBytes) {
        *truncated = true;
        break;
      }
      const unsigned char value =
          static_cast<unsigned char>(phase[input_index]);
      const bool allowed =
          (value >= 'a' && value <= 'z') ||
          (value >= 'A' && value <= 'Z') ||
          (value >= '0' && value <= '9') || value == '.' || value == '_' ||
          value == '-' || value == ':' || value == '/';
      output[output_index++] = allowed ? static_cast<char>(value) : '_';
    }
  }
  if (output_index == 0) {
    constexpr char kUnnamed[] = "unnamed";
    std::memcpy(output, kUnnamed, sizeof(kUnnamed));
    return;
  }
  output[output_index] = '\0';
}

void SetCurrentPhase(const char* phase) {
  AcquireSRWLockExclusive(&g_state_lock);
  strcpy_s(g_current_phase, phase);
  ReleaseSRWLockExclusive(&g_state_lock);
}

void GetCurrentPhase(char output[129]) {
  AcquireSRWLockShared(&g_state_lock);
  strcpy_s(output, 129, g_current_phase);
  ReleaseSRWLockShared(&g_state_lock);
}

void DebugReject(const char* reason) {
  std::string message = "NeuralStudio FG NvAPI trace rejected: ";
  message += reason;
  message += "\n";
  OutputDebugStringA(message.c_str());
}

// The opaque interface sentinel is a signature-independent x64 tail thunk.
// It only performs RIP-relative stores and an atomic counter increment before
// jumping to the real NvAPI function. RCX/RDX/R8/R9 and the caller's stack are
// never changed, so all register and stack arguments reach the real function.
// Windows x64 treats arithmetic flags as volatile across a call.
constexpr SIZE_T kOpaqueThunkCodeBytes = 42;
constexpr SIZE_T kOpaqueCountOffset = 0;
constexpr SIZE_T kOpaqueRcxOffset = 8;
constexpr SIZE_T kOpaqueRdxOffset = 16;
constexpr SIZE_T kOpaqueR8Offset = 24;
constexpr SIZE_T kOpaqueR9Offset = 32;
constexpr SIZE_T kOpaqueRealOffset = 40;

bool WriteRipRelativeDisplacement(unsigned char* code, SIZE_T displacement_at,
                                  SIZE_T next_instruction,
                                  SIZE_T target_offset) {
  const std::int64_t displacement =
      static_cast<std::int64_t>(target_offset) -
      static_cast<std::int64_t>(next_instruction);
  if (displacement < std::numeric_limits<std::int32_t>::min() ||
      displacement > std::numeric_limits<std::int32_t>::max()) {
    return false;
  }
  const auto encoded = static_cast<std::int32_t>(displacement);
  std::memcpy(code + displacement_at, &encoded, sizeof(encoded));
  return true;
}

void* ReadOpaqueThunkReal(void* base, SIZE_T page_size) {
  void* real = nullptr;
  const auto* bytes = static_cast<const unsigned char*>(base);
  std::memcpy(&real, bytes + page_size + kOpaqueRealOffset, sizeof(real));
  return real;
}

void* GetOrCreateOpaqueAd298d3fThunk(void* real, bool* mismatch) {
  *mismatch = false;
  void* existing = InterlockedCompareExchangePointer(
      &g_opaque_ad298d3f_thunk, nullptr, nullptr);
  if (existing != nullptr) {
    *mismatch = ReadOpaqueThunkReal(existing, g_opaque_ad298d3f_page_size) !=
                real;
    return *mismatch ? nullptr : existing;
  }

  SYSTEM_INFO system_info{};
  GetSystemInfo(&system_info);
  const SIZE_T page_size = system_info.dwPageSize;
  if (page_size < kOpaqueThunkCodeBytes ||
      page_size < kOpaqueRealOffset + sizeof(void*)) {
    return nullptr;
  }
  const SIZE_T allocation_bytes = page_size * 2;
  auto* candidate = static_cast<unsigned char*>(VirtualAlloc(
      nullptr, allocation_bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  if (candidate == nullptr) return nullptr;

  // lock inc qword ptr [rip+disp32]
  const unsigned char increment[] = {0xf0, 0x48, 0xff, 0x05, 0, 0, 0, 0};
  // mov qword ptr [rip+disp32], rcx/rdx/r8/r9
  const unsigned char save_rcx[] = {0x48, 0x89, 0x0d, 0, 0, 0, 0};
  const unsigned char save_rdx[] = {0x48, 0x89, 0x15, 0, 0, 0, 0};
  const unsigned char save_r8[] = {0x4c, 0x89, 0x05, 0, 0, 0, 0};
  const unsigned char save_r9[] = {0x4c, 0x89, 0x0d, 0, 0, 0, 0};
  // jmp qword ptr [rip+disp32]
  const unsigned char jump_real[] = {0xff, 0x25, 0, 0, 0, 0};
  std::memcpy(candidate + 0, increment, sizeof(increment));
  std::memcpy(candidate + 8, save_rcx, sizeof(save_rcx));
  std::memcpy(candidate + 15, save_rdx, sizeof(save_rdx));
  std::memcpy(candidate + 22, save_r8, sizeof(save_r8));
  std::memcpy(candidate + 29, save_r9, sizeof(save_r9));
  std::memcpy(candidate + 36, jump_real, sizeof(jump_real));
  const bool displacements_ok =
      WriteRipRelativeDisplacement(candidate, 4, 8,
                                   page_size + kOpaqueCountOffset) &&
      WriteRipRelativeDisplacement(candidate, 11, 15,
                                   page_size + kOpaqueRcxOffset) &&
      WriteRipRelativeDisplacement(candidate, 18, 22,
                                   page_size + kOpaqueRdxOffset) &&
      WriteRipRelativeDisplacement(candidate, 25, 29,
                                   page_size + kOpaqueR8Offset) &&
      WriteRipRelativeDisplacement(candidate, 32, 36,
                                   page_size + kOpaqueR9Offset) &&
      WriteRipRelativeDisplacement(candidate, 38, 42,
                                   page_size + kOpaqueRealOffset);
  if (!displacements_ok) {
    VirtualFree(candidate, 0, MEM_RELEASE);
    return nullptr;
  }
  std::memcpy(candidate + page_size + kOpaqueRealOffset, &real, sizeof(real));

  DWORD previous_protection = 0;
  if (!VirtualProtect(candidate, page_size, PAGE_EXECUTE_READ,
                      &previous_protection) ||
      !FlushInstructionCache(GetCurrentProcess(), candidate,
                             kOpaqueThunkCodeBytes)) {
    VirtualFree(candidate, 0, MEM_RELEASE);
    return nullptr;
  }

  g_opaque_ad298d3f_page_size = page_size;
  void* winner = InterlockedCompareExchangePointer(
      &g_opaque_ad298d3f_thunk, candidate, nullptr);
  if (winner != nullptr) {
    VirtualFree(candidate, 0, MEM_RELEASE);
    *mismatch = ReadOpaqueThunkReal(winner, g_opaque_ad298d3f_page_size) !=
                real;
    return *mismatch ? nullptr : winner;
  }
  return candidate;
}

bool FlushOpaqueAd298d3fObservation() {
  void* base = InterlockedCompareExchangePointer(
      &g_opaque_ad298d3f_thunk, nullptr, nullptr);
  if (base == nullptr) return true;
  auto* state = static_cast<unsigned char*>(base) +
                g_opaque_ad298d3f_page_size;
  auto* count = reinterpret_cast<volatile LONG64*>(
      state + kOpaqueCountOffset);
  const LONG64 total = InterlockedCompareExchange64(count, 0, 0);
  const LONG64 previous = InterlockedExchange64(
      &g_opaque_ad298d3f_flushed_count, total);
  if (total < previous) return false;
  const std::uint64_t delta = static_cast<std::uint64_t>(total - previous);

  char phase[129]{};
  GetCurrentPhase(phase);
  bool volatile_written = true;
  if (delta != 0) {
    std::uint64_t rcx = 0;
    std::uint64_t rdx = 0;
    std::uint64_t r8 = 0;
    std::uint64_t r9 = 0;
    std::memcpy(&rcx, state + kOpaqueRcxOffset, sizeof(rcx));
    std::memcpy(&rdx, state + kOpaqueRdxOffset, sizeof(rdx));
    std::memcpy(&r8, state + kOpaqueR8Offset, sizeof(r8));
    std::memcpy(&r9, state + kOpaqueR9Offset, sizeof(r9));
    std::string raw = "{\"event\":\"opaque_interface_call_raw\"";
    AppendString(raw, "interface_id", "0xAD298D3F");
    AppendString(raw, "phase", phase);
    AppendUnsigned(raw, "invocations_since_last", delta);
    AppendUnsigned(raw, "total_invocations",
                   static_cast<std::uint64_t>(total));
    AppendHex(raw, "rcx", rcx, 16);
    AppendHex(raw, "rdx", rdx, 16);
    AppendHex(raw, "r8", r8, 16);
    AppendHex(raw, "r9", r9, 16);
    FinishJson(raw);
    volatile_written = WriteVolatileTrace(raw);
  }

  std::string event = "{\"event\":\"opaque_interface_call_summary\"";
  AppendString(event, "interface_id", "0xAD298D3F");
  AppendString(event, "phase", phase);
  AppendUnsigned(event, "invocations_since_last", delta);
  AppendUnsigned(event, "total_invocations",
                 static_cast<std::uint64_t>(total));
  AppendBoolean(event, "signature_inferred", false);
  AppendBoolean(event, "tail_jump_preserved_arguments", true);
  AppendBoolean(event, "raw_arguments_in_volatile_trace", delta != 0);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  FinishJson(event);
  return volatile_written && WriteTrace(event);
}

bool CreateNewTraceFile(const std::wstring& path) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE,
                            FILE_SHARE_READ, nullptr, CREATE_NEW,
                            FILE_ATTRIBUTE_NORMAL, nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  const bool flushed = FlushFileBuffers(file) != FALSE;
  CloseHandle(file);
  return flushed;
}

BOOL CALLBACK InitializeRuntime(PINIT_ONCE, PVOID, PVOID*) {
  const std::wstring real_path =
      GetEnvironmentString(L"NEURALSTUDIO_FG_REAL_NVAPI64");
  const std::wstring expected_hash =
      GetEnvironmentString(L"NEURALSTUDIO_FG_REAL_NVAPI64_SHA256");
  g_trace_path = GetEnvironmentString(L"NEURALSTUDIO_FG_NVAPI_TRACE");
  g_volatile_trace_path =
      GetEnvironmentString(L"NEURALSTUDIO_FG_NVAPI_VOLATILE_TRACE");
  g_dump_directory =
      GetEnvironmentString(L"NEURALSTUDIO_FG_NVAPI_DUMP_DIR");

  if (!IsAbsoluteWindowsPath(real_path) || !IsAbsoluteWindowsPath(g_trace_path) ||
      !IsAbsoluteWindowsPath(g_volatile_trace_path) ||
      !IsAbsoluteWindowsPath(g_dump_directory)) {
    DebugReject("required absolute environment paths are missing");
    return TRUE;
  }
  if (!IsFile(real_path) || !IsDirectory(ParentPath(g_trace_path)) ||
      !IsDirectory(ParentPath(g_volatile_trace_path)) ||
      !IsDirectory(g_dump_directory)) {
    DebugReject(
        "real NvAPI, trace parent, volatile-trace parent, or dump directory "
        "is missing");
    return TRUE;
  }
  if (_wcsicmp(g_trace_path.c_str(), g_volatile_trace_path.c_str()) == 0) {
    DebugReject("main and volatile trace paths must be distinct");
    return TRUE;
  }
  if (_wcsicmp(BaseName(real_path).c_str(), L"nvapi64.dll") == 0) {
    DebugReject("the real NvAPI copy was not renamed");
    return TRUE;
  }

  wchar_t self_path[32768]{};
  const DWORD self_length = GetModuleFileNameW(
      g_self, self_path, static_cast<DWORD>(std::size(self_path)));
  if (self_length == 0 ||
      self_length >= static_cast<DWORD>(std::size(self_path)) ||
      _wcsicmp(self_path, real_path.c_str()) == 0) {
    DebugReject("real NvAPI path resolves to the proxy");
    return TRUE;
  }
  if (!HashFile(real_path, g_real_nvapi_sha256) ||
      !ExpectedHashMatches(expected_hash, g_real_nvapi_sha256)) {
    DebugReject("real NvAPI SHA-256 verification failed");
    return TRUE;
  }
  if (!CreateNewTraceFile(g_trace_path) ||
      !CreateNewTraceFile(g_volatile_trace_path)) {
    DebugReject("main or volatile trace could not be created exclusively");
    return TRUE;
  }

  g_real_nvapi = LoadLibraryExW(
      real_path.c_str(), nullptr,
      LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (g_real_nvapi == nullptr || g_real_nvapi == g_self) {
    DebugReject("real NvAPI could not be loaded");
    return TRUE;
  }
  g_real_query_interface = PointerCast<Dispatcher>(
      GetProcAddress(g_real_nvapi, "nvapi_QueryInterface"));
  g_real_direct_get_method = PointerCast<Dispatcher>(
      GetProcAddress(g_real_nvapi, "nvapi_Direct_GetMethod"));
  if (g_real_query_interface == nullptr || g_real_direct_get_method == nullptr) {
    DebugReject("real NvAPI dispatch exports are missing");
    return TRUE;
  }

  g_runtime_ready = true;
  std::string event =
      "{\"event\":\"runtime_ready\",\"schema\":"
      "\"neuralstudio.fg.nvapi-forward-trace.v1\"";
  AppendString(event, "real_nvapi_sha256", g_real_nvapi_sha256);
  FinishJson(event);
  if (!WriteTrace(event)) {
    g_runtime_ready = false;
    DebugReject("runtime-ready trace could not be written");
  }
  return TRUE;
}

bool RuntimeReady() {
  InitOnceExecuteOnce(&g_runtime_once, InitializeRuntime, nullptr, nullptr);
  return g_runtime_ready;
}

bool WriteDump(const char* file_name, const void* data, std::uint64_t bytes,
               DumpInfo* info) {
  if (info == nullptr) return false;
  *info = {};
  if (bytes == 0) {
    info->captured = true;
    return true;
  }
  if (data == nullptr || file_name == nullptr || file_name[0] == '\0' ||
      std::strlen(file_name) >= sizeof(info->file)) {
    return false;
  }
  if (!HashBuffer(data, static_cast<std::size_t>(bytes), info->sha256)) {
    return false;
  }

  wchar_t wide_name[160]{};
  if (MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, file_name, -1,
                          wide_name, static_cast<int>(std::size(wide_name))) ==
      0) {
    return false;
  }
  std::wstring path = g_dump_directory;
  path += L"\\";
  path += wide_name;
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;

  const unsigned char* cursor = static_cast<const unsigned char*>(data);
  std::uint64_t remaining = bytes;
  bool success = true;
  while (remaining > 0) {
    const DWORD chunk = remaining > MAXDWORD
                            ? MAXDWORD
                            : static_cast<DWORD>(remaining);
    DWORD written = 0;
    if (!WriteFile(file, cursor, chunk, &written, nullptr) ||
        written != chunk) {
      success = false;
      break;
    }
    cursor += written;
    remaining -= written;
  }
  if (success) success = FlushFileBuffers(file) != FALSE;
  CloseHandle(file);
  if (!success) return false;

  info->captured = true;
  info->bytes = bytes;
  strcpy_s(info->file, file_name);
  return true;
}

void AppendDump(std::string& event, const char* prefix, const DumpInfo& dump) {
  std::string captured_key = std::string(prefix) + "_captured";
  std::string bytes_key = std::string(prefix) + "_bytes";
  std::string file_key = std::string(prefix) + "_file";
  std::string hash_key = std::string(prefix) + "_sha256";
  AppendBoolean(event, captured_key.c_str(), dump.captured);
  AppendUnsigned(event, bytes_key.c_str(), dump.bytes);
  if (dump.file[0] != '\0') AppendString(event, file_key.c_str(), dump.file);
  if (dump.sha256[0] != '\0') {
    AppendString(event, hash_key.c_str(), dump.sha256);
  }
}

NvStatus CaptureFailure(const char* call, const char* reason) {
  std::string event = "{\"event\":\"capture_failure\"";
  AppendString(event, "call", call);
  AppendString(event, "reason", reason);
  FinishJson(event);
  WriteTrace(event);
  return kNvapiError;
}

bool AddModule(NvObjectHandle handle, std::uint64_t token) {
  if (handle == nullptr) return false;
  bool added = false;
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_modules) {
    if (!record.active) {
      record.handle = handle;
      record.token = token;
      record.active = true;
      added = true;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
  return added;
}

bool AddFunction(NvObjectHandle handle, std::uint64_t token,
                 std::uint64_t module_token, const char* name) {
  if (handle == nullptr) return false;
  bool added = false;
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_functions) {
    if (!record.active) {
      record.handle = handle;
      record.token = token;
      record.module_token = module_token;
      strncpy_s(record.name, name != nullptr ? name : "", _TRUNCATE);
      record.active = true;
      added = true;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
  return added;
}

bool AddShader(NvObjectHandle handle, std::uint64_t token, Dim3 block,
               std::uint32_t dynamic_shared_bytes, const char* name) {
  if (handle == nullptr) return false;
  bool added = false;
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_shaders) {
    if (!record.active) {
      record.handle = handle;
      record.token = token;
      record.block = block;
      record.dynamic_shared_bytes = dynamic_shared_bytes;
      strncpy_s(record.name, name != nullptr ? name : "", _TRUNCATE);
      record.active = true;
      added = true;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
  return added;
}

std::uint64_t AddResource(ResourceKind kind, std::uint64_t handle) {
  if (handle == 0) return 0;
  std::uint64_t token = 0;
  AcquireSRWLockExclusive(&g_state_lock);
  for (const auto& record : g_resources) {
    if (record.active && record.kind == kind && record.handle == handle) {
      token = record.token;
      break;
    }
  }
  if (token == 0) {
    for (auto& record : g_resources) {
      if (!record.active) {
        token = NextToken(&g_next_resource_token);
        record.handle = handle;
        record.token = token;
        record.kind = kind;
        record.active = true;
        break;
      }
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
  return token;
}

ObjectInfo FindModuleInfo(NvObjectHandle handle) {
  ObjectInfo info{};
  AcquireSRWLockShared(&g_state_lock);
  for (const auto& record : g_modules) {
    if (record.active && record.handle == handle) {
      info.tracked = true;
      info.token = record.token;
      break;
    }
  }
  ReleaseSRWLockShared(&g_state_lock);
  return info;
}

ObjectInfo FindFunctionInfo(NvObjectHandle handle) {
  ObjectInfo info{};
  AcquireSRWLockShared(&g_state_lock);
  for (const auto& record : g_functions) {
    if (record.active && record.handle == handle) {
      info.tracked = true;
      info.token = record.token;
      info.parent_token = record.module_token;
      strcpy_s(info.name, record.name);
      break;
    }
  }
  ReleaseSRWLockShared(&g_state_lock);
  return info;
}

ObjectInfo FindShaderInfo(NvObjectHandle handle) {
  ObjectInfo info{};
  AcquireSRWLockShared(&g_state_lock);
  for (const auto& record : g_shaders) {
    if (record.active && record.handle == handle) {
      info.tracked = true;
      info.token = record.token;
      info.block = record.block;
      info.dynamic_shared_bytes = record.dynamic_shared_bytes;
      strcpy_s(info.name, record.name);
      break;
    }
  }
  ReleaseSRWLockShared(&g_state_lock);
  return info;
}

void RemoveModule(NvObjectHandle handle) {
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_modules) {
    if (record.active && record.handle == handle) {
      record.active = false;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
}

void RemoveFunction(NvObjectHandle handle) {
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_functions) {
    if (record.active && record.handle == handle) {
      record.active = false;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
}

void RemoveShader(NvObjectHandle handle) {
  AcquireSRWLockExclusive(&g_state_lock);
  for (auto& record : g_shaders) {
    if (record.active && record.handle == handle) {
      record.active = false;
      break;
    }
  }
  ReleaseSRWLockExclusive(&g_state_lock);
}

void AppendToken(std::string& event, const char* key, const char* prefix,
                 std::uint64_t token, bool tracked) {
  char value[64]{};
  if (tracked && token != 0) {
    sprintf_s(value, "%s-%06llu", prefix,
              static_cast<unsigned long long>(token));
  } else {
    strcpy_s(value, "untracked");
  }
  AppendString(event, key, value);
}

template <typename Function>
Function RealFunction(std::uint32_t id) {
  InterfaceBinding* binding = FindBinding(id);
  if (binding == nullptr) return nullptr;
  return PointerCast<Function>(binding->real);
}

void* ResolveInterface(const char* dispatcher_name, bool direct,
                       std::uint32_t id) {
  if (!RuntimeReady()) return nullptr;
  const Dispatcher dispatcher =
      direct ? g_real_direct_get_method : g_real_query_interface;
  if (dispatcher == nullptr) return nullptr;
  void* real = dispatcher(id);
  InterfaceBinding* binding = FindBinding(id);
  const char* passthrough_name =
      binding == nullptr ? FindKnownPassthroughName(id) : nullptr;
  bool wrapped = false;
  bool mismatch = false;
  bool opaque_sentinel = false;
  void* returned = real;
  if (binding != nullptr && real != nullptr) {
    void* previous = InterlockedCompareExchangePointer(&binding->real, real,
                                                       nullptr);
    if (previous != nullptr && previous != real) {
      mismatch = true;
    } else {
      wrapped = true;
      returned = binding->wrapper;
    }
  } else if (binding == nullptr && id == kIdOpaqueAd298d3f && real != nullptr) {
    returned = GetOrCreateOpaqueAd298d3fThunk(real, &mismatch);
    opaque_sentinel = returned != nullptr && !mismatch;
  }

  char interface_id[16]{};
  sprintf_s(interface_id, "0x%08X", id);
  std::string event = "{\"event\":\"interface_query\"";
  AppendString(event, "dispatcher", dispatcher_name);
  AppendString(event, "interface_id", interface_id);
  if (binding != nullptr) {
    AppendString(event, "interface", binding->name);
    AppendString(event, "classification", "wrapped");
  } else if (passthrough_name != nullptr) {
    AppendString(event, "interface", passthrough_name);
    AppendString(event, "classification", "known_passthrough");
  } else if (opaque_sentinel) {
    AppendString(event, "interface", "opaque_0xAD298D3F");
    AppendString(event, "classification", "opaque_call_sentinel");
  } else {
    AppendString(event, "classification", "unknown");
  }
  AppendBoolean(event, "real_resolved", real != nullptr);
  AppendBoolean(event, "returned_wrapper", wrapped || opaque_sentinel);
  AppendBoolean(event, "opaque_call_sentinel", opaque_sentinel);
  AppendBoolean(event, "real_dispatch_mismatch", mismatch);
  FinishJson(event);
  if (!WriteTrace(event) || mismatch || returned == nullptr) return nullptr;
  return returned;
}

std::string BeginCall(const char* call) {
  std::string event = "{\"event\":\"nvapi_call\"";
  AppendString(event, "call", call);
  return event;
}

bool FinishCall(std::string& event, NvStatus status) {
  AppendSigned(event, "status", status);
  FinishJson(event);
  return WriteTrace(event);
}

DumpInfo CaptureNamedBytes(const char* prefix, std::uint64_t token,
                           const char* suffix, const void* data,
                           std::uint32_t size) {
  DumpInfo dump{};
  char file[160]{};
  sprintf_s(file, "%s-%06llu%s", prefix,
            static_cast<unsigned long long>(token), suffix);
  if (!WriteDump(file, data, size, &dump)) dump = {};
  return dump;
}

NvStatus __cdecl TraceCreateCubinExV2(CubinShaderParams* params) {
  using Function = NvStatus(__cdecl*)(CubinShaderParams*);
  const Function real = RealFunction<Function>(kIdCreateCubinExV2);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t token = NextToken(&g_next_shader_token);
  const bool readable = params != nullptr &&
                        params->struct_size_in >= sizeof(CubinShaderParams);
  DumpInfo dump{};
  std::string name;
  Dim3 block{};
  std::uint32_t shared = 0;
  if (readable) {
    name = params->shader_name != nullptr ? params->shader_name : "";
    block = {params->block_x, params->block_y, params->block_z};
    shared = params->dynamic_shared_bytes;
    dump = CaptureNamedBytes("legacy-cubin", token, ".bin", params->cubin,
                             params->size);
    if (!dump.captured) {
      return CaptureFailure("NvAPI_D3D12_CreateCubinComputeShaderExV2",
                            "cubin_dump_failed");
    }
  }
  const NvStatus status = real(params);
  bool tracked = false;
  if (status == kNvapiOk && readable && params->shader != nullptr) {
    tracked = AddShader(params->shader, token, block, shared, name.c_str());
  }
  std::string event =
      BeginCall("NvAPI_D3D12_CreateCubinComputeShaderExV2");
  AppendToken(event, "shader_token", "shader", token, tracked);
  AppendString(event, "shader_name", name.c_str());
  AppendDim3(event, "block", block);
  AppendUnsigned(event, "dynamic_shared_bytes", shared);
  AppendDump(event, "module_blob", dump);
  AppendBoolean(event, "input_struct_readable", readable);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceCreateCubin(void* device, const void* cubin,
                                  std::uint32_t size, std::uint32_t block_x,
                                  std::uint32_t block_y,
                                  std::uint32_t block_z,
                                  NvObjectHandle* shader) {
  using Function = NvStatus(__cdecl*)(void*, const void*, std::uint32_t,
                                      std::uint32_t, std::uint32_t,
                                      std::uint32_t, NvObjectHandle*);
  const Function real = RealFunction<Function>(kIdCreateCubin);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t token = NextToken(&g_next_shader_token);
  const DumpInfo dump =
      CaptureNamedBytes("legacy-cubin", token, ".bin", cubin, size);
  if (!dump.captured) {
    return CaptureFailure("NvAPI_D3D12_CreateCubinComputeShader",
                          "cubin_dump_failed");
  }
  const NvStatus status = real(device, cubin, size, block_x, block_y, block_z,
                               shader);
  const Dim3 block{block_x, block_y, block_z};
  const bool tracked = status == kNvapiOk && shader != nullptr &&
                       AddShader(*shader, token, block, 0, "");
  std::string event = BeginCall("NvAPI_D3D12_CreateCubinComputeShader");
  AppendToken(event, "shader_token", "shader", token, tracked);
  AppendDim3(event, "block", block);
  AppendUnsigned(event, "dynamic_shared_bytes", 0);
  AppendDump(event, "module_blob", dump);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceCreateCubinEx(
    void* device, const void* cubin, std::uint32_t size,
    std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
    std::uint32_t dynamic_shared_bytes, const char* shader_name,
    NvObjectHandle* shader) {
  using Function = NvStatus(__cdecl*)(
      void*, const void*, std::uint32_t, std::uint32_t, std::uint32_t,
      std::uint32_t, std::uint32_t, const char*, NvObjectHandle*);
  const Function real = RealFunction<Function>(kIdCreateCubinEx);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t token = NextToken(&g_next_shader_token);
  const DumpInfo dump =
      CaptureNamedBytes("legacy-cubin", token, ".bin", cubin, size);
  if (!dump.captured) {
    return CaptureFailure("NvAPI_D3D12_CreateCubinComputeShaderEx",
                          "cubin_dump_failed");
  }
  const std::string name = shader_name != nullptr ? shader_name : "";
  const NvStatus status =
      real(device, cubin, size, block_x, block_y, block_z,
           dynamic_shared_bytes, shader_name, shader);
  const Dim3 block{block_x, block_y, block_z};
  const bool tracked = status == kNvapiOk && shader != nullptr &&
                       AddShader(*shader, token, block, dynamic_shared_bytes,
                                 name.c_str());
  std::string event = BeginCall("NvAPI_D3D12_CreateCubinComputeShaderEx");
  AppendToken(event, "shader_token", "shader", token, tracked);
  AppendString(event, "shader_name", name.c_str());
  AppendDim3(event, "block", block);
  AppendUnsigned(event, "dynamic_shared_bytes", dynamic_shared_bytes);
  AppendDump(event, "module_blob", dump);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceCreateCubinWithName(
    void* device, const void* cubin, std::uint32_t size,
    std::uint32_t block_x, std::uint32_t block_y, std::uint32_t block_z,
    const char* shader_name, NvObjectHandle* shader) {
  using Function = NvStatus(__cdecl*)(void*, const void*, std::uint32_t,
                                      std::uint32_t, std::uint32_t,
                                      std::uint32_t, const char*,
                                      NvObjectHandle*);
  const Function real = RealFunction<Function>(kIdCreateCubinWithName);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t token = NextToken(&g_next_shader_token);
  const DumpInfo dump =
      CaptureNamedBytes("legacy-cubin", token, ".bin", cubin, size);
  if (!dump.captured) {
    return CaptureFailure("NvAPI_D3D12_CreateCubinComputeShaderWithName",
                          "cubin_dump_failed");
  }
  const std::string name = shader_name != nullptr ? shader_name : "";
  const NvStatus status = real(device, cubin, size, block_x, block_y, block_z,
                               shader_name, shader);
  const Dim3 block{block_x, block_y, block_z};
  const bool tracked = status == kNvapiOk && shader != nullptr &&
                       AddShader(*shader, token, block, 0, name.c_str());
  std::string event =
      BeginCall("NvAPI_D3D12_CreateCubinComputeShaderWithName");
  AppendToken(event, "shader_token", "shader", token, tracked);
  AppendString(event, "shader_name", name.c_str());
  AppendDim3(event, "block", block);
  AppendUnsigned(event, "dynamic_shared_bytes", 0);
  AppendDump(event, "module_blob", dump);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceLaunchCubin(void* command_list, NvObjectHandle shader,
                                  std::uint32_t grid_x,
                                  std::uint32_t grid_y,
                                  std::uint32_t grid_z, const void* params,
                                  std::uint32_t param_size) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle, std::uint32_t,
                                      std::uint32_t, std::uint32_t,
                                      const void*, std::uint32_t);
  const Function real = RealFunction<Function>(kIdLaunchCubin);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  if (param_size != 0 && params == nullptr) {
    return CaptureFailure("NvAPI_D3D12_LaunchCubinShader",
                          "parameter_pointer_missing");
  }
  const std::uint64_t launch_token = NextToken(&g_next_launch_token);
  char file[160]{};
  sprintf_s(file, "launch-%06llu-kernel-0000.params.bin",
            static_cast<unsigned long long>(launch_token));
  DumpInfo dump{};
  if (!WriteDump(file, params, param_size, &dump)) {
    return CaptureFailure("NvAPI_D3D12_LaunchCubinShader",
                          "parameter_dump_failed");
  }
  const ObjectInfo shader_info = FindShaderInfo(shader);
  const NvStatus status = real(command_list, shader, grid_x, grid_y, grid_z,
                               params, param_size);
  std::string event = BeginCall("NvAPI_D3D12_LaunchCubinShader");
  AppendToken(event, "launch_token", "launch", launch_token, true);
  AppendToken(event, "shader_token", "shader", shader_info.token,
              shader_info.tracked);
  AppendString(event, "shader_name", shader_info.name);
  AppendDim3(event, "grid", Dim3{grid_x, grid_y, grid_z});
  AppendDim3(event, "block", shader_info.block);
  AppendUnsigned(event, "dynamic_shared_bytes",
                 shader_info.dynamic_shared_bytes);
  AppendDump(event, "parameter", dump);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceDestroyCubin(void* device, NvObjectHandle shader) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle);
  const Function real = RealFunction<Function>(kIdDestroyCubin);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const ObjectInfo info = FindShaderInfo(shader);
  const NvStatus status = real(device, shader);
  if (status == kNvapiOk) RemoveShader(shader);
  std::string event = BeginCall("NvAPI_D3D12_DestroyCubinComputeShader");
  AppendToken(event, "shader_token", "shader", info.token, info.tracked);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

const char* ResourceKindName(ResourceKind kind) {
  switch (kind) {
    case ResourceKind::kMergedTextureSampler: return "merged_texture_sampler";
    case ResourceKind::kIndependentSurface: return "independent_surface";
    case ResourceKind::kIndependentTexture: return "independent_texture";
    case ResourceKind::kIndependentSampler: return "independent_sampler";
    case ResourceKind::kTexture: return "texture";
    case ResourceKind::kSurface: return "surface";
  }
  return "unknown";
}

ResourceKind IndependentKind(IndependentDescriptorType type) {
  switch (type) {
    case IndependentDescriptorType::kSurface:
      return ResourceKind::kIndependentSurface;
    case IndependentDescriptorType::kTexture:
      return ResourceKind::kIndependentTexture;
    case IndependentDescriptorType::kSampler:
      return ResourceKind::kIndependentSampler;
  }
  return ResourceKind::kIndependentTexture;
}

bool WriteRawResourceObject(
    const char* call, const char* phase, ResourceKind kind,
    std::uint64_t token, NvStatus status, std::uint64_t object_handle,
    unsigned int object_handle_bytes, const char* output_field,
    const char* descriptor_a_name, std::uintptr_t descriptor_a,
    const char* descriptor_b_name = nullptr,
    std::uintptr_t descriptor_b = 0) {
  std::string volatile_event = "{\"event\":\"resource_object_raw\"";
  AppendString(volatile_event, "call", call);
  AppendString(volatile_event, "phase", phase);
  AppendString(volatile_event, "resource_kind", ResourceKindName(kind));
  AppendToken(volatile_event, "resource_token", "resource", token,
              token != 0);
  AppendSigned(volatile_event, "status", status);
  AppendString(volatile_event, "output_field", output_field);
  AppendString(volatile_event, "byte_order", "little");
  AppendUnsigned(volatile_event, "cuda_object_handle_bytes",
                 object_handle_bytes);
  AppendHex(volatile_event, "cuda_object_handle", object_handle,
            object_handle_bytes == 4 ? 8 : 16);
  AppendHex(volatile_event, descriptor_a_name,
            static_cast<std::uint64_t>(descriptor_a), 16);
  if (descriptor_b_name != nullptr) {
    AppendHex(volatile_event, descriptor_b_name,
              static_cast<std::uint64_t>(descriptor_b), 16);
  }
  FinishJson(volatile_event);
  return WriteVolatileTrace(volatile_event);
}

NvStatus __cdecl TraceGetMergedTextureSampler(
    MergedTextureSamplerParams* params) {
  using Function = NvStatus(__cdecl*)(MergedTextureSamplerParams*);
  const Function real = RealFunction<Function>(kIdGetMergedTextureSampler);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  char phase[129]{};
  GetCurrentPhase(phase);
  const std::uintptr_t texture_descriptor =
      params != nullptr ? params->texture_descriptor.ptr : 0;
  const std::uintptr_t sampler_descriptor =
      params != nullptr ? params->sampler_descriptor.ptr : 0;
  // The real driver rewrites scratch descriptor slots; bind the immutable
  // pre-call snapshot rather than whatever remains after object creation.
  const auto descriptor_snapshot =
      neuralstudio::fg_trace::DescriptorTraceSnapshotResourceObject(
          ResourceKindName(ResourceKind::kMergedTextureSampler),
          texture_descriptor, sampler_descriptor, true);
  const NvStatus status = real(params);
  const std::uint64_t object_handle =
      status == kNvapiOk && params != nullptr ? params->texture_handle : 0;
  const std::uint64_t token =
      status == kNvapiOk && params != nullptr
          ? AddResource(ResourceKind::kMergedTextureSampler,
                        object_handle)
          : 0;
  const bool volatile_written = WriteRawResourceObject(
      "NvAPI_D3D12_GetCudaMergedTextureSamplerObject", phase,
      ResourceKind::kMergedTextureSampler, token, status, object_handle, 8,
      "textureHandle", "texture_descriptor_ptr", texture_descriptor,
      "sampler_descriptor_ptr", sampler_descriptor);
  const bool descriptor_written =
      neuralstudio::fg_trace::DescriptorTraceRecordResourceObject(
          "NvAPI_D3D12_GetCudaMergedTextureSamplerObject", phase,
          ResourceKindName(ResourceKind::kMergedTextureSampler), token,
          status, texture_descriptor, sampler_descriptor, true,
          descriptor_snapshot);
  std::string event =
      BeginCall("NvAPI_D3D12_GetCudaMergedTextureSamplerObject");
  AppendString(event, "phase", phase);
  AppendString(event, "resource_kind",
               ResourceKindName(ResourceKind::kMergedTextureSampler));
  AppendToken(event, "resource_token", "resource", token, token != 0);
  AppendBoolean(event, "descriptor_values_recorded", false);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  const bool main_written = FinishCall(event, status);
  return volatile_written && descriptor_written && main_written ? status
                                                                 : kNvapiError;
}

NvStatus __cdecl TraceGetIndependentDescriptor(
    IndependentDescriptorParams* params) {
  using Function = NvStatus(__cdecl*)(IndependentDescriptorParams*);
  const Function real = RealFunction<Function>(kIdGetIndependentDescriptor);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  char phase[129]{};
  GetCurrentPhase(phase);
  const IndependentDescriptorType input_type =
      params != nullptr ? params->type : IndependentDescriptorType::kTexture;
  const std::uintptr_t descriptor =
      params != nullptr ? params->descriptor.ptr : 0;
  const ResourceKind kind = IndependentKind(input_type);
  const auto descriptor_snapshot =
      neuralstudio::fg_trace::DescriptorTraceSnapshotResourceObject(
          ResourceKindName(kind), descriptor, 0, false);
  const NvStatus status = real(params);
  const std::uint64_t object_handle =
      status == kNvapiOk && params != nullptr ? params->handle : 0;
  const std::uint64_t token =
      status == kNvapiOk && params != nullptr
          ? AddResource(kind, object_handle)
          : 0;
  const bool volatile_written = WriteRawResourceObject(
      "NvAPI_D3D12_GetCudaIndependentDescriptorObject", phase, kind, token,
      status, object_handle, 8, "handle", "descriptor_ptr", descriptor);
  const bool descriptor_written =
      neuralstudio::fg_trace::DescriptorTraceRecordResourceObject(
          "NvAPI_D3D12_GetCudaIndependentDescriptorObject", phase,
          ResourceKindName(kind), token, status, descriptor, 0, false,
          descriptor_snapshot);
  std::string event =
      BeginCall("NvAPI_D3D12_GetCudaIndependentDescriptorObject");
  AppendString(event, "phase", phase);
  AppendString(event, "resource_kind", ResourceKindName(kind));
  AppendSigned(event, "descriptor_type",
               static_cast<std::int32_t>(input_type));
  AppendToken(event, "resource_token", "resource", token, token != 0);
  AppendBoolean(event, "descriptor_values_recorded", false);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  const bool main_written = FinishCall(event, status);
  return volatile_written && descriptor_written && main_written ? status
                                                                 : kNvapiError;
}

NvStatus __cdecl TraceGetTextureObject(
    void* device, CpuDescriptorHandle texture_descriptor,
    CpuDescriptorHandle sampler_descriptor, std::uint32_t* texture_handle) {
  using Function = NvStatus(__cdecl*)(void*, CpuDescriptorHandle,
                                      CpuDescriptorHandle, std::uint32_t*);
  const Function real = RealFunction<Function>(kIdGetTextureObject);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  char phase[129]{};
  GetCurrentPhase(phase);
  const auto descriptor_snapshot =
      neuralstudio::fg_trace::DescriptorTraceSnapshotResourceObject(
          ResourceKindName(ResourceKind::kTexture), texture_descriptor.ptr,
          sampler_descriptor.ptr, true);
  const NvStatus status =
      real(device, texture_descriptor, sampler_descriptor, texture_handle);
  const std::uint64_t object_handle =
      status == kNvapiOk && texture_handle != nullptr ? *texture_handle : 0;
  const std::uint64_t token =
      status == kNvapiOk && texture_handle != nullptr
          ? AddResource(ResourceKind::kTexture, object_handle)
          : 0;
  const bool volatile_written = WriteRawResourceObject(
      "NvAPI_D3D12_GetCudaTextureObject", phase, ResourceKind::kTexture,
      token, status, object_handle, 4, "textureHandle",
      "texture_descriptor_ptr", texture_descriptor.ptr,
      "sampler_descriptor_ptr", sampler_descriptor.ptr);
  const bool descriptor_written =
      neuralstudio::fg_trace::DescriptorTraceRecordResourceObject(
          "NvAPI_D3D12_GetCudaTextureObject", phase,
          ResourceKindName(ResourceKind::kTexture), token, status,
          texture_descriptor.ptr, sampler_descriptor.ptr, true,
          descriptor_snapshot);
  std::string event = BeginCall("NvAPI_D3D12_GetCudaTextureObject");
  AppendString(event, "phase", phase);
  AppendString(event, "resource_kind",
               ResourceKindName(ResourceKind::kTexture));
  AppendToken(event, "resource_token", "resource", token, token != 0);
  AppendBoolean(event, "descriptor_values_recorded", false);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  const bool main_written = FinishCall(event, status);
  return volatile_written && descriptor_written && main_written ? status
                                                                 : kNvapiError;
}

NvStatus __cdecl TraceGetSurfaceObject(
    void* device, CpuDescriptorHandle uav_descriptor,
    std::uint32_t* surface_handle) {
  using Function =
      NvStatus(__cdecl*)(void*, CpuDescriptorHandle, std::uint32_t*);
  const Function real = RealFunction<Function>(kIdGetSurfaceObject);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  char phase[129]{};
  GetCurrentPhase(phase);
  const auto descriptor_snapshot =
      neuralstudio::fg_trace::DescriptorTraceSnapshotResourceObject(
          ResourceKindName(ResourceKind::kSurface), uav_descriptor.ptr, 0,
          false);
  const NvStatus status = real(device, uav_descriptor, surface_handle);
  const std::uint64_t object_handle =
      status == kNvapiOk && surface_handle != nullptr ? *surface_handle : 0;
  const std::uint64_t token =
      status == kNvapiOk && surface_handle != nullptr
          ? AddResource(ResourceKind::kSurface, object_handle)
          : 0;
  const bool volatile_written = WriteRawResourceObject(
      "NvAPI_D3D12_GetCudaSurfaceObject", phase, ResourceKind::kSurface,
      token, status, object_handle, 4, "surfaceHandle",
      "uav_descriptor_ptr", uav_descriptor.ptr);
  const bool descriptor_written =
      neuralstudio::fg_trace::DescriptorTraceRecordResourceObject(
          "NvAPI_D3D12_GetCudaSurfaceObject", phase,
          ResourceKindName(ResourceKind::kSurface), token, status,
          uav_descriptor.ptr, 0, false, descriptor_snapshot);
  std::string event = BeginCall("NvAPI_D3D12_GetCudaSurfaceObject");
  AppendString(event, "phase", phase);
  AppendString(event, "resource_kind",
               ResourceKindName(ResourceKind::kSurface));
  AppendToken(event, "resource_token", "resource", token, token != 0);
  AppendBoolean(event, "descriptor_values_recorded", false);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  const bool main_written = FinishCall(event, status);
  return volatile_written && descriptor_written && main_written ? status
                                                                 : kNvapiError;
}

NvStatus __cdecl TraceIsFatbinPtxSupported(void* device, bool* supported) {
  using Function = NvStatus(__cdecl*)(void*, bool*);
  const Function real = RealFunction<Function>(kIdIsFatbinPtxSupported);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const NvStatus status = real(device, supported);
  std::string event = BeginCall("NvAPI_D3D12_IsFatbinPTXSupported");
  AppendBoolean(event, "supported",
                status == kNvapiOk && supported != nullptr && *supported);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceCreateCuModule(void* device, const void* blob,
                                     std::uint32_t size,
                                     NvObjectHandle* module) {
  using Function = NvStatus(__cdecl*)(void*, const void*, std::uint32_t,
                                      NvObjectHandle*);
  const Function real = RealFunction<Function>(kIdCreateCuModule);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t token = NextToken(&g_next_module_token);
  const DumpInfo dump =
      CaptureNamedBytes("cu-module", token, ".bin", blob, size);
  if (!dump.captured) {
    return CaptureFailure("NvAPI_D3D12_CreateCuModule",
                          "module_blob_dump_failed");
  }
  const NvStatus status = real(device, blob, size, module);
  const bool tracked = status == kNvapiOk && module != nullptr &&
                       AddModule(*module, token);
  std::string event = BeginCall("NvAPI_D3D12_CreateCuModule");
  AppendToken(event, "module_token", "module", token, tracked);
  AppendDump(event, "module_blob", dump);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceEnumFunctions(void* device, NvObjectHandle module,
                                    std::uint32_t* array_size,
                                    const char** const function_names) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle, std::uint32_t*,
                                      const char** const);
  const Function real = RealFunction<Function>(kIdEnumFunctions);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint32_t requested = array_size != nullptr ? *array_size : 0;
  const ObjectInfo module_info = FindModuleInfo(module);
  const NvStatus status = real(device, module, array_size, function_names);
  const std::uint32_t returned = array_size != nullptr ? *array_size : 0;
  std::string event = BeginCall("NvAPI_D3D12_EnumFunctionsInModule");
  AppendToken(event, "module_token", "module", module_info.token,
              module_info.tracked);
  AppendUnsigned(event, "requested_count", requested);
  AppendUnsigned(event, "returned_count", returned);
  if (!FinishCall(event, status)) return kNvapiError;

  if (status == kNvapiOk && function_names != nullptr &&
      returned <= kMaxEnumeratedFunctions) {
    for (std::uint32_t index = 0; index < returned; ++index) {
      std::string name_event = "{\"event\":\"module_function_name\"";
      AppendToken(name_event, "module_token", "module", module_info.token,
                  module_info.tracked);
      AppendUnsigned(name_event, "index", index);
      AppendString(name_event, "function_name", function_names[index]);
      FinishJson(name_event);
      if (!WriteTrace(name_event)) return kNvapiError;
    }
  }
  return status;
}

NvStatus __cdecl TraceCreateCuFunction(void* device, NvObjectHandle module,
                                       const char* name,
                                       NvObjectHandle* function) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle, const char*,
                                      NvObjectHandle*);
  const Function real = RealFunction<Function>(kIdCreateCuFunction);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const ObjectInfo module_info = FindModuleInfo(module);
  const std::uint64_t token = NextToken(&g_next_function_token);
  const std::string function_name = name != nullptr ? name : "";
  const NvStatus status = real(device, module, name, function);
  const bool tracked = status == kNvapiOk && function != nullptr &&
                       AddFunction(*function, token, module_info.token,
                                   function_name.c_str());
  std::string event = BeginCall("NvAPI_D3D12_CreateCuFunction");
  AppendToken(event, "module_token", "module", module_info.token,
              module_info.tracked);
  AppendToken(event, "function_token", "function", token, tracked);
  AppendString(event, "function_name", function_name.c_str());
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

template <bool Extended, typename Kernel>
NvStatus CaptureKernelChain(const char* call, const Kernel* kernels,
                            std::uint32_t kernel_count,
                            std::uint64_t launch_token) {
  if (kernel_count > kMaxKernelChainLength) {
    return CaptureFailure(call, "kernel_count_exceeds_capture_limit");
  }
  if (kernel_count != 0 && kernels == nullptr) {
    return CaptureFailure(call, "kernel_array_missing");
  }
  for (std::uint32_t index = 0; index < kernel_count; ++index) {
    const Kernel& kernel = kernels[index];
    const ObjectInfo function_info = FindFunctionInfo(kernel.function);
    const bool packed_params = kernel.params != nullptr || kernel.param_size != 0;
    if (kernel.param_size != 0 && kernel.params == nullptr) {
      return CaptureFailure(call, "packed_parameter_pointer_missing");
    }
    char file[160]{};
    sprintf_s(file, "launch-%06llu-kernel-%04u.params.bin",
              static_cast<unsigned long long>(launch_token), index);
    DumpInfo dump{};
    if (!WriteDump(file, kernel.params, kernel.param_size, &dump)) {
      return CaptureFailure(call, "parameter_dump_failed");
    }

    std::string event = "{\"event\":\"kernel_launch\"";
    AppendToken(event, "launch_token", "launch", launch_token, true);
    AppendUnsigned(event, "kernel_index", index);
    AppendToken(event, "function_token", "function", function_info.token,
                function_info.tracked);
    AppendToken(event, "module_token", "module", function_info.parent_token,
                function_info.tracked && function_info.parent_token != 0);
    AppendString(event, "function_name", function_info.name);
    AppendDim3(event, "grid", kernel.grid);
    AppendDim3(event, "block", kernel.block);
    AppendUnsigned(event, "dynamic_shared_bytes",
                   kernel.dynamic_shared_bytes);
    const char* parameter_mode = packed_params ? "packed" : "none";
    if constexpr (Extended) {
      if (!packed_params && kernel.kernel_params != nullptr) {
        // kernelParams is an unbounded array of pointers in this private ABI.
        // Capturing it without cubin argument metadata would read arbitrary
        // memory and leak raw addresses into the normalized receipt.
        parameter_mode = "pointer_array_unbounded_not_captured";
      }
    }
    AppendString(event, "parameter_mode", parameter_mode);
    AppendDump(event, "parameter", dump);
    FinishJson(event);
    if (!WriteTrace(event)) return kNvapiError;
  }
  return kNvapiOk;
}

NvStatus __cdecl TraceLaunchKernelChain(void* command_list,
                                        const KernelLaunchParams* kernels,
                                        std::uint32_t kernel_count) {
  using Function = NvStatus(__cdecl*)(void*, const KernelLaunchParams*,
                                      std::uint32_t);
  const Function real = RealFunction<Function>(kIdLaunchKernelChain);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t launch_token = NextToken(&g_next_launch_token);
  const NvStatus capture = CaptureKernelChain<false>(
      "NvAPI_D3D12_LaunchCuKernelChain", kernels, kernel_count, launch_token);
  if (capture != kNvapiOk) return capture;
  const NvStatus status = real(command_list, kernels, kernel_count);
  std::string event = BeginCall("NvAPI_D3D12_LaunchCuKernelChain");
  AppendToken(event, "launch_token", "launch", launch_token, true);
  AppendUnsigned(event, "kernel_count", kernel_count);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceLaunchKernelChainEx(void* command_list,
                                          const KernelLaunchParamsEx* kernels,
                                          std::uint32_t kernel_count) {
  using Function = NvStatus(__cdecl*)(void*, const KernelLaunchParamsEx*,
                                      std::uint32_t);
  const Function real = RealFunction<Function>(kIdLaunchKernelChainEx);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const std::uint64_t launch_token = NextToken(&g_next_launch_token);
  const NvStatus capture = CaptureKernelChain<true>(
      "NvAPI_D3D12_LaunchCuKernelChainEx", kernels, kernel_count,
      launch_token);
  if (capture != kNvapiOk) return capture;
  const NvStatus status = real(command_list, kernels, kernel_count);
  std::string event = BeginCall("NvAPI_D3D12_LaunchCuKernelChainEx");
  AppendToken(event, "launch_token", "launch", launch_token, true);
  AppendUnsigned(event, "kernel_count", kernel_count);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceDestroyCuModule(void* device, NvObjectHandle module) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle);
  const Function real = RealFunction<Function>(kIdDestroyCuModule);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const ObjectInfo info = FindModuleInfo(module);
  const NvStatus status = real(device, module);
  if (status == kNvapiOk) RemoveModule(module);
  std::string event = BeginCall("NvAPI_D3D12_DestroyCuModule");
  AppendToken(event, "module_token", "module", info.token, info.tracked);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceDestroyCuFunction(void* device,
                                        NvObjectHandle function) {
  using Function = NvStatus(__cdecl*)(void*, NvObjectHandle);
  const Function real = RealFunction<Function>(kIdDestroyCuFunction);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;
  const ObjectInfo info = FindFunctionInfo(function);
  const NvStatus status = real(device, function);
  if (status == kNvapiOk) RemoveFunction(function);
  std::string event = BeginCall("NvAPI_D3D12_DestroyCuFunction");
  AppendToken(event, "function_token", "function", info.token, info.tracked);
  AppendToken(event, "module_token", "module", info.parent_token,
              info.tracked && info.parent_token != 0);
  AppendString(event, "function_name", info.name);
  if (!FinishCall(event, status)) return kNvapiError;
  return status;
}

NvStatus __cdecl TraceCaptureUavInfo(void* device, UavInfoV2* info) {
  using Function = NvStatus(__cdecl*)(void*, UavInfoV2*);
  const Function real = RealFunction<Function>(kIdCaptureUavInfo);
  if (!RuntimeReady() || real == nullptr) return kNvapiError;

  char phase[129]{};
  GetCurrentPhase(phase);
  const std::uint64_t token = NextToken(&g_next_uav_info_token);
  const std::uint32_t input_version = info != nullptr ? info->version : 0;
  const NvStatus status = real(device, info);
  const std::uint32_t version = info != nullptr ? info->version : input_version;
  const bool output_valid = status == kNvapiOk && info != nullptr;
  const std::uint32_t surface_handle =
      output_valid ? info->surface_handle : 0;
  const std::uint64_t gpu_va_start = output_valid ? info->gpu_va_start : 0;
  const std::uint64_t gpu_va_size = output_valid ? info->gpu_va_size : 0;
  const std::uint64_t out_flags = output_valid ? info->out_flags : 0;

  std::string volatile_event = "{\"event\":\"uav_info_raw\"";
  AppendString(volatile_event, "phase", phase);
  AppendToken(volatile_event, "uav_info_token", "uav-info", token, true);
  AppendSigned(volatile_event, "status", status);
  AppendUnsigned(volatile_event, "version", version);
  AppendHex(volatile_event, "gpu_va_start", gpu_va_start, 16);
  AppendUnsigned(volatile_event, "gpu_va_size", gpu_va_size);
  AppendHex(volatile_event, "surface_handle", surface_handle, 8);
  FinishJson(volatile_event);
  const bool volatile_written = WriteVolatileTrace(volatile_event);

  std::string event = BeginCall("NvAPI_D3D12_CaptureUAVInfo");
  AppendString(event, "phase", phase);
  AppendToken(event, "uav_info_token", "uav-info", token, true);
  AppendUnsigned(event, "version", version);
  AppendUnsigned(event, "gpu_va_size", gpu_va_size);
  AppendUnsigned(event, "out_flags", out_flags);
  AppendBoolean(event, "volatile_trace_written", volatile_written);
  const bool main_written = FinishCall(event, status);
  return volatile_written && main_written ? status : kNvapiError;
}

}  // namespace

extern "C" __declspec(dllexport) void* __cdecl nvapi_QueryInterface(
    std::uint32_t id) {
  return ResolveInterface("nvapi_QueryInterface", false, id);
}

extern "C" __declspec(dllexport) void* __cdecl nvapi_Direct_GetMethod(
    std::uint32_t id) {
  return ResolveInterface("nvapi_Direct_GetMethod", true, id);
}

extern "C" __declspec(dllexport) void WINAPI
NeuralStudioFgNvapiTraceSetPhase(const char* phase) {
  if (!RuntimeReady()) return;
  if (!FlushOpaqueAd298d3fObservation()) {
    DebugReject("opaque interface call observation could not be written");
    return;
  }
  char sanitized[129]{};
  bool truncated = false;
  SanitizePhase(phase, sanitized, &truncated);
  SetCurrentPhase(sanitized);
  neuralstudio::fg_trace::DescriptorTraceSetPhase(sanitized);
  std::string event = "{\"event\":\"phase_marker\"";
  AppendString(event, "phase", sanitized);
  AppendBoolean(event, "truncated", truncated);
  FinishJson(event);
  if (!WriteTrace(event)) {
    DebugReject("phase marker could not be written");
  }
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_self = instance;
    DisableThreadLibraryCalls(instance);
  }
  return TRUE;
}
