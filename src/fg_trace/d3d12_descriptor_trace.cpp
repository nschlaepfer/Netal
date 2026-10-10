#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <d3d12.h>

#include "d3d12_descriptor_trace.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cwchar>
#include <iterator>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kMaxResourceRecords = 16384;
constexpr std::size_t kMaxHeapRecords = 4096;
constexpr std::size_t kMaxDescriptorRecords = 131072;
constexpr std::size_t kMaxHookRecords = 16;
constexpr std::uint32_t kMaxCopiedDescriptors = 1u << 20;

constexpr std::size_t kDeviceCreateDescriptorHeapIndex = 14;
constexpr std::size_t kDeviceCreateShaderResourceViewIndex = 18;
constexpr std::size_t kDeviceCreateUnorderedAccessViewIndex = 19;
constexpr std::size_t kDeviceCreateSamplerIndex = 22;
constexpr std::size_t kDeviceCopyDescriptorsIndex = 23;
constexpr std::size_t kDeviceCopyDescriptorsSimpleIndex = 24;
constexpr std::size_t kDeviceCreateCommittedResourceIndex = 27;
constexpr std::size_t kDeviceCreatePlacedResourceIndex = 29;
constexpr std::size_t kDeviceCreateReservedResourceIndex = 30;

enum class DescriptorKind : std::uint32_t {
  kUnknown = 0,
  kShaderResourceView = 1,
  kUnorderedAccessView = 2,
  kSampler = 3,
};

struct ResourceRecord {
  std::uintptr_t identity = 0;
  std::uint64_t token = 0;
  D3D12_RESOURCE_DESC description{};
  bool active = false;
};

struct HeapRecord {
  std::uintptr_t identity = 0;
  std::uintptr_t cpu_start = 0;
  std::uint64_t token = 0;
  D3D12_DESCRIPTOR_HEAP_DESC description{};
  std::uint32_t increment = 0;
  bool active = false;
};

struct DescriptorRecord {
  std::uintptr_t handle = 0;
  std::uint64_t token = 0;
  std::uint64_t heap_token = 0;
  std::uint32_t heap_slot = 0;
  DescriptorKind kind = DescriptorKind::kUnknown;
  std::uint64_t resource_token = 0;
  std::uint64_t counter_resource_token = 0;
  bool default_view = false;
  D3D12_SHADER_RESOURCE_VIEW_DESC srv{};
  D3D12_UNORDERED_ACCESS_VIEW_DESC uav{};
  D3D12_SAMPLER_DESC sampler{};
  bool active = false;
};

struct HookRecord {
  void** slot = nullptr;
  void* original = nullptr;
  void* replacement = nullptr;
  std::size_t index = 0;
  bool active = false;
};

using CreateDescriptorHeapFunction = HRESULT(STDMETHODCALLTYPE*)(
    ID3D12Device*, const D3D12_DESCRIPTOR_HEAP_DESC*, REFIID, void**);
using CreateShaderResourceViewFunction = void(STDMETHODCALLTYPE*)(
    ID3D12Device*, ID3D12Resource*,
    const D3D12_SHADER_RESOURCE_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateUnorderedAccessViewFunction = void(STDMETHODCALLTYPE*)(
    ID3D12Device*, ID3D12Resource*, ID3D12Resource*,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CreateSamplerFunction = void(STDMETHODCALLTYPE*)(
    ID3D12Device*, const D3D12_SAMPLER_DESC*, D3D12_CPU_DESCRIPTOR_HANDLE);
using CopyDescriptorsFunction = void(STDMETHODCALLTYPE*)(
    ID3D12Device*, UINT, const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*, UINT,
    const D3D12_CPU_DESCRIPTOR_HANDLE*, const UINT*,
    D3D12_DESCRIPTOR_HEAP_TYPE);
using CopyDescriptorsSimpleFunction = void(STDMETHODCALLTYPE*)(
    ID3D12Device*, UINT, D3D12_CPU_DESCRIPTOR_HANDLE,
    D3D12_CPU_DESCRIPTOR_HANDLE, D3D12_DESCRIPTOR_HEAP_TYPE);
using CreateCommittedResourceFunction = HRESULT(STDMETHODCALLTYPE*)(
    ID3D12Device*, const D3D12_HEAP_PROPERTIES*, D3D12_HEAP_FLAGS,
    const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreatePlacedResourceFunction = HRESULT(STDMETHODCALLTYPE*)(
    ID3D12Device*, ID3D12Heap*, UINT64, const D3D12_RESOURCE_DESC*,
    D3D12_RESOURCE_STATES, const D3D12_CLEAR_VALUE*, REFIID, void**);
using CreateReservedResourceFunction = HRESULT(STDMETHODCALLTYPE*)(
    ID3D12Device*, const D3D12_RESOURCE_DESC*, D3D12_RESOURCE_STATES,
    const D3D12_CLEAR_VALUE*, REFIID, void**);

std::wstring g_semantic_path;
std::wstring g_volatile_path;
char g_phase[129] = "unmarked";
ID3D12Device* g_attached_device = nullptr;
void** g_device_vtable = nullptr;
bool g_capture_enabled = false;
bool g_capture_accepting = false;
bool g_capture_healthy = true;
SRWLOCK g_file_lock = SRWLOCK_INIT;
SRWLOCK g_descriptor_lock = SRWLOCK_INIT;
ResourceRecord g_resource_records[kMaxResourceRecords]{};
HeapRecord g_heap_records[kMaxHeapRecords]{};
DescriptorRecord g_descriptor_records[kMaxDescriptorRecords]{};
HookRecord g_hook_records[kMaxHookRecords]{};
std::size_t g_resource_count = 0;
std::size_t g_heap_count = 0;
std::size_t g_descriptor_count = 0;
std::size_t g_hook_count = 0;
std::uint64_t g_request_count = 0;
std::uint64_t g_successful_request_count = 0;
std::uint64_t g_unresolved_request_count = 0;
std::uint64_t g_copy_count = 0;
volatile LONG64 g_next_resource_token = 0;
volatile LONG64 g_next_heap_token = 0;
volatile LONG64 g_next_descriptor_token = 0;
volatile LONG64 g_next_request_token = 0;

CreateDescriptorHeapFunction g_real_create_descriptor_heap = nullptr;
CreateShaderResourceViewFunction g_real_create_shader_resource_view = nullptr;
CreateUnorderedAccessViewFunction g_real_create_unordered_access_view = nullptr;
CreateSamplerFunction g_real_create_sampler = nullptr;
CopyDescriptorsFunction g_real_copy_descriptors = nullptr;
CopyDescriptorsSimpleFunction g_real_copy_descriptors_simple = nullptr;
CreateCommittedResourceFunction g_real_create_committed_resource = nullptr;
CreatePlacedResourceFunction g_real_create_placed_resource = nullptr;
CreateReservedResourceFunction g_real_create_reserved_resource = nullptr;

std::uint64_t NextToken(volatile LONG64* counter) {
  return static_cast<std::uint64_t>(InterlockedIncrement64(counter));
}

template <typename Destination, typename Source>
Destination PointerCast(Source source) {
  static_assert(sizeof(Destination) == sizeof(Source));
  Destination destination{};
  std::memcpy(&destination, &source, sizeof(destination));
  return destination;
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

bool IsDirectory(const std::wstring& path) {
  const DWORD attributes = GetFileAttributesW(path.c_str());
  return attributes != INVALID_FILE_ATTRIBUTES &&
         (attributes & FILE_ATTRIBUTE_DIRECTORY) != 0;
}

bool CreateNewTraceFile(const std::wstring& path) {
  HANDLE file = CreateFileW(path.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                            nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  if (file == INVALID_HANDLE_VALUE) return false;
  const bool result = FlushFileBuffers(file) != FALSE;
  CloseHandle(file);
  return result;
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

void AppendToken(std::string& json, const char* key, const char* prefix,
                 std::uint64_t token) {
  if (token == 0) {
    AppendString(json, key, "untracked");
    return;
  }
  char formatted[96]{};
  sprintf_s(formatted, "%s-%06llu", prefix,
            static_cast<unsigned long long>(token));
  AppendString(json, key, formatted);
}

void FinishJson(std::string& json) { json.push_back('}'); }

bool WriteJsonLine(const std::wstring& path, const std::string& json) {
  if (path.empty()) return false;
  std::string line = json;
  line.push_back('\n');
  AcquireSRWLockExclusive(&g_file_lock);
  HANDLE file = CreateFileW(path.c_str(), FILE_APPEND_DATA, FILE_SHARE_READ,
                            nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL,
                            nullptr);
  bool result = false;
  if (file != INVALID_HANDLE_VALUE) {
    DWORD written = 0;
    result = line.size() <= MAXDWORD &&
             WriteFile(file, line.data(), static_cast<DWORD>(line.size()),
                       &written, nullptr) &&
             written == static_cast<DWORD>(line.size()) &&
             FlushFileBuffers(file) != FALSE;
    CloseHandle(file);
  }
  ReleaseSRWLockExclusive(&g_file_lock);
  return result;
}

bool WriteSemantic(const std::string& event) {
  const bool result = WriteJsonLine(g_semantic_path, event);
  if (!result) g_capture_healthy = false;
  return result;
}

bool WriteVolatile(const std::string& event) {
  const bool result = WriteJsonLine(g_volatile_path, event);
  if (!result) g_capture_healthy = false;
  return result;
}

void CopyPhase(char output[129]) {
  AcquireSRWLockShared(&g_descriptor_lock);
  strcpy_s(output, 129, g_phase);
  ReleaseSRWLockShared(&g_descriptor_lock);
}

std::uint32_t FloatBits(float value) {
  std::uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value));
  std::memcpy(&bits, &value, sizeof(bits));
  return bits;
}

const char* DescriptorKindName(DescriptorKind kind) {
  switch (kind) {
    case DescriptorKind::kShaderResourceView: return "shader_resource_view";
    case DescriptorKind::kUnorderedAccessView: return "unordered_access_view";
    case DescriptorKind::kSampler: return "sampler";
    case DescriptorKind::kUnknown: break;
  }
  return "unknown";
}

void AppendResourceDescription(std::string& event,
                               const D3D12_RESOURCE_DESC& description) {
  AppendUnsigned(event, "dimension",
                 static_cast<std::uint32_t>(description.Dimension));
  AppendUnsigned(event, "alignment", description.Alignment);
  AppendUnsigned(event, "width", description.Width);
  AppendUnsigned(event, "height", description.Height);
  AppendUnsigned(event, "depth_or_array_size", description.DepthOrArraySize);
  AppendUnsigned(event, "mip_levels", description.MipLevels);
  AppendUnsigned(event, "format",
                 static_cast<std::uint32_t>(description.Format));
  AppendUnsigned(event, "sample_count", description.SampleDesc.Count);
  AppendUnsigned(event, "sample_quality", description.SampleDesc.Quality);
  AppendUnsigned(event, "layout",
                 static_cast<std::uint32_t>(description.Layout));
  AppendUnsigned(event, "resource_flags",
                 static_cast<std::uint32_t>(description.Flags));
}

void AppendClearValue(std::string& event, const D3D12_CLEAR_VALUE* value) {
  AppendBoolean(event, "clear_value_present", value != nullptr);
  if (value == nullptr) return;
  AppendUnsigned(event, "clear_value_format",
                 static_cast<std::uint32_t>(value->Format));
  std::array<std::uint32_t, 4> words{};
  static_assert(sizeof(words) == sizeof(value->Color));
  std::memcpy(words.data(), value->Color, sizeof(words));
  char encoded[64]{};
  sprintf_s(encoded, "0x%08X,0x%08X,0x%08X,0x%08X", words[0], words[1],
            words[2], words[3]);
  AppendString(event, "clear_value_union_words", encoded);
}

void AppendSrvDescription(std::string& event,
                          const D3D12_SHADER_RESOURCE_VIEW_DESC& value) {
  AppendUnsigned(event, "view_format", static_cast<std::uint32_t>(value.Format));
  AppendUnsigned(event, "view_dimension",
                 static_cast<std::uint32_t>(value.ViewDimension));
  AppendUnsigned(event, "shader_component_mapping",
                 value.Shader4ComponentMapping);
  switch (value.ViewDimension) {
    case D3D12_SRV_DIMENSION_BUFFER:
      AppendUnsigned(event, "first_element", value.Buffer.FirstElement);
      AppendUnsigned(event, "num_elements", value.Buffer.NumElements);
      AppendUnsigned(event, "structure_byte_stride",
                     value.Buffer.StructureByteStride);
      AppendUnsigned(event, "buffer_flags",
                     static_cast<std::uint32_t>(value.Buffer.Flags));
      break;
    case D3D12_SRV_DIMENSION_TEXTURE1D:
      AppendUnsigned(event, "most_detailed_mip", value.Texture1D.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.Texture1D.MipLevels);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.Texture1D.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURE1DARRAY:
      AppendUnsigned(event, "most_detailed_mip",
                     value.Texture1DArray.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.Texture1DArray.MipLevels);
      AppendUnsigned(event, "first_array_slice",
                     value.Texture1DArray.FirstArraySlice);
      AppendUnsigned(event, "array_size", value.Texture1DArray.ArraySize);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.Texture1DArray.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURE2D:
      AppendUnsigned(event, "most_detailed_mip", value.Texture2D.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.Texture2D.MipLevels);
      AppendUnsigned(event, "plane_slice", value.Texture2D.PlaneSlice);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.Texture2D.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURE2DARRAY:
      AppendUnsigned(event, "most_detailed_mip",
                     value.Texture2DArray.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.Texture2DArray.MipLevels);
      AppendUnsigned(event, "first_array_slice",
                     value.Texture2DArray.FirstArraySlice);
      AppendUnsigned(event, "array_size", value.Texture2DArray.ArraySize);
      AppendUnsigned(event, "plane_slice", value.Texture2DArray.PlaneSlice);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.Texture2DArray.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURE2DMSARRAY:
      AppendUnsigned(event, "first_array_slice",
                     value.Texture2DMSArray.FirstArraySlice);
      AppendUnsigned(event, "array_size", value.Texture2DMSArray.ArraySize);
      break;
    case D3D12_SRV_DIMENSION_TEXTURE3D:
      AppendUnsigned(event, "most_detailed_mip", value.Texture3D.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.Texture3D.MipLevels);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.Texture3D.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURECUBE:
      AppendUnsigned(event, "most_detailed_mip",
                     value.TextureCube.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.TextureCube.MipLevels);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.TextureCube.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_TEXTURECUBEARRAY:
      AppendUnsigned(event, "most_detailed_mip",
                     value.TextureCubeArray.MostDetailedMip);
      AppendUnsigned(event, "mip_count", value.TextureCubeArray.MipLevels);
      AppendUnsigned(event, "first_array_slice",
                     value.TextureCubeArray.First2DArrayFace);
      AppendUnsigned(event, "cube_count", value.TextureCubeArray.NumCubes);
      AppendHex(event, "min_lod_clamp_bits",
                FloatBits(value.TextureCubeArray.ResourceMinLODClamp), 8);
      break;
    case D3D12_SRV_DIMENSION_RAYTRACING_ACCELERATION_STRUCTURE:
      AppendBoolean(event, "gpu_virtual_address_recorded", false);
      break;
    case D3D12_SRV_DIMENSION_UNKNOWN:
    case D3D12_SRV_DIMENSION_TEXTURE2DMS:
      break;
  }
}

void AppendUavDescription(std::string& event,
                          const D3D12_UNORDERED_ACCESS_VIEW_DESC& value) {
  AppendUnsigned(event, "view_format", static_cast<std::uint32_t>(value.Format));
  AppendUnsigned(event, "view_dimension",
                 static_cast<std::uint32_t>(value.ViewDimension));
  switch (value.ViewDimension) {
    case D3D12_UAV_DIMENSION_BUFFER:
      AppendUnsigned(event, "first_element", value.Buffer.FirstElement);
      AppendUnsigned(event, "num_elements", value.Buffer.NumElements);
      AppendUnsigned(event, "structure_byte_stride",
                     value.Buffer.StructureByteStride);
      AppendUnsigned(event, "counter_offset_bytes",
                     value.Buffer.CounterOffsetInBytes);
      AppendUnsigned(event, "buffer_flags",
                     static_cast<std::uint32_t>(value.Buffer.Flags));
      break;
    case D3D12_UAV_DIMENSION_TEXTURE1D:
      AppendUnsigned(event, "mip_slice", value.Texture1D.MipSlice);
      break;
    case D3D12_UAV_DIMENSION_TEXTURE1DARRAY:
      AppendUnsigned(event, "mip_slice", value.Texture1DArray.MipSlice);
      AppendUnsigned(event, "first_array_slice",
                     value.Texture1DArray.FirstArraySlice);
      AppendUnsigned(event, "array_size", value.Texture1DArray.ArraySize);
      break;
    case D3D12_UAV_DIMENSION_TEXTURE2D:
      AppendUnsigned(event, "mip_slice", value.Texture2D.MipSlice);
      AppendUnsigned(event, "plane_slice", value.Texture2D.PlaneSlice);
      break;
    case D3D12_UAV_DIMENSION_TEXTURE2DARRAY:
      AppendUnsigned(event, "mip_slice", value.Texture2DArray.MipSlice);
      AppendUnsigned(event, "first_array_slice",
                     value.Texture2DArray.FirstArraySlice);
      AppendUnsigned(event, "array_size", value.Texture2DArray.ArraySize);
      AppendUnsigned(event, "plane_slice", value.Texture2DArray.PlaneSlice);
      break;
    case D3D12_UAV_DIMENSION_TEXTURE3D:
      AppendUnsigned(event, "mip_slice", value.Texture3D.MipSlice);
      AppendUnsigned(event, "first_depth_slice",
                     value.Texture3D.FirstWSlice);
      AppendUnsigned(event, "depth_size", value.Texture3D.WSize);
      break;
    case D3D12_UAV_DIMENSION_UNKNOWN:
      break;
  }
}

void AppendSamplerDescription(std::string& event,
                              const D3D12_SAMPLER_DESC& value) {
  AppendUnsigned(event, "filter", static_cast<std::uint32_t>(value.Filter));
  AppendUnsigned(event, "address_u",
                 static_cast<std::uint32_t>(value.AddressU));
  AppendUnsigned(event, "address_v",
                 static_cast<std::uint32_t>(value.AddressV));
  AppendUnsigned(event, "address_w",
                 static_cast<std::uint32_t>(value.AddressW));
  AppendHex(event, "mip_lod_bias_bits", FloatBits(value.MipLODBias), 8);
  AppendUnsigned(event, "max_anisotropy", value.MaxAnisotropy);
  AppendUnsigned(event, "comparison_function",
                 static_cast<std::uint32_t>(value.ComparisonFunc));
  char border[64]{};
  sprintf_s(border, "0x%08X,0x%08X,0x%08X,0x%08X",
            FloatBits(value.BorderColor[0]), FloatBits(value.BorderColor[1]),
            FloatBits(value.BorderColor[2]), FloatBits(value.BorderColor[3]));
  AppendString(event, "border_color_bits", border);
  AppendHex(event, "min_lod_bits", FloatBits(value.MinLOD), 8);
  AppendHex(event, "max_lod_bits", FloatBits(value.MaxLOD), 8);
}

std::uintptr_t CanonicalIdentity(IUnknown* object) {
  if (object == nullptr) return 0;
  IUnknown* identity = nullptr;
  const HRESULT status = object->QueryInterface(
      __uuidof(IUnknown), reinterpret_cast<void**>(&identity));
  if (FAILED(status) || identity == nullptr) return 0;
  const std::uintptr_t result = reinterpret_cast<std::uintptr_t>(identity);
  identity->Release();
  return result;
}

ResourceRecord FindResource(std::uintptr_t identity) {
  ResourceRecord result{};
  if (identity == 0) return result;
  AcquireSRWLockShared(&g_descriptor_lock);
  for (std::size_t index = g_resource_count; index > 0; --index) {
    const ResourceRecord& candidate = g_resource_records[index - 1];
    if (candidate.active && candidate.identity == identity) {
      result = candidate;
      break;
    }
  }
  ReleaseSRWLockShared(&g_descriptor_lock);
  return result;
}

ResourceRecord FindResourceByToken(std::uint64_t token) {
  ResourceRecord result{};
  if (token == 0) return result;
  AcquireSRWLockShared(&g_descriptor_lock);
  for (std::size_t index = g_resource_count; index > 0; --index) {
    const ResourceRecord& candidate = g_resource_records[index - 1];
    if (candidate.active && candidate.token == token) {
      result = candidate;
      break;
    }
  }
  ReleaseSRWLockShared(&g_descriptor_lock);
  return result;
}

ResourceRecord RecordResource(
    IUnknown* object, const D3D12_RESOURCE_DESC& description,
    const char* creation_method, const D3D12_HEAP_PROPERTIES* heap_properties,
    D3D12_HEAP_FLAGS heap_flags, D3D12_RESOURCE_STATES initial_state,
    const D3D12_CLEAR_VALUE* clear_value) {
  ResourceRecord record{};
  const std::uintptr_t identity = CanonicalIdentity(object);
  if (identity == 0) {
    g_capture_healthy = false;
    return record;
  }
  AcquireSRWLockExclusive(&g_descriptor_lock);
  if (g_resource_count < kMaxResourceRecords) {
    record.identity = identity;
    record.token = NextToken(&g_next_resource_token);
    record.description = description;
    record.active = true;
    g_resource_records[g_resource_count++] = record;
  } else {
    g_capture_healthy = false;
  }
  ReleaseSRWLockExclusive(&g_descriptor_lock);
  if (!record.active) return record;

  char phase[129]{};
  CopyPhase(phase);
  std::string event = "{\"event\":\"resource_created\"";
  AppendString(event, "phase", phase);
  AppendToken(event, "resource_token", "d3d-resource", record.token);
  AppendString(event, "creation_method", creation_method);
  AppendResourceDescription(event, description);
  AppendBoolean(event, "heap_properties_present", heap_properties != nullptr);
  if (heap_properties != nullptr) {
    AppendUnsigned(event, "heap_type",
                   static_cast<std::uint32_t>(heap_properties->Type));
    AppendUnsigned(event, "cpu_page_property",
                   static_cast<std::uint32_t>(heap_properties->CPUPageProperty));
    AppendUnsigned(event, "memory_pool_preference",
                   static_cast<std::uint32_t>(heap_properties->MemoryPoolPreference));
    AppendUnsigned(event, "creation_node_mask",
                   heap_properties->CreationNodeMask);
    AppendUnsigned(event, "visible_node_mask", heap_properties->VisibleNodeMask);
  }
  AppendUnsigned(event, "heap_flags", static_cast<std::uint32_t>(heap_flags));
  AppendUnsigned(event, "initial_state",
                 static_cast<std::uint32_t>(initial_state));
  AppendClearValue(event, clear_value);
  FinishJson(event);

  std::string raw = "{\"event\":\"resource_created_raw\"";
  AppendToken(raw, "resource_token", "d3d-resource", record.token);
  AppendHex(raw, "resource_identity", record.identity, 16);
  FinishJson(raw);
  if (!WriteSemantic(event) || !WriteVolatile(raw)) g_capture_healthy = false;
  return record;
}

ResourceRecord GetOrObserveResource(ID3D12Resource* resource) {
  if (resource == nullptr) return {};
  const std::uintptr_t identity = CanonicalIdentity(resource);
  ResourceRecord record = FindResource(identity);
  if (record.active) return record;
  return RecordResource(resource, resource->GetDesc(), "descriptor_observed",
                        nullptr, D3D12_HEAP_FLAG_NONE,
                        D3D12_RESOURCE_STATE_COMMON, nullptr);
}

HeapRecord FindHeapSlot(std::uintptr_t handle,
                        D3D12_DESCRIPTOR_HEAP_TYPE expected_type,
                        bool enforce_type) {
  HeapRecord result{};
  if (handle == 0) return result;
  AcquireSRWLockShared(&g_descriptor_lock);
  for (std::size_t index = g_heap_count; index > 0; --index) {
    const HeapRecord& candidate = g_heap_records[index - 1];
    if (!candidate.active || candidate.increment == 0 ||
        (enforce_type && candidate.description.Type != expected_type) ||
        handle < candidate.cpu_start) {
      continue;
    }
    const std::uintptr_t delta = handle - candidate.cpu_start;
    if (delta % candidate.increment != 0) continue;
    const std::uint64_t slot = delta / candidate.increment;
    if (slot >= candidate.description.NumDescriptors) continue;
    result = candidate;
    break;
  }
  ReleaseSRWLockShared(&g_descriptor_lock);
  return result;
}

std::uint32_t HeapSlot(const HeapRecord& heap, std::uintptr_t handle) {
  if (!heap.active || heap.increment == 0 || handle < heap.cpu_start) return 0;
  return static_cast<std::uint32_t>((handle - heap.cpu_start) / heap.increment);
}

DescriptorRecord FindDescriptor(std::uintptr_t handle) {
  DescriptorRecord result{};
  if (handle == 0) return result;
  AcquireSRWLockShared(&g_descriptor_lock);
  for (std::size_t index = g_descriptor_count; index > 0; --index) {
    const DescriptorRecord& candidate = g_descriptor_records[index - 1];
    if (candidate.active && candidate.handle == handle) {
      result = candidate;
      break;
    }
  }
  ReleaseSRWLockShared(&g_descriptor_lock);
  return result;
}

bool WriteDescriptorRecord(const DescriptorRecord& record, const char* event_name,
                           std::uint64_t source_descriptor_token,
                           std::uintptr_t source_handle) {
  std::string event = "{\"event\":" + JsonQuote(event_name);
  char phase[129]{};
  CopyPhase(phase);
  AppendString(event, "phase", phase);
  AppendToken(event, "descriptor_token", "descriptor", record.token);
  AppendToken(event, "descriptor_heap_token", "descriptor-heap",
              record.heap_token);
  AppendUnsigned(event, "descriptor_heap_slot", record.heap_slot);
  AppendString(event, "descriptor_kind", DescriptorKindName(record.kind));
  AppendToken(event, "resource_token", "d3d-resource", record.resource_token);
  AppendToken(event, "counter_resource_token", "d3d-resource",
              record.counter_resource_token);
  AppendBoolean(event, "default_view", record.default_view);
  if (source_descriptor_token != 0) {
    AppendToken(event, "source_descriptor_token", "descriptor",
                source_descriptor_token);
  }
  if (record.kind == DescriptorKind::kShaderResourceView &&
      !record.default_view) {
    AppendSrvDescription(event, record.srv);
  } else if (record.kind == DescriptorKind::kUnorderedAccessView &&
             !record.default_view) {
    AppendUavDescription(event, record.uav);
  } else if (record.kind == DescriptorKind::kSampler) {
    AppendSamplerDescription(event, record.sampler);
  }
  FinishJson(event);

  std::string raw = "{\"event\":\"descriptor_location_raw\"";
  AppendToken(raw, "descriptor_token", "descriptor", record.token);
  AppendHex(raw, "cpu_descriptor_handle", record.handle, 16);
  const ResourceRecord resource = FindResourceByToken(record.resource_token);
  const ResourceRecord counter =
      FindResourceByToken(record.counter_resource_token);
  if (resource.active) {
    AppendHex(raw, "resource_identity", resource.identity, 16);
  }
  if (counter.active) {
    AppendHex(raw, "counter_resource_identity", counter.identity, 16);
  }
  if (source_handle != 0) {
    AppendHex(raw, "source_cpu_descriptor_handle", source_handle, 16);
  }
  FinishJson(raw);
  return WriteSemantic(event) && WriteVolatile(raw);
}

DescriptorRecord StoreDescriptor(DescriptorRecord record,
                                 const char* event_name,
                                 std::uint64_t source_descriptor_token = 0,
                                 std::uintptr_t source_handle = 0) {
  const D3D12_DESCRIPTOR_HEAP_TYPE type =
      record.kind == DescriptorKind::kSampler
          ? D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER
          : D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
  const HeapRecord heap = FindHeapSlot(record.handle, type, true);
  record.heap_token = heap.token;
  record.heap_slot = HeapSlot(heap, record.handle);
  if (!heap.active) g_capture_healthy = false;

  AcquireSRWLockExclusive(&g_descriptor_lock);
  if (g_descriptor_count < kMaxDescriptorRecords) {
    record.token = NextToken(&g_next_descriptor_token);
    record.active = true;
    g_descriptor_records[g_descriptor_count++] = record;
  } else {
    g_capture_healthy = false;
  }
  ReleaseSRWLockExclusive(&g_descriptor_lock);
  if (record.active && !WriteDescriptorRecord(
                           record, event_name, source_descriptor_token,
                           source_handle)) {
    g_capture_healthy = false;
  }
  return record;
}

HRESULT STDMETHODCALLTYPE HookCreateDescriptorHeap(
    ID3D12Device* device, const D3D12_DESCRIPTOR_HEAP_DESC* description,
    REFIID riid, void** heap_out) {
  const HRESULT status =
      g_real_create_descriptor_heap(device, description, riid, heap_out);
  if (!g_capture_accepting || FAILED(status) || heap_out == nullptr ||
      *heap_out == nullptr || description == nullptr) {
    return status;
  }
  auto* heap = reinterpret_cast<ID3D12DescriptorHeap*>(*heap_out);
  HeapRecord record{};
  record.identity = CanonicalIdentity(heap);
  record.cpu_start = heap->GetCPUDescriptorHandleForHeapStart().ptr;
  record.description = *description;
  record.increment =
      device->GetDescriptorHandleIncrementSize(description->Type);
  AcquireSRWLockExclusive(&g_descriptor_lock);
  if (record.identity != 0 && record.increment != 0 &&
      g_heap_count < kMaxHeapRecords) {
    record.token = NextToken(&g_next_heap_token);
    record.active = true;
    g_heap_records[g_heap_count++] = record;
  } else {
    g_capture_healthy = false;
  }
  ReleaseSRWLockExclusive(&g_descriptor_lock);
  if (!record.active) return status;

  char phase[129]{};
  CopyPhase(phase);
  std::string event = "{\"event\":\"descriptor_heap_created\"";
  AppendString(event, "phase", phase);
  AppendToken(event, "descriptor_heap_token", "descriptor-heap", record.token);
  AppendUnsigned(event, "heap_type",
                 static_cast<std::uint32_t>(description->Type));
  AppendUnsigned(event, "descriptor_count", description->NumDescriptors);
  AppendUnsigned(event, "heap_flags",
                 static_cast<std::uint32_t>(description->Flags));
  AppendUnsigned(event, "node_mask", description->NodeMask);
  AppendUnsigned(event, "descriptor_increment", record.increment);
  FinishJson(event);
  std::string raw = "{\"event\":\"descriptor_heap_raw\"";
  AppendToken(raw, "descriptor_heap_token", "descriptor-heap", record.token);
  AppendHex(raw, "heap_identity", record.identity, 16);
  AppendHex(raw, "cpu_descriptor_start", record.cpu_start, 16);
  FinishJson(raw);
  if (!WriteSemantic(event) || !WriteVolatile(raw)) g_capture_healthy = false;
  return status;
}

void STDMETHODCALLTYPE HookCreateShaderResourceView(
    ID3D12Device* device, ID3D12Resource* resource,
    const D3D12_SHADER_RESOURCE_VIEW_DESC* description,
    D3D12_CPU_DESCRIPTOR_HANDLE destination) {
  g_real_create_shader_resource_view(device, resource, description, destination);
  if (!g_capture_accepting) return;
  DescriptorRecord record{};
  record.handle = destination.ptr;
  record.kind = DescriptorKind::kShaderResourceView;
  record.resource_token = GetOrObserveResource(resource).token;
  record.default_view = description == nullptr;
  if (description != nullptr) record.srv = *description;
  StoreDescriptor(record, "descriptor_written");
}

void STDMETHODCALLTYPE HookCreateUnorderedAccessView(
    ID3D12Device* device, ID3D12Resource* resource,
    ID3D12Resource* counter_resource,
    const D3D12_UNORDERED_ACCESS_VIEW_DESC* description,
    D3D12_CPU_DESCRIPTOR_HANDLE destination) {
  g_real_create_unordered_access_view(device, resource, counter_resource,
                                      description, destination);
  if (!g_capture_accepting) return;
  DescriptorRecord record{};
  record.handle = destination.ptr;
  record.kind = DescriptorKind::kUnorderedAccessView;
  record.resource_token = GetOrObserveResource(resource).token;
  record.counter_resource_token = GetOrObserveResource(counter_resource).token;
  record.default_view = description == nullptr;
  if (description != nullptr) record.uav = *description;
  StoreDescriptor(record, "descriptor_written");
}

void STDMETHODCALLTYPE HookCreateSampler(
    ID3D12Device* device, const D3D12_SAMPLER_DESC* description,
    D3D12_CPU_DESCRIPTOR_HANDLE destination) {
  g_real_create_sampler(device, description, destination);
  if (!g_capture_accepting) return;
  DescriptorRecord record{};
  record.handle = destination.ptr;
  record.kind = DescriptorKind::kSampler;
  if (description != nullptr) {
    record.sampler = *description;
  } else {
    g_capture_healthy = false;
  }
  StoreDescriptor(record, "descriptor_written");
}

void StoreDescriptorCopy(std::uintptr_t destination,
                         std::uintptr_t source) {
  const DescriptorRecord source_record = FindDescriptor(source);
  if (!source_record.active) {
    g_capture_healthy = false;
    return;
  }
  DescriptorRecord destination_record = source_record;
  destination_record.handle = destination;
  destination_record.token = 0;
  destination_record.heap_token = 0;
  destination_record.heap_slot = 0;
  destination_record.active = false;
  StoreDescriptor(destination_record, "descriptor_copied",
                  source_record.token, source);
  ++g_copy_count;
}

void STDMETHODCALLTYPE HookCopyDescriptors(
    ID3D12Device* device, UINT destination_range_count,
    const D3D12_CPU_DESCRIPTOR_HANDLE* destination_range_starts,
    const UINT* destination_range_sizes, UINT source_range_count,
    const D3D12_CPU_DESCRIPTOR_HANDLE* source_range_starts,
    const UINT* source_range_sizes, D3D12_DESCRIPTOR_HEAP_TYPE heap_type) {
  const UINT increment = device->GetDescriptorHandleIncrementSize(heap_type);
  std::vector<std::uintptr_t> destinations;
  std::vector<std::uintptr_t> sources;
  if (g_capture_accepting && increment != 0 &&
      destination_range_starts != nullptr && source_range_starts != nullptr) {
    for (UINT range = 0; range < destination_range_count; ++range) {
      const UINT count =
          destination_range_sizes != nullptr ? destination_range_sizes[range] : 1u;
      if (destinations.size() + count > kMaxCopiedDescriptors) break;
      for (UINT index = 0; index < count; ++index) {
        destinations.push_back(destination_range_starts[range].ptr +
                               static_cast<std::uintptr_t>(index) * increment);
      }
    }
    for (UINT range = 0; range < source_range_count; ++range) {
      const UINT count =
          source_range_sizes != nullptr ? source_range_sizes[range] : 1u;
      if (sources.size() + count > kMaxCopiedDescriptors) break;
      for (UINT index = 0; index < count; ++index) {
        sources.push_back(source_range_starts[range].ptr +
                          static_cast<std::uintptr_t>(index) * increment);
      }
    }
  }
  g_real_copy_descriptors(device, destination_range_count,
                          destination_range_starts, destination_range_sizes,
                          source_range_count, source_range_starts,
                          source_range_sizes, heap_type);
  if (!g_capture_accepting) return;
  if (destinations.size() != sources.size() || destinations.empty()) {
    g_capture_healthy = false;
    return;
  }
  for (std::size_t index = 0; index < destinations.size(); ++index) {
    StoreDescriptorCopy(destinations[index], sources[index]);
  }
}

void STDMETHODCALLTYPE HookCopyDescriptorsSimple(
    ID3D12Device* device, UINT descriptor_count,
    D3D12_CPU_DESCRIPTOR_HANDLE destination_start,
    D3D12_CPU_DESCRIPTOR_HANDLE source_start,
    D3D12_DESCRIPTOR_HEAP_TYPE heap_type) {
  const UINT increment = device->GetDescriptorHandleIncrementSize(heap_type);
  g_real_copy_descriptors_simple(device, descriptor_count, destination_start,
                                 source_start, heap_type);
  if (!g_capture_accepting) return;
  if (increment == 0 || descriptor_count > kMaxCopiedDescriptors) {
    g_capture_healthy = false;
    return;
  }
  for (UINT index = 0; index < descriptor_count; ++index) {
    const std::uintptr_t offset =
        static_cast<std::uintptr_t>(index) * increment;
    StoreDescriptorCopy(destination_start.ptr + offset,
                        source_start.ptr + offset);
  }
}

HRESULT STDMETHODCALLTYPE HookCreateCommittedResource(
    ID3D12Device* device, const D3D12_HEAP_PROPERTIES* heap_properties,
    D3D12_HEAP_FLAGS heap_flags, const D3D12_RESOURCE_DESC* description,
    D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE* clear_value,
    REFIID riid, void** resource_out) {
  const HRESULT status = g_real_create_committed_resource(
      device, heap_properties, heap_flags, description, initial_state,
      clear_value, riid, resource_out);
  if (g_capture_accepting && SUCCEEDED(status) && description != nullptr &&
      resource_out != nullptr && *resource_out != nullptr) {
    RecordResource(reinterpret_cast<IUnknown*>(*resource_out), *description,
                   "create_committed_resource", heap_properties, heap_flags,
                   initial_state, clear_value);
  }
  return status;
}

HRESULT STDMETHODCALLTYPE HookCreatePlacedResource(
    ID3D12Device* device, ID3D12Heap* heap, UINT64 heap_offset,
    const D3D12_RESOURCE_DESC* description,
    D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE* clear_value,
    REFIID riid, void** resource_out) {
  const HRESULT status = g_real_create_placed_resource(
      device, heap, heap_offset, description, initial_state, clear_value, riid,
      resource_out);
  if (g_capture_accepting && SUCCEEDED(status) && description != nullptr &&
      resource_out != nullptr && *resource_out != nullptr) {
    D3D12_HEAP_PROPERTIES properties{};
    D3D12_HEAP_FLAGS flags = D3D12_HEAP_FLAG_NONE;
    if (heap != nullptr) {
      const D3D12_HEAP_DESC heap_description = heap->GetDesc();
      properties = heap_description.Properties;
      flags = heap_description.Flags;
    }
    RecordResource(reinterpret_cast<IUnknown*>(*resource_out), *description,
                   "create_placed_resource", heap != nullptr ? &properties : nullptr,
                   flags, initial_state, clear_value);
  }
  return status;
}

HRESULT STDMETHODCALLTYPE HookCreateReservedResource(
    ID3D12Device* device, const D3D12_RESOURCE_DESC* description,
    D3D12_RESOURCE_STATES initial_state, const D3D12_CLEAR_VALUE* clear_value,
    REFIID riid, void** resource_out) {
  const HRESULT status = g_real_create_reserved_resource(
      device, description, initial_state, clear_value, riid, resource_out);
  if (g_capture_accepting && SUCCEEDED(status) && description != nullptr &&
      resource_out != nullptr && *resource_out != nullptr) {
    RecordResource(reinterpret_cast<IUnknown*>(*resource_out), *description,
                   "create_reserved_resource", nullptr,
                   D3D12_HEAP_FLAG_NONE, initial_state, clear_value);
  }
  return status;
}

bool PatchSlot(std::size_t index, void* replacement, void** original_out) {
  if (g_device_vtable == nullptr || replacement == nullptr ||
      original_out == nullptr || g_hook_count >= kMaxHookRecords) {
    return false;
  }
  void** slot = &g_device_vtable[index];
  DWORD old_protection = 0;
  if (!VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &old_protection)) {
    return false;
  }
  const void* original = InterlockedExchangePointer(
      reinterpret_cast<PVOID volatile*>(slot), replacement);
  DWORD ignored = 0;
  const bool protection_restored =
      VirtualProtect(slot, sizeof(void*), old_protection, &ignored) != FALSE;
  if (original == nullptr || !protection_restored) return false;
  *original_out = const_cast<void*>(original);
  g_hook_records[g_hook_count++] =
      HookRecord{slot, const_cast<void*>(original), replacement, index, true};
  return true;
}

bool InstallHooks(ID3D12Device* device) {
  g_device_vtable = *reinterpret_cast<void***>(device);
  void* original = nullptr;
#define INSTALL_HOOK(index, hook, destination)                                \
  do {                                                                        \
    if (!PatchSlot(index, PointerCast<void*>(&hook), &original)) return false; \
    destination = PointerCast<decltype(destination)>(original);                \
  } while (false)
  INSTALL_HOOK(kDeviceCreateDescriptorHeapIndex, HookCreateDescriptorHeap,
               g_real_create_descriptor_heap);
  INSTALL_HOOK(kDeviceCreateShaderResourceViewIndex,
               HookCreateShaderResourceView, g_real_create_shader_resource_view);
  INSTALL_HOOK(kDeviceCreateUnorderedAccessViewIndex,
               HookCreateUnorderedAccessView,
               g_real_create_unordered_access_view);
  INSTALL_HOOK(kDeviceCreateSamplerIndex, HookCreateSampler,
               g_real_create_sampler);
  INSTALL_HOOK(kDeviceCopyDescriptorsIndex, HookCopyDescriptors,
               g_real_copy_descriptors);
  INSTALL_HOOK(kDeviceCopyDescriptorsSimpleIndex, HookCopyDescriptorsSimple,
               g_real_copy_descriptors_simple);
  INSTALL_HOOK(kDeviceCreateCommittedResourceIndex, HookCreateCommittedResource,
               g_real_create_committed_resource);
  INSTALL_HOOK(kDeviceCreatePlacedResourceIndex, HookCreatePlacedResource,
               g_real_create_placed_resource);
  INSTALL_HOOK(kDeviceCreateReservedResourceIndex, HookCreateReservedResource,
               g_real_create_reserved_resource);
#undef INSTALL_HOOK
  return true;
}

bool RestoreHooks() {
  bool result = true;
  for (std::size_t index = g_hook_count; index > 0; --index) {
    HookRecord& hook = g_hook_records[index - 1];
    if (!hook.active || hook.slot == nullptr) continue;
    DWORD old_protection = 0;
    if (!VirtualProtect(hook.slot, sizeof(void*), PAGE_READWRITE,
                        &old_protection)) {
      result = false;
      continue;
    }
    void* observed = InterlockedCompareExchangePointer(
        reinterpret_cast<PVOID volatile*>(hook.slot), hook.original,
        hook.replacement);
    if (observed != hook.replacement) result = false;
    DWORD ignored = 0;
    if (!VirtualProtect(hook.slot, sizeof(void*), old_protection, &ignored)) {
      result = false;
    }
    hook.active = false;
  }
  return result;
}

DescriptorKind ExpectedDescriptorKind(const char* resource_kind,
                                      bool sampler_position) {
  if (sampler_position ||
      std::strcmp(resource_kind, "independent_sampler") == 0) {
    return DescriptorKind::kSampler;
  }
  if (std::strcmp(resource_kind, "independent_surface") == 0 ||
      std::strcmp(resource_kind, "surface") == 0) {
    return DescriptorKind::kUnorderedAccessView;
  }
  return DescriptorKind::kShaderResourceView;
}

}  // namespace

namespace neuralstudio::fg_trace {

void DescriptorTraceSetPhase(const char* phase) {
  if (phase == nullptr) return;
  AcquireSRWLockExclusive(&g_descriptor_lock);
  strncpy_s(g_phase, phase, _TRUNCATE);
  ReleaseSRWLockExclusive(&g_descriptor_lock);
}

DescriptorTraceRequestSnapshot DescriptorTraceSnapshotResourceObject(
    const char* resource_kind, std::uintptr_t descriptor_a,
    std::uintptr_t descriptor_b, bool has_descriptor_b) {
  DescriptorTraceRequestSnapshot snapshot{};
  if (!g_capture_enabled) return snapshot;
  const DescriptorRecord first = FindDescriptor(descriptor_a);
  const DescriptorRecord second =
      has_descriptor_b ? FindDescriptor(descriptor_b) : DescriptorRecord{};
  snapshot.descriptor_token = first.token;
  snapshot.sampler_descriptor_token = second.token;
  snapshot.descriptor_kind_matches =
      first.active &&
      first.kind == ExpectedDescriptorKind(resource_kind, false);
  snapshot.sampler_kind_matches =
      !has_descriptor_b ||
      (second.active && second.kind == DescriptorKind::kSampler);
  snapshot.capture_enabled = true;
  return snapshot;
}

bool DescriptorTraceRecordResourceObject(
    const char* call, const char* phase, const char* resource_kind,
    std::uint64_t nvapi_resource_token, std::int32_t status,
    std::uintptr_t descriptor_a, std::uintptr_t descriptor_b,
    bool has_descriptor_b, const DescriptorTraceRequestSnapshot& snapshot) {
  if (!g_capture_enabled) return true;
  const DescriptorKind expected_first =
      ExpectedDescriptorKind(resource_kind, false);
  const bool first_valid = snapshot.capture_enabled &&
                           snapshot.descriptor_kind_matches;
  const bool second_valid = snapshot.capture_enabled &&
                            snapshot.sampler_kind_matches;
  const bool resolved = status != 0 || (first_valid && second_valid);
  const std::uint64_t request_token = NextToken(&g_next_request_token);

  AcquireSRWLockExclusive(&g_descriptor_lock);
  ++g_request_count;
  if (status == 0) {
    ++g_successful_request_count;
    if (!resolved) {
      ++g_unresolved_request_count;
      g_capture_healthy = false;
    }
  }
  ReleaseSRWLockExclusive(&g_descriptor_lock);

  std::string event = "{\"event\":\"resource_object_request\"";
  AppendToken(event, "request_token", "resource-request", request_token);
  AppendString(event, "phase", phase);
  AppendString(event, "call", call);
  AppendString(event, "resource_kind", resource_kind);
  AppendString(event, "snapshot_timing", "pre_real_nvapi_call");
  AppendToken(event, "nvapi_resource_token", "resource",
              nvapi_resource_token);
  AppendSigned(event, "status", status);
  AppendToken(event, "descriptor_token", "descriptor",
              snapshot.descriptor_token);
  AppendString(event, "expected_descriptor_kind",
               DescriptorKindName(expected_first));
  AppendBoolean(event, "descriptor_kind_matches", first_valid);
  if (has_descriptor_b) {
    AppendToken(event, "sampler_descriptor_token", "descriptor",
                snapshot.sampler_descriptor_token);
    AppendBoolean(event, "sampler_kind_matches", second_valid);
  }
  AppendBoolean(event, "mapping_resolved", resolved);
  FinishJson(event);

  std::string raw = "{\"event\":\"resource_object_request_raw\"";
  AppendToken(raw, "request_token", "resource-request", request_token);
  AppendToken(raw, "nvapi_resource_token", "resource",
              nvapi_resource_token);
  AppendHex(raw, "cpu_descriptor_handle", descriptor_a, 16);
  if (has_descriptor_b) {
    AppendHex(raw, "sampler_cpu_descriptor_handle", descriptor_b, 16);
  }
  FinishJson(raw);
  const bool written = WriteSemantic(event) && WriteVolatile(raw);
  return written && resolved;
}

}  // namespace neuralstudio::fg_trace

extern "C" __declspec(dllexport) BOOL WINAPI
NeuralStudioFgNvapiTraceAttachD3D12Device(void* device_pointer) {
  g_semantic_path =
      GetEnvironmentString(L"NEURALSTUDIO_FG_D3D12_DESCRIPTOR_TRACE");
  g_volatile_path = GetEnvironmentString(
      L"NEURALSTUDIO_FG_D3D12_DESCRIPTOR_VOLATILE_TRACE");
  if (g_semantic_path.empty() && g_volatile_path.empty()) return TRUE;
  if (g_semantic_path.empty() || g_volatile_path.empty() ||
      !IsAbsoluteWindowsPath(g_semantic_path) ||
      !IsAbsoluteWindowsPath(g_volatile_path) ||
      _wcsicmp(g_semantic_path.c_str(), g_volatile_path.c_str()) == 0 ||
      !IsDirectory(ParentPath(g_semantic_path)) ||
      !IsDirectory(ParentPath(g_volatile_path)) || device_pointer == nullptr ||
      !CreateNewTraceFile(g_semantic_path) ||
      !CreateNewTraceFile(g_volatile_path)) {
    g_capture_healthy = false;
    return FALSE;
  }

  g_attached_device = static_cast<ID3D12Device*>(device_pointer);
  g_capture_enabled = true;
  std::string ready =
      "{\"event\":\"descriptor_runtime_ready\",\"schema\":"
      "\"neuralstudio.fg.resource-descriptor-trace.v1\"";
  AppendString(ready, "instrumentation", "id3d12device_vtable");
  AppendBoolean(ready, "addresses_recorded", false);
  AppendBoolean(ready, "opaque_cuda_handles_recorded", false);
  AppendBoolean(ready, "gpu_virtual_addresses_recorded", false);
  FinishJson(ready);
  std::string raw_ready =
      "{\"event\":\"descriptor_volatile_runtime_ready\",\"schema\":"
      "\"neuralstudio.fg.resource-descriptor-volatile.v1\"}";
  if (!WriteSemantic(ready) || !WriteVolatile(raw_ready) ||
      !InstallHooks(g_attached_device)) {
    g_capture_accepting = false;
    g_capture_healthy = false;
    RestoreHooks();
    return FALSE;
  }
  g_capture_accepting = true;
  char phase[129]{};
  CopyPhase(phase);
  std::string attached = "{\"event\":\"device_attached\"";
  AppendString(attached, "phase", phase);
  AppendUnsigned(attached, "hook_count", g_hook_count);
  AppendBoolean(attached, "device_address_recorded", false);
  FinishJson(attached);
  if (!WriteSemantic(attached)) {
    g_capture_healthy = false;
    return FALSE;
  }
  return TRUE;
}

extern "C" __declspec(dllexport) BOOL WINAPI
NeuralStudioFgNvapiTraceDetachD3D12Device() {
  if (!g_capture_enabled) return TRUE;
  g_capture_accepting = false;
  const bool hooks_restored = RestoreHooks();
  if (!hooks_restored) g_capture_healthy = false;
  char phase[129]{};
  CopyPhase(phase);
  std::string summary = "{\"event\":\"descriptor_capture_summary\"";
  AppendString(summary, "phase", phase);
  AppendUnsigned(summary, "resource_count", g_resource_count);
  AppendUnsigned(summary, "descriptor_heap_count", g_heap_count);
  AppendUnsigned(summary, "descriptor_write_count", g_descriptor_count);
  AppendUnsigned(summary, "descriptor_copy_count", g_copy_count);
  AppendUnsigned(summary, "request_count", g_request_count);
  AppendUnsigned(summary, "successful_request_count",
                 g_successful_request_count);
  AppendUnsigned(summary, "unresolved_successful_request_count",
                 g_unresolved_request_count);
  AppendBoolean(summary, "hooks_restored", hooks_restored);
  AppendBoolean(summary, "capture_healthy", g_capture_healthy);
  AppendBoolean(summary, "addresses_recorded", false);
  FinishJson(summary);
  const bool written = WriteSemantic(summary);
  g_capture_enabled = false;
  g_attached_device = nullptr;
  g_device_vtable = nullptr;
  return written && hooks_restored && g_capture_healthy ? TRUE : FALSE;
}
