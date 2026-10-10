#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <bcrypt.h>
#include <d3d12.h>
#include <d3d12sdklayers.h>
#include <dxgi1_6.h>
#include <wrl/client.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_defs_dlssg.h>

#include "fg_parameter_contract.h"
#include "ngx_parameter_store.h"
#include "runtime_allocation_capture.h"
#include "synthetic_contract.h"
#include "offline_sequence.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cwchar>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#pragma comment(lib, "bcrypt.lib")
#pragma comment(lib, "d3d12.lib")
#pragma comment(lib, "dxgi.lib")

using Microsoft::WRL::ComPtr;

namespace {

constexpr char kExpectedRuntimeSha256[] =
    "5d5cbf14d2727d47f93fd10bf77bd91708ae122482a6f86fd564971641ebd47b";
constexpr NVSDK_NGX_Feature kFrameGenerationFeature =
    NVSDK_NGX_Feature_FrameGeneration;
constexpr DXGI_FORMAT kColorFormat = DXGI_FORMAT_R8G8B8A8_UNORM;
constexpr DXGI_FORMAT kMotionFormat = DXGI_FORMAT_R16G16_FLOAT;
constexpr DXGI_FORMAT kDepthFormat = DXGI_FORMAT_R32_FLOAT;
constexpr DXGI_FORMAT kUiAlphaFormat = DXGI_FORMAT_R8_UNORM;
constexpr DXGI_FORMAT kDistortionFormat = DXGI_FORMAT_R16G16B16A16_FLOAT;

struct LibraryCloser {
  void operator()(HMODULE module) const noexcept {
    if (module != nullptr) FreeLibrary(module);
  }
};
using UniqueLibrary =
    std::unique_ptr<std::remove_pointer_t<HMODULE>, LibraryCloser>;

struct HandleCloser {
  void operator()(HANDLE handle) const noexcept {
    if (handle != nullptr && handle != INVALID_HANDLE_VALUE) CloseHandle(handle);
  }
};
using UniqueHandle =
    std::unique_ptr<std::remove_pointer_t<HANDLE>, HandleCloser>;

std::string WideToUtf8(std::wstring_view input) {
  if (input.empty()) return {};
  const int size = WideCharToMultiByte(
      CP_UTF8, WC_ERR_INVALID_CHARS, input.data(),
      static_cast<int>(input.size()), nullptr, 0, nullptr, nullptr);
  if (size <= 0) throw std::runtime_error("WideCharToMultiByte size failed");
  std::string output(static_cast<std::size_t>(size), '\0');
  if (WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, input.data(),
                          static_cast<int>(input.size()), output.data(), size,
                          nullptr, nullptr) != size) {
    throw std::runtime_error("WideCharToMultiByte conversion failed");
  }
  return output;
}

std::string JsonEscape(std::string_view value) {
  std::ostringstream output;
  for (const unsigned char byte : value) {
    switch (byte) {
      case '\\': output << "\\\\"; break;
      case '"': output << "\\\""; break;
      case '\b': output << "\\b"; break;
      case '\f': output << "\\f"; break;
      case '\n': output << "\\n"; break;
      case '\r': output << "\\r"; break;
      case '\t': output << "\\t"; break;
      default:
        if (byte < 0x20u) {
          output << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                 << static_cast<unsigned int>(byte) << std::dec;
        } else {
          output << static_cast<char>(byte);
        }
    }
  }
  return output.str();
}

SRWLOCK g_ngx_log_lock = SRWLOCK_INIT;
HANDLE g_ngx_log_file = INVALID_HANDLE_VALUE;
volatile LONG g_ngx_log_write_failed = 0;

void NVSDK_CONV CaptureNgxLog(const char* message,
                              NVSDK_NGX_Logging_Level level,
                              NVSDK_NGX_Feature component) {
  constexpr std::size_t kMaxMessageBytes = 16384u;
  std::string_view source = message != nullptr ? std::string_view(message)
                                               : std::string_view{};
  const bool truncated = source.size() > kMaxMessageBytes;
  if (truncated) source = source.substr(0, kMaxMessageBytes);
  std::ostringstream line;
  line << "level=" << static_cast<unsigned int>(level)
       << " component=" << static_cast<int>(component) << " message=\""
       << JsonEscape(source) << '"';
  if (truncated) line << " truncated=true";
  line << '\n';
  const std::string encoded = line.str();

  AcquireSRWLockExclusive(&g_ngx_log_lock);
  bool written = false;
  if (g_ngx_log_file != INVALID_HANDLE_VALUE &&
      encoded.size() <= static_cast<std::size_t>(MAXDWORD)) {
    DWORD bytes_written = 0;
    written = WriteFile(g_ngx_log_file, encoded.data(),
                        static_cast<DWORD>(encoded.size()), &bytes_written,
                        nullptr) != FALSE &&
              bytes_written == static_cast<DWORD>(encoded.size()) &&
              FlushFileBuffers(g_ngx_log_file) != FALSE;
  }
  if (!written) InterlockedExchange(&g_ngx_log_write_failed, 1);
  ReleaseSRWLockExclusive(&g_ngx_log_lock);
}

class NgxLogSink final {
 public:
  explicit NgxLogSink(const std::filesystem::path& path) : path_(path) {
    HANDLE file = CreateFileW(path_.c_str(), GENERIC_WRITE, FILE_SHARE_READ,
                              nullptr, CREATE_NEW, FILE_ATTRIBUTE_NORMAL,
                              nullptr);
    if (file == INVALID_HANDLE_VALUE) {
      throw std::runtime_error("creating NGX callback log failed");
    }
    AcquireSRWLockExclusive(&g_ngx_log_lock);
    if (g_ngx_log_file != INVALID_HANDLE_VALUE) {
      ReleaseSRWLockExclusive(&g_ngx_log_lock);
      CloseHandle(file);
      throw std::runtime_error("NGX callback log sink is already active");
    }
    g_ngx_log_file = file;
    InterlockedExchange(&g_ngx_log_write_failed, 0);
    ReleaseSRWLockExclusive(&g_ngx_log_lock);
  }

  NgxLogSink(const NgxLogSink&) = delete;
  NgxLogSink& operator=(const NgxLogSink&) = delete;

  void Close() {
    AcquireSRWLockExclusive(&g_ngx_log_lock);
    if (g_ngx_log_file != INVALID_HANDLE_VALUE) {
      (void)FlushFileBuffers(g_ngx_log_file);
      CloseHandle(g_ngx_log_file);
      g_ngx_log_file = INVALID_HANDLE_VALUE;
    }
    ReleaseSRWLockExclusive(&g_ngx_log_lock);
  }

  bool write_failed() const {
    return InterlockedCompareExchange(&g_ngx_log_write_failed, 0, 0) != 0;
  }

  const std::filesystem::path& path() const { return path_; }

  ~NgxLogSink() { Close(); }

 private:
  std::filesystem::path path_;
};

std::wstring GetArgument(int argc, wchar_t** argv, std::wstring_view name) {
  for (int index = 1; index + 1 < argc; ++index) {
    if (name == argv[index]) return argv[index + 1];
  }
  return {};
}

bool HasFlag(int argc, wchar_t** argv, std::wstring_view name) {
  for (int index = 1; index < argc; ++index) {
    if (name == argv[index]) return true;
  }
  return false;
}

std::uint32_t GetUintArgument(int argc, wchar_t** argv,
                              std::wstring_view name,
                              std::uint32_t fallback,
                              std::uint32_t minimum,
                              std::uint32_t maximum) {
  const std::wstring value = GetArgument(argc, argv, name);
  if (value.empty()) return fallback;
  std::size_t consumed = 0;
  const unsigned long parsed = std::stoul(value, &consumed, 10);
  if (consumed != value.size() || parsed < minimum || parsed > maximum) {
    throw std::invalid_argument("numeric command-line argument out of range");
  }
  return static_cast<std::uint32_t>(parsed);
}

std::uint64_t ParseUint64Argument(const std::wstring& value,
                                  std::uint64_t minimum,
                                  std::uint64_t maximum) {
  std::size_t consumed = 0;
  const unsigned long long parsed = std::stoull(value, &consumed, 10);
  if (consumed != value.size() || parsed < minimum || parsed > maximum) {
    throw std::invalid_argument("64-bit command-line argument out of range");
  }
  return static_cast<std::uint64_t>(parsed);
}

std::vector<std::uint64_t> ParseTokenList(const std::wstring& value) {
  if (value.empty()) throw std::invalid_argument("allocation token list is empty");
  std::vector<std::uint64_t> tokens;
  std::size_t begin = 0;
  while (begin < value.size()) {
    const std::size_t end = value.find(L',', begin);
    const std::wstring item = value.substr(
        begin, end == std::wstring::npos ? std::wstring::npos : end - begin);
    tokens.push_back(ParseUint64Argument(
        item, 1, (std::numeric_limits<std::uint64_t>::max)()));
    if (end == std::wstring::npos) break;
    begin = end + 1u;
  }
  std::vector<std::uint64_t> sorted = tokens;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    throw std::invalid_argument("allocation token list contains duplicates");
  }
  return tokens;
}

std::string NormalizeSha256Argument(const std::wstring& value) {
  const std::string utf8 = WideToUtf8(value);
  if (utf8.size() != 64u) {
    throw std::invalid_argument("SHA-256 argument must contain 64 hex digits");
  }
  std::string normalized;
  normalized.reserve(utf8.size());
  for (const unsigned char byte : utf8) {
    if (!std::isxdigit(byte)) {
      throw std::invalid_argument(
          "SHA-256 argument contains a non-hex character");
    }
    normalized.push_back(static_cast<char>(std::tolower(byte)));
  }
  return normalized;
}

void RequireHr(HRESULT result, std::string_view operation) {
  if (SUCCEEDED(result)) return;
  std::ostringstream message;
  message << operation << " failed hr=0x" << std::hex << std::uppercase
          << static_cast<unsigned long>(result);
  throw std::runtime_error(message.str());
}

void RequireNt(NTSTATUS result, std::string_view operation) {
  if (result >= 0) return;
  std::ostringstream message;
  message << operation << " failed status=0x" << std::hex << std::uppercase
          << static_cast<unsigned long>(result);
  throw std::runtime_error(message.str());
}

std::string NgxHex(NVSDK_NGX_Result result) {
  std::ostringstream text;
  text << "0x" << std::hex << std::uppercase
       << static_cast<unsigned int>(result);
  return text.str();
}

void PrintNgx(std::string_view name, NVSDK_NGX_Result result) {
  std::cout << name << '=' << NgxHex(result) << '\n' << std::flush;
}

template <typename T>
std::vector<std::uint8_t> AsBytes(const std::vector<T>& values) {
  const auto* begin = reinterpret_cast<const std::uint8_t*>(values.data());
  return std::vector<std::uint8_t>(
      begin, begin + values.size() * sizeof(typename std::vector<T>::value_type));
}

std::vector<std::uint8_t> ReadFile(const std::filesystem::path& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("opening file for hashing failed");
  const std::streamoff end = input.tellg();
  if (end < 0) throw std::runtime_error("reading file size failed");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(end));
  input.seekg(0, std::ios::beg);
  if (!bytes.empty()) {
    input.read(reinterpret_cast<char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  }
  if (!input) throw std::runtime_error("reading file for hashing failed");
  return bytes;
}

std::string Sha256(const void* data, std::size_t size) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  RequireNt(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                        nullptr, 0),
            "BCryptOpenAlgorithmProvider");
  struct AlgorithmCloser {
    BCRYPT_ALG_HANDLE value;
    ~AlgorithmCloser() {
      if (value != nullptr) BCryptCloseAlgorithmProvider(value, 0);
    }
  } algorithm_closer{algorithm};

  DWORD object_size = 0;
  DWORD hash_size = 0;
  DWORD copied = 0;
  RequireNt(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                              reinterpret_cast<PUCHAR>(&object_size),
                              sizeof(object_size), &copied, 0),
            "BCryptGetProperty(object length)");
  RequireNt(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                              reinterpret_cast<PUCHAR>(&hash_size),
                              sizeof(hash_size), &copied, 0),
            "BCryptGetProperty(hash length)");
  std::vector<std::uint8_t> object(object_size);
  std::vector<std::uint8_t> digest(hash_size);
  RequireNt(BCryptCreateHash(algorithm, &hash, object.data(), object_size,
                             nullptr, 0, 0),
            "BCryptCreateHash");
  struct HashCloser {
    BCRYPT_HASH_HANDLE value;
    ~HashCloser() {
      if (value != nullptr) BCryptDestroyHash(value);
    }
  } hash_closer{hash};

  const auto* bytes = static_cast<const std::uint8_t*>(data);
  std::size_t offset = 0;
  while (offset < size) {
    const ULONG chunk = static_cast<ULONG>(
        std::min<std::size_t>(size - offset,
                              std::numeric_limits<ULONG>::max()));
    RequireNt(BCryptHashData(hash, const_cast<PUCHAR>(bytes + offset), chunk,
                             0),
              "BCryptHashData");
    offset += chunk;
  }
  RequireNt(BCryptFinishHash(hash, digest.data(), hash_size, 0),
            "BCryptFinishHash");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : digest) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  return Sha256(bytes.data(), bytes.size());
}

std::string RandomNonce() {
  std::array<std::uint8_t, 16> bytes{};
  RequireNt(BCryptGenRandom(nullptr, bytes.data(),
                            static_cast<ULONG>(bytes.size()),
                            BCRYPT_USE_SYSTEM_PREFERRED_RNG),
            "BCryptGenRandom");
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (const std::uint8_t byte : bytes) {
    output << std::setw(2) << static_cast<unsigned int>(byte);
  }
  return output.str();
}

template <typename T>
std::string Sha256(const std::vector<T>& values) {
  return Sha256(values.data(), values.size() * sizeof(T));
}

void WriteBinary(const std::filesystem::path& path, const void* data,
                 std::size_t byte_count) {
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) throw std::runtime_error("opening artifact output failed");
  if (byte_count != 0) {
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(byte_count));
  }
  if (!output) throw std::runtime_error("writing artifact output failed");
}

template <typename T>
void WriteBinary(const std::filesystem::path& path,
                 const std::vector<T>& values) {
  WriteBinary(path, values.data(), values.size() * sizeof(T));
}

void WriteDirectoryInventory(const std::filesystem::path& output_path,
                             const std::filesystem::path& root,
                             std::string_view phase) {
  struct Entry {
    std::string path;
    std::uintmax_t bytes;
    std::string sha256;
  };
  std::vector<Entry> entries;
  if (std::filesystem::is_directory(root)) {
    for (const auto& item : std::filesystem::recursive_directory_iterator(root)) {
      if (!item.is_regular_file()) continue;
      const auto relative = std::filesystem::relative(item.path(), root);
      const auto content = ReadFile(item.path());
      entries.push_back({WideToUtf8(relative.native()), item.file_size(),
                         Sha256(content)});
    }
  }
  std::sort(entries.begin(), entries.end(),
            [](const Entry& left, const Entry& right) {
              return left.path < right.path;
            });
  std::ofstream output(output_path, std::ios::trunc);
  if (!output) throw std::runtime_error("opening directory inventory failed");
  output << "{\n  \"schema\": \"neuralstudio.directory_inventory.v1\",\n"
         << "  \"phase\": \"" << JsonEscape(phase) << "\",\n"
         << "  \"root\": \"" << JsonEscape(WideToUtf8(root.native()))
         << "\",\n  \"entries\": [\n";
  for (std::size_t index = 0; index < entries.size(); ++index) {
    const Entry& entry = entries[index];
    output << "    {\"path\": \"" << JsonEscape(entry.path)
           << "\", \"bytes\": " << entry.bytes << ", \"sha256\": \""
           << entry.sha256 << "\"}"
           << (index + 1u == entries.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  if (!output) throw std::runtime_error("writing directory inventory failed");
  std::cout << "inventory." << phase << ".files=" << entries.size() << '\n';
}

void WriteScenarioSnapshot(const std::filesystem::path& output_path,
                           std::string_view phase, std::uint32_t width,
                           std::uint32_t height, std::uint32_t rendered_frames,
                           std::uint32_t subrect_x, std::uint32_t subrect_y,
                           std::uint32_t subrect_width,
                           std::uint32_t subrect_height,
                           std::uint32_t multiframe_count,
                           std::uint32_t multiframe_index,
                           std::string_view resource_profile,
                           std::uint32_t output_poison_byte,
                           std::optional<bool> accepted) {
  std::ofstream output(output_path, std::ios::trunc);
  if (!output) throw std::runtime_error("opening scenario snapshot failed");
  output << "{\n"
         << "  \"schema\": \"neuralstudio.dlss_fg.synthetic_scenario.v1\",\n"
         << "  \"phase\": \"" << JsonEscape(phase) << "\",\n"
         << "  \"generator\": \"translated_tiles_ui_v1\",\n"
         << "  \"width\": " << width << ",\n"
         << "  \"height\": " << height << ",\n"
         << "  \"resource_width\": " << width << ",\n"
         << "  \"resource_height\": " << height << ",\n"
         << "  \"rendered_frames\": " << rendered_frames << ",\n"
         << "  \"subrect\": {\"x\": " << subrect_x
         << ", \"y\": " << subrect_y << ", \"width\": "
         << subrect_width << ", \"height\": " << subrect_height << "},\n"
         << "  \"legacy_backbuffer_subrect_mirrored\": true,\n"
         << "  \"dynamic_resolution_scaling\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? "true"
                 : "false")
         << ",\n"
         << "  \"feature_create_width\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? subrect_width
                 : width)
         << ",\n"
         << "  \"feature_create_height\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? subrect_height
                 : height)
         << ",\n"
         << "  \"multiframe_count\": " << multiframe_count << ",\n"
         << "  \"multiframe_index\": " << multiframe_index << ",\n"
         << "  \"resource_profile\": \""
         << JsonEscape(resource_profile) << "\",\n"
         << "  \"output_poison_byte\": " << output_poison_byte << ",\n"
         << "  \"poison_invariance_verified_in_this_process\": false,\n"
         << "  \"poison_invariance_requires_distinct_process\": true,\n"
         << "  \"accepted\": ";
  if (accepted) {
    output << (*accepted ? "true" : "false");
  } else {
    output << "null";
  }
  output << "\n}\n";
  if (!output) throw std::runtime_error("writing scenario snapshot failed");
}

std::filesystem::path LoadedModulePath(HMODULE module);

void WriteLoadedModuleInventory(const std::filesystem::path& output_path) {
  const std::array<const wchar_t*, 6> module_names{{
      L"nvngx_dlssg.dll", L"_nvngx.dll", L"nvapi64.dll",
      L"nvofapi64.dll", L"nvldumdx.dll", L"nvwgf2umx.dll",
  }};
  std::ofstream output(output_path, std::ios::trunc);
  if (!output) throw std::runtime_error("opening module inventory failed");
  output << "{\n  \"schema\": \"neuralstudio.loaded_modules.v1\",\n"
         << "  \"modules\": [\n";
  for (std::size_t index = 0; index < module_names.size(); ++index) {
    const std::wstring name = module_names[index];
    HMODULE module = GetModuleHandleW(name.c_str());
    output << "    {\"name\": \"" << JsonEscape(WideToUtf8(name))
           << "\", \"loaded\": " << (module != nullptr ? "true" : "false");
    if (module != nullptr) {
      const std::filesystem::path path = LoadedModulePath(module);
      const std::vector<std::uint8_t> bytes = ReadFile(path);
      const std::string sha256 = Sha256(bytes);
      output << ", \"path\": \"" << JsonEscape(WideToUtf8(path.native()))
             << "\", \"bytes\": " << bytes.size()
             << ", \"sha256\": \"" << sha256 << "\"";
      std::cout << "module." << WideToUtf8(name) << ".sha256=" << sha256
                << '\n';
    } else {
      output << ", \"availability\": \"not_loaded_or_not_exposed\"";
      std::cout << "module." << WideToUtf8(name)
                << ".availability=not_loaded_or_not_exposed\n";
    }
    output << "}" << (index + 1u == module_names.size() ? "\n" : ",\n");
  }
  output << "  ]\n}\n";
  if (!output) throw std::runtime_error("writing module inventory failed");
}

template <typename Function>
Function LoadFunction(HMODULE module, const char* name) {
  const FARPROC raw = GetProcAddress(module, name);
  Function function = nullptr;
  static_assert(sizeof(function) == sizeof(raw));
  std::memcpy(&function, &raw, sizeof(function));
  return function;
}

std::filesystem::path LoadedModulePath(HMODULE module) {
  std::wstring storage(32768u, L'\0');
  const DWORD length = GetModuleFileNameW(module, storage.data(),
                                          static_cast<DWORD>(storage.size()));
  if (length == 0 || length >= storage.size()) {
    throw std::runtime_error("GetModuleFileNameW failed");
  }
  storage.resize(length);
  return std::filesystem::weakly_canonical(storage);
}

struct GpuContext {
  std::wstring adapter_name;
  DXGI_ADAPTER_DESC1 adapter_desc{};
  ComPtr<IDXGIAdapter1> adapter;
  ComPtr<ID3D12Device> device;
  ComPtr<ID3D12CommandQueue> queue;
  ComPtr<ID3D12CommandAllocator> allocator;
  ComPtr<ID3D12GraphicsCommandList> list;
  ComPtr<ID3D12Fence> fence;
  UniqueHandle event;
  std::uint64_t next_fence = 1;
  std::vector<ComPtr<ID3D12Resource>> pending_uploads;
  bool debug_layer_enabled = false;
  std::uint64_t max_observed_local_video_memory = 0;
};

void SampleLocalVideoMemory(GpuContext& gpu, std::string_view phase) {
  ComPtr<IDXGIAdapter3> adapter3;
  if (FAILED(gpu.adapter.As(&adapter3))) {
    std::cout << "memory." << phase << ".availability=unavailable\n";
    return;
  }
  DXGI_QUERY_VIDEO_MEMORY_INFO info{};
  if (FAILED(adapter3->QueryVideoMemoryInfo(
          0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &info))) {
    std::cout << "memory." << phase << ".availability=unavailable\n";
    return;
  }
  gpu.max_observed_local_video_memory =
      std::max<std::uint64_t>(gpu.max_observed_local_video_memory,
                              info.CurrentUsage);
  std::cout << "memory." << phase << ".current_usage_bytes="
            << info.CurrentUsage << '\n';
  std::cout << "memory." << phase << ".budget_bytes=" << info.Budget << '\n';
}

GpuContext CreateGpuContext() {
  GpuContext gpu;
  ComPtr<ID3D12Debug> debug;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)))) {
    debug->EnableDebugLayer();
    gpu.debug_layer_enabled = true;
  }
  ComPtr<ID3D12DeviceRemovedExtendedDataSettings1> dred;
  if (SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
    dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
    dred->SetBreadcrumbContextEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
  }

  ComPtr<IDXGIFactory6> factory;
  RequireHr(CreateDXGIFactory2(0, IID_PPV_ARGS(&factory)),
            "CreateDXGIFactory2");
  for (UINT index = 0;; ++index) {
    ComPtr<IDXGIAdapter1> candidate;
    const HRESULT result = factory->EnumAdapterByGpuPreference(
        index, DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,
        IID_PPV_ARGS(&candidate));
    if (result == DXGI_ERROR_NOT_FOUND) break;
    RequireHr(result, "EnumAdapterByGpuPreference");
    DXGI_ADAPTER_DESC1 description{};
    RequireHr(candidate->GetDesc1(&description), "IDXGIAdapter1::GetDesc1");
    if ((description.Flags & DXGI_ADAPTER_FLAG_SOFTWARE) != 0 ||
        description.VendorId != 0x10deu) {
      continue;
    }
    ComPtr<ID3D12Device> device;
    if (SUCCEEDED(D3D12CreateDevice(candidate.Get(), D3D_FEATURE_LEVEL_12_0,
                                    IID_PPV_ARGS(&device)))) {
      gpu.adapter_name = description.Description;
      gpu.adapter_desc = description;
      gpu.adapter = std::move(candidate);
      gpu.device = std::move(device);
      break;
    }
  }
  if (gpu.device == nullptr) {
    throw std::runtime_error("no NVIDIA D3D12 feature-level-12 adapter found");
  }

  D3D12_COMMAND_QUEUE_DESC queue_description{};
  queue_description.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  RequireHr(gpu.device->CreateCommandQueue(&queue_description,
                                            IID_PPV_ARGS(&gpu.queue)),
            "CreateCommandQueue");
  RequireHr(gpu.device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT,
                                                IID_PPV_ARGS(&gpu.allocator)),
            "CreateCommandAllocator");
  RequireHr(gpu.device->CreateCommandList(
                0, D3D12_COMMAND_LIST_TYPE_DIRECT, gpu.allocator.Get(), nullptr,
                IID_PPV_ARGS(&gpu.list)),
            "CreateCommandList");
  RequireHr(gpu.device->CreateFence(0, D3D12_FENCE_FLAG_NONE,
                                     IID_PPV_ARGS(&gpu.fence)),
            "CreateFence");
  gpu.event.reset(CreateEventW(nullptr, FALSE, FALSE, nullptr));
  if (!gpu.event) throw std::runtime_error("CreateEventW failed");
  return gpu;
}

void ExecuteAndReset(GpuContext& gpu) {
  RequireHr(gpu.list->Close(), "ID3D12GraphicsCommandList::Close");
  ID3D12CommandList* lists[] = {gpu.list.Get()};
  gpu.queue->ExecuteCommandLists(1, lists);
  const std::uint64_t fence_value = gpu.next_fence++;
  RequireHr(gpu.queue->Signal(gpu.fence.Get(), fence_value),
            "ID3D12CommandQueue::Signal");
  if (gpu.fence->GetCompletedValue() < fence_value) {
    RequireHr(gpu.fence->SetEventOnCompletion(fence_value, gpu.event.get()),
              "ID3D12Fence::SetEventOnCompletion");
    if (WaitForSingleObject(gpu.event.get(), 120000) != WAIT_OBJECT_0) {
      throw std::runtime_error("D3D12 fence wait timed out");
    }
  }
  gpu.pending_uploads.clear();
  RequireHr(gpu.allocator->Reset(), "ID3D12CommandAllocator::Reset");
  RequireHr(gpu.list->Reset(gpu.allocator.Get(), nullptr),
            "ID3D12GraphicsCommandList::Reset");
}

ComPtr<ID3D12Resource> CreateBuffer(ID3D12Device* device, std::uint64_t size,
                                    D3D12_HEAP_TYPE heap_type,
                                    D3D12_RESOURCE_FLAGS flags,
                                    D3D12_RESOURCE_STATES initial_state) {
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = heap_type;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
  description.Width = size;
  description.Height = 1;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
  description.Flags = flags;
  ComPtr<ID3D12Resource> resource;
  RequireHr(device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description, initial_state,
                nullptr, IID_PPV_ARGS(&resource)),
            "CreateCommittedResource(buffer)");
  return resource;
}

struct GpuTexture {
  ComPtr<ID3D12Resource> resource;
  std::uint32_t width = 0;
  std::uint32_t height = 0;
  std::uint32_t bytes_per_pixel = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

struct GpuBuffer {
  ComPtr<ID3D12Resource> resource;
  std::uint64_t size = 0;
  D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
};

void Transition(ID3D12GraphicsCommandList* list, ID3D12Resource* resource,
                D3D12_RESOURCE_STATES before,
                D3D12_RESOURCE_STATES after) {
  if (before == after) return;
  D3D12_RESOURCE_BARRIER barrier{};
  barrier.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
  barrier.Transition.pResource = resource;
  barrier.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
  barrier.Transition.StateBefore = before;
  barrier.Transition.StateAfter = after;
  list->ResourceBarrier(1, &barrier);
}

GpuTexture CreateUploadedTexture(GpuContext& gpu, DXGI_FORMAT format,
                                 std::uint32_t width, std::uint32_t height,
                                 std::uint32_t bytes_per_pixel,
                                 D3D12_RESOURCE_FLAGS flags,
                                 D3D12_RESOURCE_STATES final_state,
                                 const void* source,
                                 std::size_t source_row_bytes) {
  if (source == nullptr ||
      source_row_bytes != static_cast<std::size_t>(width) * bytes_per_pixel) {
    throw std::invalid_argument("invalid uploaded texture source");
  }
  D3D12_HEAP_PROPERTIES heap{};
  heap.Type = D3D12_HEAP_TYPE_DEFAULT;
  D3D12_RESOURCE_DESC description{};
  description.Dimension = D3D12_RESOURCE_DIMENSION_TEXTURE2D;
  description.Width = width;
  description.Height = height;
  description.DepthOrArraySize = 1;
  description.MipLevels = 1;
  description.Format = format;
  description.SampleDesc.Count = 1;
  description.Layout = D3D12_TEXTURE_LAYOUT_UNKNOWN;
  description.Flags = flags;
  GpuTexture texture;
  texture.width = width;
  texture.height = height;
  texture.bytes_per_pixel = bytes_per_pixel;
  texture.state = final_state;
  RequireHr(gpu.device->CreateCommittedResource(
                &heap, D3D12_HEAP_FLAG_NONE, &description,
                D3D12_RESOURCE_STATE_COPY_DEST, nullptr,
                IID_PPV_ARGS(&texture.resource)),
            "CreateCommittedResource(texture)");

  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 upload_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &upload_size);
  if (rows != height || row_size < source_row_bytes) {
    throw std::runtime_error("unexpected texture upload footprint");
  }
  ComPtr<ID3D12Resource> upload = CreateBuffer(
      gpu.device.Get(), upload_size, D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(texture upload)");
  for (std::uint32_t row = 0; row < height; ++row) {
    std::memcpy(static_cast<std::uint8_t*>(mapped) +
                    static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
                static_cast<const std::uint8_t*>(source) +
                    static_cast<std::size_t>(row) * source_row_bytes,
                source_row_bytes);
  }
  upload->Unmap(0, nullptr);
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = texture.resource.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION upload_source{};
  upload_source.pResource = upload.Get();
  upload_source.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  upload_source.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &upload_source, nullptr);
  Transition(gpu.list.Get(), texture.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, final_state);
  gpu.pending_uploads.push_back(std::move(upload));
  return texture;
}

GpuBuffer CreateUploadedBuffer(GpuContext& gpu,
                               D3D12_RESOURCE_FLAGS flags,
                               D3D12_RESOURCE_STATES final_state,
                               const void* source, std::size_t byte_count) {
  if (source == nullptr || byte_count == 0) {
    throw std::invalid_argument("invalid uploaded buffer source");
  }
  GpuBuffer buffer;
  buffer.size = byte_count;
  buffer.state = final_state;
  buffer.resource = CreateBuffer(gpu.device.Get(), byte_count,
                                 D3D12_HEAP_TYPE_DEFAULT, flags,
                                 D3D12_RESOURCE_STATE_COPY_DEST);
  ComPtr<ID3D12Resource> upload = CreateBuffer(
      gpu.device.Get(), byte_count, D3D12_HEAP_TYPE_UPLOAD,
      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_GENERIC_READ);
  void* mapped = nullptr;
  const D3D12_RANGE no_read{0, 0};
  RequireHr(upload->Map(0, &no_read, &mapped), "Map(buffer upload)");
  std::memcpy(mapped, source, byte_count);
  upload->Unmap(0, nullptr);
  gpu.list->CopyBufferRegion(buffer.resource.Get(), 0, upload.Get(), 0,
                             byte_count);
  Transition(gpu.list.Get(), buffer.resource.Get(),
             D3D12_RESOURCE_STATE_COPY_DEST, final_state);
  gpu.pending_uploads.push_back(std::move(upload));
  return buffer;
}

std::vector<std::uint8_t> ReadTexture(GpuContext& gpu, GpuTexture& texture) {
  const D3D12_RESOURCE_DESC description = texture.resource->GetDesc();
  D3D12_PLACED_SUBRESOURCE_FOOTPRINT footprint{};
  UINT rows = 0;
  UINT64 row_size = 0;
  UINT64 readback_size = 0;
  gpu.device->GetCopyableFootprints(&description, 0, 1, 0, &footprint, &rows,
                                    &row_size, &readback_size);
  const std::size_t output_row_bytes =
      static_cast<std::size_t>(texture.width) * texture.bytes_per_pixel;
  if (rows != texture.height || row_size < output_row_bytes) {
    throw std::runtime_error("unexpected texture readback footprint");
  }
  ComPtr<ID3D12Resource> readback = CreateBuffer(
      gpu.device.Get(), readback_size, D3D12_HEAP_TYPE_READBACK,
      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), texture.resource.Get(), texture.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  texture.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  D3D12_TEXTURE_COPY_LOCATION source{};
  source.pResource = texture.resource.Get();
  source.Type = D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
  D3D12_TEXTURE_COPY_LOCATION destination{};
  destination.pResource = readback.Get();
  destination.Type = D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;
  destination.PlacedFootprint = footprint;
  gpu.list->CopyTextureRegion(&destination, 0, 0, 0, &source, nullptr);
  ExecuteAndReset(gpu);

  std::vector<std::uint8_t> output(output_row_bytes * texture.height);
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(readback_size)};
  RequireHr(readback->Map(0, &read_range, &mapped), "Map(texture readback)");
  for (std::uint32_t row = 0; row < texture.height; ++row) {
    std::memcpy(output.data() + static_cast<std::size_t>(row) * output_row_bytes,
                static_cast<const std::uint8_t*>(mapped) +
                    static_cast<std::size_t>(row) * footprint.Footprint.RowPitch,
                output_row_bytes);
  }
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return output;
}

std::vector<std::uint8_t> ReadBuffer(GpuContext& gpu, GpuBuffer& buffer) {
  ComPtr<ID3D12Resource> readback = CreateBuffer(
      gpu.device.Get(), buffer.size, D3D12_HEAP_TYPE_READBACK,
      D3D12_RESOURCE_FLAG_NONE, D3D12_RESOURCE_STATE_COPY_DEST);
  Transition(gpu.list.Get(), buffer.resource.Get(), buffer.state,
             D3D12_RESOURCE_STATE_COPY_SOURCE);
  buffer.state = D3D12_RESOURCE_STATE_COPY_SOURCE;
  gpu.list->CopyBufferRegion(readback.Get(), 0, buffer.resource.Get(), 0,
                             buffer.size);
  ExecuteAndReset(gpu);
  std::vector<std::uint8_t> output(static_cast<std::size_t>(buffer.size));
  void* mapped = nullptr;
  const D3D12_RANGE read_range{0, static_cast<SIZE_T>(buffer.size)};
  RequireHr(readback->Map(0, &read_range, &mapped), "Map(buffer readback)");
  std::memcpy(output.data(), mapped, output.size());
  const D3D12_RANGE no_write{0, 0};
  readback->Unmap(0, &no_write);
  return output;
}

std::uint64_t CountSevereDebugMessages(ID3D12Device* device) {
  ComPtr<ID3D12InfoQueue> info;
  if (FAILED(device->QueryInterface(IID_PPV_ARGS(&info)))) {
    std::cout << "d3d12.debug_queue=unavailable\n";
    return 0;
  }
  std::uint64_t severe = 0;
  const UINT64 count = info->GetNumStoredMessagesAllowedByRetrievalFilter();
  for (UINT64 index = 0; index < count; ++index) {
    SIZE_T size = 0;
    if (FAILED(info->GetMessage(index, nullptr, &size)) || size == 0) continue;
    std::vector<std::uint8_t> storage(size);
    auto* message = reinterpret_cast<D3D12_MESSAGE*>(storage.data());
    if (FAILED(info->GetMessage(index, message, &size))) continue;
    if (message->Severity == D3D12_MESSAGE_SEVERITY_CORRUPTION ||
        message->Severity == D3D12_MESSAGE_SEVERITY_ERROR) {
      ++severe;
      std::cerr << "d3d12.severe id=" << static_cast<unsigned int>(message->ID)
                << " description=\""
                << (message->pDescription != nullptr ? message->pDescription
                                                     : "")
                << "\"\n";
    }
  }
  std::cout << "d3d12.debug_stored_messages=" << count << '\n';
  std::cout << "d3d12.severe_messages=" << severe << '\n';
  return severe;
}

std::string ReadDriverVersion(IDXGIAdapter1* adapter) {
  LARGE_INTEGER version{};
  if (FAILED(adapter->CheckInterfaceSupport(__uuidof(IDXGIDevice), &version))) {
    return "unavailable";
  }
  const auto high = static_cast<std::uint32_t>(version.HighPart);
  const auto low = static_cast<std::uint32_t>(version.LowPart);
  std::ostringstream output;
  output << HIWORD(high) << '.' << LOWORD(high) << '.' << HIWORD(low) << '.'
         << LOWORD(low);
  return output.str();
}

using GetRuntimeU32 = std::uint32_t(NVSDK_CONV*)();
using Populate = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter*);
using Init = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using Create = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
    const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle*);
using Shutdown = NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device*);
using Requirements = NVSDK_NGX_Result(NVSDK_CONV*)(
    IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*,
    NVSDK_NGX_FeatureRequirement*);

using ThunkPopulate = NVSDK_NGX_Result(NVSDK_CONV*)(
    Populate, NVSDK_NGX_Parameter*);
using ThunkInit = NVSDK_NGX_Result(NVSDK_CONV*)(
    Init, unsigned long long, const wchar_t*, ID3D12Device*, NVSDK_NGX_Version,
    const NVSDK_NGX_Parameter*);
using ThunkCreate = NVSDK_NGX_Result(NVSDK_CONV*)(
    Create, ID3D12GraphicsCommandList*, NVSDK_NGX_Feature,
    const NVSDK_NGX_Parameter*, NVSDK_NGX_Handle**);
using ThunkEvaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
    Evaluate, ID3D12GraphicsCommandList*, const NVSDK_NGX_Handle*,
    const NVSDK_NGX_Parameter*, PFN_NVSDK_NGX_ProgressCallback);
using ThunkRelease =
    NVSDK_NGX_Result(NVSDK_CONV*)(Release, NVSDK_NGX_Handle*);
using ThunkShutdown =
    NVSDK_NGX_Result(NVSDK_CONV*)(Shutdown, ID3D12Device*);
using ThunkRequirements = NVSDK_NGX_Result(NVSDK_CONV*)(
    Requirements, IDXGIAdapter*, const NVSDK_NGX_FeatureDiscoveryInfo*,
    NVSDK_NGX_FeatureRequirement*);
using FgNvapiSetPhase = void(WINAPI*)(const char*);
using FgNvapiAttachD3D12Device = BOOL(WINAPI*)(void*);
using FgNvapiDetachD3D12Device = BOOL(WINAPI*)();

void WINAPI NoopNvapiSetPhase(const char*) {}

struct ScopedFgDescriptorTraceAttachment {
  FgNvapiDetachD3D12Device detach = nullptr;
  bool active = false;

  ~ScopedFgDescriptorTraceAttachment() {
    if (active && detach != nullptr && !detach()) {
      OutputDebugStringA("FG descriptor tracer detach reported failure.\n");
    }
  }
};

struct RuntimeApi {
  GetRuntimeU32 get_application_id = nullptr;
  GetRuntimeU32 get_api_version = nullptr;
  GetRuntimeU32 get_snippet_version = nullptr;
  Populate populate = nullptr;
  Init init = nullptr;
  Create create = nullptr;
  Evaluate evaluate = nullptr;
  Release release = nullptr;
  Shutdown shutdown = nullptr;
  Requirements requirements = nullptr;
};

struct ThunkApi {
  ThunkPopulate populate = nullptr;
  ThunkInit init = nullptr;
  ThunkCreate create = nullptr;
  ThunkEvaluate evaluate = nullptr;
  ThunkRelease release = nullptr;
  ThunkShutdown shutdown = nullptr;
  ThunkRequirements requirements = nullptr;
};

RuntimeApi LoadRuntimeApi(HMODULE runtime) {
  RuntimeApi api;
  api.get_application_id =
      LoadFunction<GetRuntimeU32>(runtime, "NVSDK_NGX_GetApplicationId");
  api.get_api_version =
      LoadFunction<GetRuntimeU32>(runtime, "NVSDK_NGX_GetAPIVersion");
  api.get_snippet_version =
      LoadFunction<GetRuntimeU32>(runtime, "NVSDK_NGX_GetSnippetVersion");
  api.populate = LoadFunction<Populate>(
      runtime, "NVSDK_NGX_D3D12_PopulateParameters_Impl");
  api.init = LoadFunction<Init>(runtime, "NVSDK_NGX_D3D12_Init_Ext");
  api.create =
      LoadFunction<Create>(runtime, "NVSDK_NGX_D3D12_CreateFeature");
  api.evaluate =
      LoadFunction<Evaluate>(runtime, "NVSDK_NGX_D3D12_EvaluateFeature");
  api.release =
      LoadFunction<Release>(runtime, "NVSDK_NGX_D3D12_ReleaseFeature");
  api.shutdown =
      LoadFunction<Shutdown>(runtime, "NVSDK_NGX_D3D12_Shutdown1");
  api.requirements = LoadFunction<Requirements>(
      runtime, "NVSDK_NGX_D3D12_GetFeatureRequirements");
  const std::array<std::pair<const char*, bool>, 10> required{{
      {"NVSDK_NGX_GetApplicationId", api.get_application_id != nullptr},
      {"NVSDK_NGX_GetAPIVersion", api.get_api_version != nullptr},
      {"NVSDK_NGX_GetSnippetVersion", api.get_snippet_version != nullptr},
      {"NVSDK_NGX_D3D12_PopulateParameters_Impl", api.populate != nullptr},
      {"NVSDK_NGX_D3D12_Init_Ext", api.init != nullptr},
      {"NVSDK_NGX_D3D12_CreateFeature", api.create != nullptr},
      {"NVSDK_NGX_D3D12_EvaluateFeature", api.evaluate != nullptr},
      {"NVSDK_NGX_D3D12_ReleaseFeature", api.release != nullptr},
      {"NVSDK_NGX_D3D12_Shutdown1", api.shutdown != nullptr},
      {"NVSDK_NGX_D3D12_GetFeatureRequirements",
       api.requirements != nullptr},
  }};
  for (const auto& [name, present] : required) {
    if (!present) throw std::runtime_error(std::string("missing export: ") + name);
  }
  return api;
}

ThunkApi LoadThunkApi(HMODULE thunk) {
  ThunkApi api;
  api.populate =
      LoadFunction<ThunkPopulate>(thunk, "NeuralStudioPopulateD3D12");
  api.init = LoadFunction<ThunkInit>(thunk, "NeuralStudioInitD3D12");
  api.create = LoadFunction<ThunkCreate>(thunk, "NeuralStudioCreateD3D12");
  api.evaluate =
      LoadFunction<ThunkEvaluate>(thunk, "NeuralStudioEvaluateD3D12");
  api.release =
      LoadFunction<ThunkRelease>(thunk, "NeuralStudioReleaseD3D12");
  api.shutdown =
      LoadFunction<ThunkShutdown>(thunk, "NeuralStudioShutdownD3D12");
  api.requirements = LoadFunction<ThunkRequirements>(
      thunk, "NeuralStudioRequirementsD3D12");
  if (api.populate == nullptr || api.init == nullptr || api.create == nullptr ||
      api.evaluate == nullptr || api.release == nullptr ||
      api.shutdown == nullptr || api.requirements == nullptr) {
    throw std::runtime_error("typed D3D12 NGX thunk export is missing");
  }
  return api;
}

void SetIdentity(float matrix[4][4]) {
  std::memset(matrix, 0, sizeof(float) * 16u);
  for (std::size_t index = 0; index < 4u; ++index) matrix[index][index] = 1.0f;
}

std::optional<long long> ReadIntegral(
    const neuralstudio::fg::ParameterStore& parameters, const char* name) {
  int signed_value = 0;
  if (NVSDK_NGX_SUCCEED(parameters.GetI32(name, &signed_value))) {
    return signed_value;
  }
  unsigned int unsigned_value = 0;
  if (NVSDK_NGX_SUCCEED(parameters.GetU32(name, &unsigned_value))) {
    return static_cast<long long>(unsigned_value);
  }
  unsigned long long wide_value = 0;
  if (NVSDK_NGX_SUCCEED(parameters.GetU64(name, &wide_value)) &&
      wide_value <= static_cast<unsigned long long>(
                        std::numeric_limits<long long>::max())) {
    return static_cast<long long>(wide_value);
  }
  return std::nullopt;
}

struct RuntimeCleanup {
  RuntimeApi* runtime = nullptr;
  ThunkApi* thunk = nullptr;
  ID3D12Device* device = nullptr;
  FgNvapiSetPhase set_trace_phase = &NoopNvapiSetPhase;
  NVSDK_NGX_Handle* handle = nullptr;
  bool initialized = false;
  bool finished = false;
  NVSDK_NGX_Result release_result = NVSDK_NGX_Result_FAIL_NotInitialized;
  NVSDK_NGX_Result shutdown_result = NVSDK_NGX_Result_FAIL_NotInitialized;

  void Finish() {
    if (finished) return;
    if (handle != nullptr) {
      set_trace_phase("release");
      release_result = thunk->release(runtime->release, handle);
      set_trace_phase("release.end");
      PrintNgx("runtime.release_result", release_result);
      handle = nullptr;
    } else {
      release_result = NVSDK_NGX_Result_Success;
    }
    if (initialized) {
      set_trace_phase("shutdown");
      shutdown_result = thunk->shutdown(runtime->shutdown, device);
      set_trace_phase("shutdown.end");
      PrintNgx("runtime.shutdown_result", shutdown_result);
      initialized = false;
    } else {
      shutdown_result = NVSDK_NGX_Result_Success;
    }
    finished = true;
  }

  ~RuntimeCleanup() {
    if (finished || runtime == nullptr || thunk == nullptr) return;
    if (handle != nullptr) {
      set_trace_phase("release");
      (void)thunk->release(runtime->release, handle);
      set_trace_phase("release.end");
    }
    if (initialized && device != nullptr) {
      set_trace_phase("shutdown");
      (void)thunk->shutdown(runtime->shutdown, device);
      set_trace_phase("shutdown.end");
    }
  }
};

struct FrameReceipt {
  std::uint32_t frame_id = 0;
  bool reset = false;
  NVSDK_NGX_Result evaluate_result = NVSDK_NGX_Result_FAIL_NotInitialized;
  double evaluate_ms = 0.0;
  std::string input_sha256;
  std::string output_interpolated_sha256;
  std::string output_real_sha256;
  std::string disable_sha256;
  bool output_real_provided = false;
  bool disable_output_provided = false;
  bool disable_interpolation = true;
  std::size_t output_nonzero_bytes = 0;
  std::size_t output_untouched_pixels = 0;
  bool differs_from_previous = false;
  bool differs_from_current = false;
  bool generated_output_accepted = false;
};

struct AllocationCaptureReceipt {
  bool enabled = false;
  bool healthy = true;
  bool baseline_output_match = true;
  std::size_t allocations = 0;
  std::size_t releases = 0;
  std::size_t leaks = 0;
  std::string expected_baseline_sha256 = "not_requested";
  std::string semantic_ledger_sha256 = "not_requested";
  std::string volatile_ledger_sha256 = "not_requested";
  bool upload_capture_enabled = false;
  std::size_t upload_dump_count = 0;
  std::uint64_t upload_dump_bytes = 0;
  std::size_t expected_upload_dump_count = 0;
  std::uint64_t expected_upload_dump_bytes = 0;
  std::string upload_manifest_sha256 = "not_requested";
  bool default_capture_enabled = false;
  std::size_t default_dump_count = 0;
  std::uint64_t default_dump_bytes = 0;
  std::size_t expected_default_dump_count = 0;
  std::uint64_t expected_default_dump_bytes = 0;
  std::string default_manifest_sha256 = "not_requested";
  std::string default_capture_point = "not_requested";
  std::string default_manifest = "not_requested";
};

bool RegionDiffers(const std::vector<std::uint8_t>& left,
                   const std::vector<std::uint8_t>& right,
                   std::uint32_t full_width, std::uint32_t x,
                   std::uint32_t y, std::uint32_t width,
                   std::uint32_t height) {
  if (left.size() != right.size()) return true;
  const std::size_t row_bytes = static_cast<std::size_t>(width) * 4u;
  for (std::uint32_t row = 0; row < height; ++row) {
    const std::size_t offset =
        (static_cast<std::size_t>(y + row) * full_width + x) * 4u;
    if (!std::equal(left.begin() + static_cast<std::ptrdiff_t>(offset),
                    left.begin() +
                        static_cast<std::ptrdiff_t>(offset + row_bytes),
                    right.begin() + static_cast<std::ptrdiff_t>(offset))) {
      return true;
    }
  }
  return false;
}

bool RegionContainsNonzero(const std::vector<std::uint8_t>& bytes,
                           std::uint32_t full_width, std::uint32_t x,
                           std::uint32_t y, std::uint32_t width,
                           std::uint32_t height) {
  for (std::uint32_t row = 0; row < height; ++row) {
    const std::size_t offset =
        (static_cast<std::size_t>(y + row) * full_width + x) * 4u;
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 4u;
    if (std::any_of(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                    bytes.begin() +
                        static_cast<std::ptrdiff_t>(offset + row_bytes),
                    [](std::uint8_t byte) { return byte != 0u; })) {
      return true;
    }
  }
  return false;
}

bool RegionAllValue(const std::vector<std::uint8_t>& bytes,
                    std::uint8_t value, std::uint32_t full_width,
                    std::uint32_t x, std::uint32_t y, std::uint32_t width,
                    std::uint32_t height) {
  for (std::uint32_t row = 0; row < height; ++row) {
    const std::size_t offset =
        (static_cast<std::size_t>(y + row) * full_width + x) * 4u;
    const std::size_t row_bytes = static_cast<std::size_t>(width) * 4u;
    if (!std::all_of(bytes.begin() + static_cast<std::ptrdiff_t>(offset),
                     bytes.begin() +
                         static_cast<std::ptrdiff_t>(offset + row_bytes),
                     [value](std::uint8_t byte) { return byte == value; })) {
      return false;
    }
  }
  return true;
}

std::size_t CountPixelsAllValue(const std::vector<std::uint8_t>& bytes,
                                std::uint8_t value,
                                std::uint32_t full_width, std::uint32_t x,
                                std::uint32_t y, std::uint32_t width,
                                std::uint32_t height) {
  std::size_t count = 0;
  for (std::uint32_t row = 0; row < height; ++row) {
    for (std::uint32_t column = 0; column < width; ++column) {
      const std::size_t offset =
          (static_cast<std::size_t>(y + row) * full_width + x + column) * 4u;
      if (bytes[offset] == value && bytes[offset + 1u] == value &&
          bytes[offset + 2u] == value && bytes[offset + 3u] == value) {
        ++count;
      }
    }
  }
  return count;
}

void WriteReceipt(const std::filesystem::path& path,
                  const std::filesystem::path& runtime_path,
                  const std::filesystem::path& thunk_path,
                  std::string_view runtime_sha256,
                  std::string_view thunk_sha256,
                  std::string_view run_nonce,
                  std::string_view queue_nonce,
                  std::string_view driver_version,
                  std::string_view nvapi_proxy_path,
                  std::string_view nvapi_proxy_sha256,
                  std::string_view ngx_callback_log_sha256,
                  const GpuContext& gpu, std::uint32_t application_id,
                  std::uint32_t api_version, std::uint32_t snippet_version,
                  std::uint32_t width, std::uint32_t height,
                  std::uint32_t subrect_x, std::uint32_t subrect_y,
                  std::uint32_t subrect_width, std::uint32_t subrect_height,
                  std::uint32_t multiframe_count,
                  std::uint32_t multiframe_index,
                  std::string_view resource_profile,
                  std::uint32_t output_poison_byte,
                  std::uint64_t estimated_vram, NVSDK_NGX_Result create_result,
                  NVSDK_NGX_Result release_result,
                  NVSDK_NGX_Result shutdown_result,
                  const std::vector<FrameReceipt>& frames,
                  const AllocationCaptureReceipt& allocation_capture,
                  std::uint64_t severe_messages, bool accepted,
                  std::string_view input_sequence_sha256,
                  std::string_view motion_scale_contract) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("opening JSON receipt failed");
  output << "{\n"
         << "  \"schema\": \"neuralstudio.dlss_fg.standalone_d3d12.v1\",\n"
         << "  \"claim\": \"execution_receipt_only_no_parity_claim\",\n"
         << "  \"input_sequence_sha256\": \"" << input_sequence_sha256 << "\",\n"
         << "  \"motion_scale_contract\": \"" << motion_scale_contract << "\",\n"
         << "  \"external_input_readback_verified\": " << (input_sequence_sha256 == "builtin-synthetic" ? "false" : "true") << ",\n"
         << "  \"runtime_path\": \""
         << JsonEscape(WideToUtf8(runtime_path.native())) << "\",\n"
         << "  \"runtime_sha256\": \"" << runtime_sha256 << "\",\n"
         << "  \"thunk_path\": \""
         << JsonEscape(WideToUtf8(thunk_path.native())) << "\",\n"
         << "  \"thunk_sha256\": \"" << thunk_sha256 << "\",\n"
         << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
         << "  \"run_nonce\": \"" << run_nonce << "\",\n"
         << "  \"queue_nonce\": \"" << queue_nonce << "\",\n"
         << "  \"driver_version\": \"" << driver_version << "\",\n"
         << "  \"nvapi_proxy_path\": \""
         << JsonEscape(nvapi_proxy_path) << "\",\n"
         << "  \"nvapi_proxy_sha256\": \"" << nvapi_proxy_sha256
         << "\",\n"
         << "  \"ngx_callback_log\": \"ngx-callback.log\",\n"
         << "  \"ngx_callback_log_sha256\": \""
         << ngx_callback_log_sha256 << "\",\n"
         << "  \"application_id\": " << application_id << ",\n"
         << "  \"api_version\": " << api_version << ",\n"
         << "  \"snippet_version\": " << snippet_version << ",\n"
         << "  \"feature_id\": "
         << static_cast<unsigned int>(kFrameGenerationFeature) << ",\n"
         << "  \"adapter\": \""
         << JsonEscape(WideToUtf8(gpu.adapter_name)) << "\",\n"
         << "  \"adapter_vendor_id\": " << gpu.adapter_desc.VendorId
         << ",\n"
         << "  \"adapter_device_id\": " << gpu.adapter_desc.DeviceId
         << ",\n"
         << "  \"width\": " << width << ",\n"
         << "  \"height\": " << height << ",\n"
         << "  \"resource_width\": " << width << ",\n"
         << "  \"resource_height\": " << height << ",\n"
         << "  \"subrect\": {\"x\": " << subrect_x
         << ", \"y\": " << subrect_y << ", \"width\": "
         << subrect_width << ", \"height\": " << subrect_height << "},\n"
         << "  \"legacy_backbuffer_subrect_mirrored\": true,\n"
         << "  \"dynamic_resolution_scaling\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? "true"
                 : "false")
         << ",\n"
         << "  \"feature_create_width\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? subrect_width
                 : width)
         << ",\n"
         << "  \"feature_create_height\": "
         << ((subrect_x != 0u || subrect_y != 0u || subrect_width != width ||
              subrect_height != height)
                 ? subrect_height
                 : height)
         << ",\n"
         << "  \"multiframe_count\": " << multiframe_count << ",\n"
         << "  \"multiframe_index\": " << multiframe_index << ",\n"
         << "  \"scale\": \"" << (multiframe_count + 1u) << "x\",\n"
         << "  \"resource_profile\": \""
         << JsonEscape(resource_profile) << "\",\n"
         << "  \"output_poison_byte\": " << output_poison_byte << ",\n"
         << "  \"poison_invariance_verified_in_this_process\": false,\n"
         << "  \"poison_invariance_requires_distinct_process\": true,\n"
         << "  \"estimated_vram_bytes\": " << estimated_vram << ",\n"
         << "  \"create_result\": \"" << NgxHex(create_result) << "\",\n"
         << "  \"release_result\": \"" << NgxHex(release_result)
         << "\",\n"
         << "  \"shutdown_result\": \"" << NgxHex(shutdown_result)
         << "\",\n"
         << "  \"d3d12_severe_messages\": " << severe_messages << ",\n"
         << "  \"gpu_memory_metric\": \"max_observed_dxgi_process_sample\",\n"
         << "  \"gpu_memory_max_observed_bytes\": "
         << gpu.max_observed_local_video_memory << ",\n"
         << "  \"runtime_allocation_capture\": {\n"
         << "    \"enabled\": "
         << (allocation_capture.enabled ? "true" : "false") << ",\n"
         << "    \"healthy\": "
         << (allocation_capture.healthy ? "true" : "false") << ",\n"
         << "    \"expected_baseline_output_sha256\": \""
         << allocation_capture.expected_baseline_sha256 << "\",\n"
         << "    \"baseline_output_match\": "
         << (allocation_capture.baseline_output_match ? "true" : "false")
         << ",\n"
         << "    \"allocations\": " << allocation_capture.allocations
         << ",\n"
         << "    \"releases\": " << allocation_capture.releases << ",\n"
         << "    \"leaks\": " << allocation_capture.leaks << ",\n"
         << "    \"semantic_ledger\": \"runtime-allocations-semantic.jsonl\",\n"
         << "    \"semantic_ledger_sha256\": \""
         << allocation_capture.semantic_ledger_sha256 << "\",\n"
         << "    \"volatile_ledger\": \"runtime-allocations-volatile.jsonl\",\n"
         << "    \"volatile_ledger_sha256\": \""
         << allocation_capture.volatile_ledger_sha256 << "\",\n"
         << "    \"upload_capture_enabled\": "
         << (allocation_capture.upload_capture_enabled ? "true" : "false")
         << ",\n"
         << "    \"upload_dump_count\": "
         << allocation_capture.upload_dump_count << ",\n"
         << "    \"upload_dump_bytes\": "
         << allocation_capture.upload_dump_bytes << ",\n"
         << "    \"expected_upload_dump_count\": "
         << allocation_capture.expected_upload_dump_count << ",\n"
         << "    \"expected_upload_dump_bytes\": "
         << allocation_capture.expected_upload_dump_bytes << ",\n"
         << "    \"upload_manifest\": \"runtime-upload-buffers-manifest.json\",\n"
         << "    \"upload_manifest_sha256\": \""
         << allocation_capture.upload_manifest_sha256 << "\",\n"
         << "    \"default_capture_enabled\": "
         << (allocation_capture.default_capture_enabled ? "true" : "false")
         << ",\n"
         << "    \"default_capture_point\": \""
         << allocation_capture.default_capture_point << "\",\n"
         << "    \"default_dump_count\": "
         << allocation_capture.default_dump_count << ",\n"
         << "    \"default_dump_bytes\": "
         << allocation_capture.default_dump_bytes << ",\n"
         << "    \"expected_default_dump_count\": "
         << allocation_capture.expected_default_dump_count << ",\n"
         << "    \"expected_default_dump_bytes\": "
         << allocation_capture.expected_default_dump_bytes << ",\n"
         << "    \"default_manifest\": \""
         << allocation_capture.default_manifest << "\",\n"
         << "    \"default_manifest_sha256\": \""
         << allocation_capture.default_manifest_sha256 << "\"\n"
         << "  },\n"
         << "  \"frames\": [\n";
  for (std::size_t index = 0; index < frames.size(); ++index) {
    const FrameReceipt& frame = frames[index];
    output << "    {\n"
           << "      \"frame_id\": " << frame.frame_id << ",\n"
           << "      \"reset\": " << (frame.reset ? "true" : "false")
           << ",\n"
           << "      \"evaluate_result\": \""
           << NgxHex(frame.evaluate_result) << "\",\n"
           << "      \"evaluate_ms\": " << std::fixed
           << std::setprecision(3) << frame.evaluate_ms << ",\n"
           << "      \"input_sha256\": \"" << frame.input_sha256
           << "\",\n"
           << "      \"output_interpolated_sha256\": \""
           << frame.output_interpolated_sha256 << "\",\n"
           << "      \"output_real_sha256\": \""
           << frame.output_real_sha256 << "\",\n"
           << "      \"disable_sha256\": \"" << frame.disable_sha256
           << "\",\n"
           << "      \"output_real_provided\": "
           << (frame.output_real_provided ? "true" : "false") << ",\n"
           << "      \"disable_output_provided\": "
           << (frame.disable_output_provided ? "true" : "false") << ",\n"
           << "      \"disable_interpolation\": "
           << (frame.disable_interpolation ? "true" : "false") << ",\n"
           << "      \"output_nonzero_bytes\": "
           << frame.output_nonzero_bytes << ",\n"
           << "      \"output_untouched_pixels\": "
           << frame.output_untouched_pixels << ",\n"
           << "      \"differs_from_previous\": "
           << (frame.differs_from_previous ? "true" : "false") << ",\n"
           << "      \"differs_from_current\": "
           << (frame.differs_from_current ? "true" : "false") << ",\n"
           << "      \"generated_output_accepted\": "
           << (frame.generated_output_accepted ? "true" : "false") << '\n'
           << "    }" << (index + 1u == frames.size() ? "\n" : ",\n");
  }
  output << "  ],\n"
         << "  \"accepted\": " << (accepted ? "true" : "false") << "\n"
         << "}\n";
  if (!output) throw std::runtime_error("writing JSON receipt failed");
}

}  // namespace

int wmain(int argc, wchar_t** argv) {
  try {
    if (!HasFlag(argc, argv, L"--arm-execution")) {
      std::cerr << "error=explicit_arm_execution_required\n";
      return 2;
    }
    const std::wstring runtime_argument =
        GetArgument(argc, argv, L"--runtime");
    const std::wstring thunk_argument = GetArgument(argc, argv, L"--thunk");
    const std::wstring output_argument =
        GetArgument(argc, argv, L"--output-dir");
    const std::wstring nvapi_proxy_argument =
        GetArgument(argc, argv, L"--nvapi-proxy");
    const bool capture_runtime_allocations =
        HasFlag(argc, argv, L"--capture-runtime-allocations");
    const bool capture_runtime_upload_buffers =
        HasFlag(argc, argv, L"--capture-runtime-upload-buffers");
    const bool capture_runtime_default_buffers =
        HasFlag(argc, argv, L"--capture-runtime-default-buffers");
    const std::wstring expected_baseline_argument =
        GetArgument(argc, argv, L"--expected-baseline-output-sha256");
    const std::wstring expected_upload_count_argument =
        GetArgument(argc, argv, L"--expected-runtime-upload-buffer-count");
    const std::wstring expected_upload_bytes_argument =
        GetArgument(argc, argv, L"--expected-runtime-upload-bytes");
    const std::wstring default_buffer_tokens_argument =
        GetArgument(argc, argv, L"--runtime-default-buffer-tokens");
    const std::wstring expected_default_count_argument =
        GetArgument(argc, argv, L"--expected-runtime-default-buffer-count");
    const std::wstring expected_default_bytes_argument =
        GetArgument(argc, argv, L"--expected-runtime-default-buffer-bytes");
    const std::wstring default_capture_point_argument =
        GetArgument(argc, argv, L"--runtime-default-buffer-capture-point");
    if (runtime_argument.empty() || thunk_argument.empty() ||
        output_argument.empty()) {
      std::wcerr << L"usage: neuralstudio-fg-standalone.exe --arm-execution "
                    L"--runtime <nvngx_dlssg.dll> --thunk <nvngx.dll> "
                    L"--output-dir <new-or-empty-directory> "
                    L"[--width 640 --height 360 --rendered-frames 2]\n";
      return 2;
    }
    const std::uint32_t width =
        GetUintArgument(argc, argv, L"--width", 640, 128, 3840);
    const std::uint32_t height =
        GetUintArgument(argc, argv, L"--height", 360, 72, 2160);
    const std::uint32_t rendered_frames =
        GetUintArgument(argc, argv, L"--rendered-frames", 2, 2, 16);
    const bool use_hudless = HasFlag(argc, argv, L"--hudless");
    const bool use_ui_alpha = HasFlag(argc, argv, L"--ui-alpha");
    const bool use_ui_color_alpha =
        HasFlag(argc, argv, L"--ui-color-alpha");
    const bool use_distortion =
        HasFlag(argc, argv, L"--bidirectional-distortion");
    const bool use_output_real = HasFlag(argc, argv, L"--output-real");
    const bool use_disable_output =
        HasFlag(argc, argv, L"--disable-interpolation-output");
    const bool color_buffers_hdr = HasFlag(argc, argv, L"--hdr");
    const bool menu_detection =
        HasFlag(argc, argv, L"--menu-detection");
    const bool reset_every_frame =
        HasFlag(argc, argv, L"--reset-every-frame");
    const std::uint32_t multiframe_count =
        GetUintArgument(argc, argv, L"--multi-frame-count", 1, 1, 3);
    const std::uint32_t multiframe_index =
        GetUintArgument(argc, argv, L"--multi-frame-index", 1, 1, 3);
    const std::uint32_t output_poison_byte =
        GetUintArgument(argc, argv, L"--output-poison-byte", 0xa5u, 0, 255);
    const std::uint32_t subrect_x =
        GetUintArgument(argc, argv, L"--subrect-x", 0, 0, width - 1u);
    const std::uint32_t subrect_y =
        GetUintArgument(argc, argv, L"--subrect-y", 0, 0, height - 1u);
    const std::uint32_t subrect_width = GetUintArgument(
        argc, argv, L"--subrect-width", width - subrect_x, 1,
        width - subrect_x);
    const std::uint32_t subrect_height = GetUintArgument(
        argc, argv, L"--subrect-height", height - subrect_y, 1,
        height - subrect_y);
    std::string expected_baseline_output_sha256 = "not_requested";
    std::size_t expected_runtime_upload_buffer_count = 0;
    std::uint64_t expected_runtime_upload_bytes = 0;
    std::vector<std::uint64_t> runtime_default_buffer_tokens;
    std::size_t expected_runtime_default_buffer_count = 0;
    std::uint64_t expected_runtime_default_buffer_bytes = 0;
    std::string runtime_default_capture_point = "not_requested";
    std::string runtime_default_capture_slug = "not_requested";
    if (capture_runtime_allocations) {
      if (expected_baseline_argument.empty()) {
        throw std::invalid_argument(
            "--capture-runtime-allocations requires --expected-baseline-output-sha256");
      }
      expected_baseline_output_sha256 =
          NormalizeSha256Argument(expected_baseline_argument);
    } else if (!expected_baseline_argument.empty()) {
      throw std::invalid_argument(
          "baseline output hash is only valid with runtime allocation capture");
    }
    if (capture_runtime_upload_buffers && !capture_runtime_allocations) {
      throw std::invalid_argument(
          "--capture-runtime-upload-buffers requires --capture-runtime-allocations");
    }
    if (capture_runtime_upload_buffers) {
      if (expected_upload_count_argument.empty() ||
          expected_upload_bytes_argument.empty()) {
        throw std::invalid_argument(
            "runtime upload capture requires expected buffer count and bytes");
      }
      expected_runtime_upload_buffer_count = static_cast<std::size_t>(
          ParseUint64Argument(expected_upload_count_argument, 1, 1000000));
      expected_runtime_upload_bytes = ParseUint64Argument(
          expected_upload_bytes_argument, 1,
          (std::numeric_limits<std::uint64_t>::max)());
    } else if (!expected_upload_count_argument.empty() ||
               !expected_upload_bytes_argument.empty()) {
      throw std::invalid_argument(
          "runtime upload expectations require upload capture");
    }
    if (capture_runtime_default_buffers && !capture_runtime_allocations) {
      throw std::invalid_argument(
          "--capture-runtime-default-buffers requires --capture-runtime-allocations");
    }
    if (capture_runtime_default_buffers && capture_runtime_upload_buffers) {
      throw std::invalid_argument(
          "upload mapping and DEFAULT readback captures must run separately");
    }
    if (capture_runtime_default_buffers) {
      if (default_buffer_tokens_argument.empty() ||
          expected_default_count_argument.empty() ||
          expected_default_bytes_argument.empty() ||
          default_capture_point_argument.empty()) {
        throw std::invalid_argument(
            "DEFAULT readback requires token, count, and byte expectations");
      }
      runtime_default_buffer_tokens =
          ParseTokenList(default_buffer_tokens_argument);
      expected_runtime_default_buffer_count = static_cast<std::size_t>(
          ParseUint64Argument(expected_default_count_argument, 1, 1000000));
      expected_runtime_default_buffer_bytes = ParseUint64Argument(
          expected_default_bytes_argument, 1,
          (std::numeric_limits<std::uint64_t>::max)());
      if (runtime_default_buffer_tokens.size() !=
          expected_runtime_default_buffer_count) {
        throw std::invalid_argument(
            "DEFAULT token list size disagrees with expected count");
      }
      if (default_capture_point_argument == L"post_create") {
        runtime_default_capture_point = "post_create_fence_pre_evaluate";
        runtime_default_capture_slug = "post-create";
      } else if (default_capture_point_argument == L"post_evaluate_0") {
        runtime_default_capture_point =
            "post_evaluate_0_fence_pre_evaluate_1";
        runtime_default_capture_slug = "post-evaluate-0";
      } else if (default_capture_point_argument == L"post_evaluate_1") {
        if (rendered_frames < 3u) {
          throw std::invalid_argument(
              "post_evaluate_1 DEFAULT capture requires at least three "
              "rendered frames");
        }
        runtime_default_capture_point =
            "post_evaluate_1_fence_pre_evaluate_2";
        runtime_default_capture_slug = "post-evaluate-1";
      } else {
        throw std::invalid_argument("unknown DEFAULT buffer capture point");
      }
    } else if (!default_buffer_tokens_argument.empty() ||
               !expected_default_count_argument.empty() ||
               !expected_default_bytes_argument.empty() ||
               !default_capture_point_argument.empty()) {
      throw std::invalid_argument(
          "DEFAULT buffer arguments require DEFAULT readback capture");
    }
    std::ostringstream resource_profile_builder;
    resource_profile_builder << "core";
    if (use_hudless) resource_profile_builder << "+hudless";
    if (use_ui_alpha) resource_profile_builder << "+ui_alpha";
    if (use_ui_color_alpha) resource_profile_builder << "+ui_color_alpha";
    if (use_distortion) resource_profile_builder << "+bdf";
    if (use_output_real) resource_profile_builder << "+output_real";
    if (use_disable_output) resource_profile_builder << "+disable_output";
    const std::string resource_profile = resource_profile_builder.str();
    std::cout << "config.resource_profile=" << resource_profile << '\n';
    std::cout << "config.color_buffers_hdr="
              << (color_buffers_hdr ? "true" : "false") << '\n';
    std::cout << "config.menu_detection="
              << (menu_detection ? "true" : "false") << '\n';
    std::cout << "config.reset_every_frame="
              << (reset_every_frame ? "true" : "false") << '\n';
    std::cout << "config.multiframe_count=" << multiframe_count << '\n';
    std::cout << "config.multiframe_index=" << multiframe_index << '\n';
    std::cout << "config.output_poison_byte=" << output_poison_byte << '\n';
    std::cout << "config.capture_runtime_allocations="
              << (capture_runtime_allocations ? "true" : "false") << '\n';
    std::cout << "config.capture_runtime_upload_buffers="
              << (capture_runtime_upload_buffers ? "true" : "false") << '\n';
    std::cout << "config.capture_runtime_default_buffers="
              << (capture_runtime_default_buffers ? "true" : "false") << '\n';
    std::cout << "config.expected_baseline_output_sha256="
              << expected_baseline_output_sha256 << '\n';
    std::cout << "config.expected_runtime_upload_buffer_count="
              << expected_runtime_upload_buffer_count << '\n';
    std::cout << "config.expected_runtime_upload_bytes="
              << expected_runtime_upload_bytes << '\n';
    std::cout << "config.expected_runtime_default_buffer_count="
              << expected_runtime_default_buffer_count << '\n';
    std::cout << "config.expected_runtime_default_buffer_bytes="
              << expected_runtime_default_buffer_bytes << '\n';
    std::cout << "config.runtime_default_capture_point="
              << runtime_default_capture_point << '\n';
    if (use_ui_alpha && use_ui_color_alpha) {
      throw std::invalid_argument(
          "UI alpha and UI color+alpha are mutually exclusive");
    }
    if ((use_ui_alpha || use_ui_color_alpha) && !use_hudless) {
      throw std::invalid_argument(
          "UI recomposition requires --hudless with one UI input");
    }
    if (multiframe_index > multiframe_count) {
      throw std::invalid_argument(
          "multi-frame index cannot exceed multi-frame count");
    }
    if (color_buffers_hdr) {
      throw std::invalid_argument(
            "HDR requested but this pinned first-stage host only implements the supported RGBA8 SDR contract");
    }

    // Explicit synthetic sequence input leaves the archived built-in fixture
    // and its default parameter values unchanged. Verify before loading a DLL.
    const auto sequence_argument = GetArgument(argc, argv, L"--input-sequence");
    const auto sequence_hash_argument = GetArgument(argc, argv, L"--input-sequence-sha256");
    const auto motion_scale_argument = GetArgument(argc, argv, L"--reference-motion-scale");
    std::string input_sequence_sha256 = "builtin-synthetic";
    std::string motion_scale_contract = "recorded-inverse-extent";
    bool unit_motion_scale = false;
    std::vector<neuralstudio::fg::offline::Frame> input_sequence;
    if (!sequence_argument.empty()) {
      if (sequence_hash_argument.empty()) throw std::invalid_argument("external sequence requires its expected SHA256");
      if (width != 640 || height != 360 || use_hudless || use_ui_alpha || use_ui_color_alpha || use_distortion ||
          reset_every_frame || multiframe_count != 1 || multiframe_index != 1 || subrect_x != 0 || subrect_y != 0 ||
          subrect_width != width || subrect_height != height || capture_runtime_allocations || menu_detection)
        throw std::invalid_argument("external sequence requires the unmodified 640x360 core 2x profile");
      const std::filesystem::path sequence_path(sequence_argument);
      if (!std::filesystem::is_regular_file(sequence_path) ||
          std::filesystem::file_size(sequence_path) > neuralstudio::fg::offline::kMaximumBytes)
        throw std::invalid_argument("external sequence missing or oversized");
      const auto bytes = ReadFile(sequence_path);
      input_sequence_sha256 = Sha256(bytes);
      if (input_sequence_sha256 != NormalizeSha256Argument(sequence_hash_argument))
        throw std::invalid_argument("external sequence SHA256 mismatch");
      input_sequence = neuralstudio::fg::offline::Decode(bytes);
      if (input_sequence.size() != rendered_frames) throw std::invalid_argument("external sequence frame count differs from request");
      if (!motion_scale_argument.empty() && motion_scale_argument != L"recorded" && motion_scale_argument != L"one")
        throw std::invalid_argument("reference motion scale must be recorded or one");
      unit_motion_scale = motion_scale_argument == L"one";
      if (unit_motion_scale) motion_scale_contract = "explicit-one";
    } else if (!sequence_hash_argument.empty() || !motion_scale_argument.empty()) {
      throw std::invalid_argument("sequence hash and motion override require an external sequence");
    }
    std::cout << "config.input_sequence_sha256=" << input_sequence_sha256 << '\n';
    std::cout << "config.motion_scale_contract=" << motion_scale_contract << '\n';

    const std::filesystem::path runtime_path =
        std::filesystem::weakly_canonical(runtime_argument);
    const std::filesystem::path thunk_path =
        std::filesystem::weakly_canonical(thunk_argument);
    std::optional<std::filesystem::path> nvapi_proxy_path;
    if (!nvapi_proxy_argument.empty()) {
      const std::filesystem::path requested_proxy(nvapi_proxy_argument);
      if (!requested_proxy.is_absolute()) {
        throw std::invalid_argument("--nvapi-proxy must be an absolute path");
      }
      nvapi_proxy_path = std::filesystem::weakly_canonical(requested_proxy);
      if (!std::filesystem::is_regular_file(*nvapi_proxy_path) ||
          _wcsicmp(nvapi_proxy_path->filename().c_str(), L"nvapi64.dll") != 0) {
        throw std::invalid_argument(
            "--nvapi-proxy must name an existing nvapi64.dll file");
      }
    }
    if (!std::filesystem::is_regular_file(runtime_path) ||
        !std::filesystem::is_regular_file(thunk_path)) {
      throw std::runtime_error("runtime or thunk path is not a regular file");
    }
    std::filesystem::path output_directory =
        std::filesystem::absolute(output_argument);
    if (std::filesystem::exists(output_directory)) {
      if (!std::filesystem::is_directory(output_directory) ||
          !std::filesystem::is_empty(output_directory)) {
        throw std::runtime_error("output directory must be new or empty");
      }
    } else {
      std::filesystem::create_directories(output_directory);
    }
    output_directory = std::filesystem::weakly_canonical(output_directory);
    NgxLogSink ngx_log_sink(output_directory / L"ngx-callback.log");
    WriteScenarioSnapshot(
        output_directory / L"scenario-before.json", "before", width, height,
        rendered_frames, subrect_x, subrect_y, subrect_width, subrect_height,
        multiframe_count, multiframe_index, resource_profile,
        output_poison_byte, std::nullopt);

    const std::vector<std::uint8_t> runtime_bytes = ReadFile(runtime_path);
    const std::vector<std::uint8_t> thunk_bytes = ReadFile(thunk_path);
    const std::string runtime_sha256 = Sha256(runtime_bytes);
    const std::string thunk_sha256 = Sha256(thunk_bytes);
    const std::string requested_nvapi_proxy_sha256 = nvapi_proxy_path
        ? Sha256(ReadFile(*nvapi_proxy_path))
        : "not_requested";
    std::string loaded_nvapi_proxy_path_utf8 = "not_requested";
    std::string loaded_nvapi_proxy_sha256_value = "not_requested";
    const std::string run_nonce = RandomNonce();
    const std::string queue_nonce = RandomNonce();
    std::cout << "probe=dlss_fg_standalone_d3d12\n";
    std::cout << "scope=execution_receipt_only\n";
    std::cout << "claim.parity=false\n";
    std::cout << "claim.mac_port=false\n";
    std::cout << "runtime.expected_sha256=" << kExpectedRuntimeSha256 << '\n';
    std::cout << "runtime.sha256=" << runtime_sha256 << '\n';
    std::cout << "thunk.sha256=" << thunk_sha256 << '\n';
    std::cout << "runtime.bytes=" << runtime_bytes.size() << '\n';
    std::cout << "process.id=" << GetCurrentProcessId() << '\n';
    std::cout << "process.run_nonce=" << run_nonce << '\n';
    std::cout << "d3d12.queue_nonce=" << queue_nonce << '\n';
    std::cout << "nvapi.proxy.requested="
              << (nvapi_proxy_path ? "true" : "false") << '\n';
    if (runtime_sha256 != kExpectedRuntimeSha256) {
      throw std::runtime_error("exact nvngx_dlssg.dll hash mismatch");
    }

    if (!SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_SYSTEM32 |
                                  LOAD_LIBRARY_SEARCH_USER_DIRS)) {
      throw std::runtime_error("SetDefaultDllDirectories failed");
    }
    const DLL_DIRECTORY_COOKIE runtime_cookie =
        AddDllDirectory(runtime_path.parent_path().c_str());
    if (runtime_cookie == nullptr) {
      throw std::runtime_error("AddDllDirectory(runtime) failed");
    }
    struct CookieCloser {
      DLL_DIRECTORY_COOKIE value;
      ~CookieCloser() {
        if (value != nullptr) RemoveDllDirectory(value);
      }
    } cookie_closer{runtime_cookie};

    UniqueLibrary nvapi_proxy;
    FgNvapiSetPhase set_trace_phase = &NoopNvapiSetPhase;
    FgNvapiAttachD3D12Device attach_descriptor_trace = nullptr;
    FgNvapiDetachD3D12Device detach_descriptor_trace = nullptr;
    if (nvapi_proxy_path) {
      if (GetModuleHandleW(L"nvapi64.dll") != nullptr) {
        throw std::runtime_error(
            "nvapi64.dll was already loaded before explicit proxy preload");
      }
      nvapi_proxy.reset(LoadLibraryExW(
          nvapi_proxy_path->c_str(), nullptr,
          LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 |
              LOAD_LIBRARY_SEARCH_USER_DIRS));
      if (!nvapi_proxy) {
        throw std::runtime_error("explicit NvAPI proxy preload failed");
      }
      const std::filesystem::path loaded_nvapi_proxy =
          LoadedModulePath(nvapi_proxy.get());
      const std::string loaded_nvapi_proxy_sha256 =
          Sha256(ReadFile(loaded_nvapi_proxy));
      if (_wcsicmp(loaded_nvapi_proxy.filename().c_str(), L"nvapi64.dll") !=
              0 ||
          !std::filesystem::equivalent(*nvapi_proxy_path,
                                       loaded_nvapi_proxy) ||
          loaded_nvapi_proxy_sha256 != requested_nvapi_proxy_sha256) {
        throw std::runtime_error(
            "loaded NvAPI proxy basename, path, or hash differs from request");
      }
      std::wcout << L"nvapi.proxy.path=\"" << loaded_nvapi_proxy.c_str()
                 << L"\"\n";
      std::cout << "nvapi.proxy.sha256=" << loaded_nvapi_proxy_sha256
                << '\n';
      loaded_nvapi_proxy_path_utf8 = WideToUtf8(loaded_nvapi_proxy.native());
      loaded_nvapi_proxy_sha256_value = loaded_nvapi_proxy_sha256;
      set_trace_phase = LoadFunction<FgNvapiSetPhase>(
          nvapi_proxy.get(), "NeuralStudioFgNvapiTraceSetPhase");
      if (set_trace_phase == nullptr) {
        throw std::runtime_error(
            "NvAPI proxy phase-marker export is required but missing");
      }
      attach_descriptor_trace = LoadFunction<FgNvapiAttachD3D12Device>(
          nvapi_proxy.get(),
          "NeuralStudioFgNvapiTraceAttachD3D12Device");
      detach_descriptor_trace = LoadFunction<FgNvapiDetachD3D12Device>(
          nvapi_proxy.get(),
          "NeuralStudioFgNvapiTraceDetachD3D12Device");
      if ((attach_descriptor_trace == nullptr) !=
          (detach_descriptor_trace == nullptr)) {
        throw std::runtime_error(
            "NvAPI proxy descriptor attach/detach exports are inconsistent");
      }
      std::cout << "nvapi.proxy.phase_export=required_present\n";
      std::cout << "nvapi.proxy.descriptor_trace_exports="
                << (attach_descriptor_trace != nullptr ? "present" : "absent")
                << '\n';
    } else {
      std::cout << "nvapi.proxy.availability=not_requested\n";
      std::cout << "nvapi.proxy.phase_export=not_required\n";
    }

    UniqueLibrary thunk(LoadLibraryExW(
        thunk_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 |
            LOAD_LIBRARY_SEARCH_USER_DIRS));
    if (!thunk) throw std::runtime_error("loading typed NGX thunk failed");
    UniqueLibrary runtime(LoadLibraryExW(
        runtime_path.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_SYSTEM32 |
            LOAD_LIBRARY_SEARCH_USER_DIRS));
    if (!runtime) throw std::runtime_error("loading nvngx_dlssg.dll failed");
    const std::filesystem::path loaded_runtime = LoadedModulePath(runtime.get());
    const std::filesystem::path loaded_thunk = LoadedModulePath(thunk.get());
    if (!std::filesystem::equivalent(runtime_path, loaded_runtime) ||
        !std::filesystem::equivalent(thunk_path, loaded_thunk)) {
      throw std::runtime_error("loaded module path identity mismatch");
    }
    std::wcout << L"runtime.path=\"" << loaded_runtime.c_str() << L"\"\n";
    std::wcout << L"thunk.path=\"" << loaded_thunk.c_str() << L"\"\n";

    RuntimeApi runtime_api = LoadRuntimeApi(runtime.get());
    ThunkApi thunk_api = LoadThunkApi(thunk.get());
    const std::uint32_t application_id = runtime_api.get_application_id();
    const std::uint32_t api_version = runtime_api.get_api_version();
    const std::uint32_t snippet_version = runtime_api.get_snippet_version();
    std::cout << "runtime.application_id=0x" << std::hex << std::uppercase
              << application_id << std::dec << '\n';
    std::cout << "runtime.api_version=" << api_version << '\n';
    std::cout << "runtime.snippet_version=0x" << std::hex << std::uppercase
              << snippet_version << std::dec << '\n';
    std::cout << "runtime.feature_id="
              << static_cast<unsigned int>(kFrameGenerationFeature) << '\n';

    set_trace_phase("pre_device");
    GpuContext gpu = CreateGpuContext();
    ScopedFgDescriptorTraceAttachment descriptor_trace_attachment{
        detach_descriptor_trace, false};
    if (attach_descriptor_trace != nullptr) {
      if (!attach_descriptor_trace(gpu.device.Get())) {
        throw std::runtime_error("NvAPI proxy D3D12 descriptor attach failed");
      }
      descriptor_trace_attachment.active = true;
      std::cout << "nvapi.proxy.descriptor_trace_attach=accepted\n";
    } else {
      std::cout << "nvapi.proxy.descriptor_trace_attach=not_available\n";
    }
    const std::filesystem::path allocation_semantic_path =
        output_directory / L"runtime-allocations-semantic.jsonl";
    const std::filesystem::path allocation_volatile_path =
        output_directory / L"runtime-allocations-volatile.jsonl";
    std::optional<neuralstudio::fg::RuntimeAllocationCapture>
        runtime_allocation_capture;
    if (capture_runtime_allocations) {
      runtime_allocation_capture.emplace(
          gpu.device.Get(), allocation_semantic_path, allocation_volatile_path);
      std::cout << "runtime.allocation_capture=armed\n";
    }
    const std::string driver_version = ReadDriverVersion(gpu.adapter.Get());
    std::wcout << L"adapter.name=\"" << gpu.adapter_name << L"\"\n";
    std::cout << "adapter.vendor_id=0x" << std::hex << std::uppercase
              << gpu.adapter_desc.VendorId << '\n';
    std::cout << "adapter.device_id=0x" << gpu.adapter_desc.DeviceId
              << std::dec << '\n';
    std::cout << "adapter.luid_high=" << gpu.adapter_desc.AdapterLuid.HighPart
              << '\n';
    std::cout << "adapter.luid_low=" << gpu.adapter_desc.AdapterLuid.LowPart
              << '\n';
    std::cout << "adapter.driver_version=" << driver_version << '\n';
    std::cout << "d3d12.debug_layer_enabled="
              << (gpu.debug_layer_enabled ? "true" : "false") << '\n';
    SampleLocalVideoMemory(gpu, "baseline");
    std::cout << "capture.internal_module_events=opaque_d3d12_nvapi\n";
    std::cout << "capture.internal_kernel_events=opaque_d3d12_nvapi\n";
    std::cout << "capture.cuda_trace=not_applicable\n";

    neuralstudio::fg::ParameterStore parameters;
    const NVSDK_NGX_Result populate_result =
        thunk_api.populate(runtime_api.populate, parameters.AsNgx());
    PrintNgx("runtime.populate_result", populate_result);
    if (NVSDK_NGX_FAILED(populate_result)) {
      throw std::runtime_error("DLSSG parameter population failed");
    }
    NVSDK_NGX_AppLogCallback ngx_log_callback = &CaptureNgxLog;
    void* ngx_log_callback_address = nullptr;
    static_assert(sizeof(ngx_log_callback_address) == sizeof(ngx_log_callback));
    std::memcpy(&ngx_log_callback_address, &ngx_log_callback,
                sizeof(ngx_log_callback_address));
    parameters.SetI32("Minimum.Logging.Level",
                      static_cast<int>(NVSDK_NGX_LOGGING_LEVEL_VERBOSE));
    parameters.SetPointerValue("Log.Callback", ngx_log_callback_address);
    parameters.SetI32("Disable.Other.Logging.Sinks", 0);
    if (runtime_allocation_capture) {
      PFN_NVSDK_NGX_D3D12_ResourceAllocCallback allocation_callback =
          neuralstudio::fg::RuntimeAllocationCapture::allocation_callback();
      PFN_NVSDK_NGX_ResourceReleaseCallback release_callback =
          neuralstudio::fg::RuntimeAllocationCapture::release_callback();
      void* allocation_callback_address = nullptr;
      void* release_callback_address = nullptr;
      static_assert(sizeof(allocation_callback_address) ==
                    sizeof(allocation_callback));
      static_assert(sizeof(release_callback_address) ==
                    sizeof(release_callback));
      std::memcpy(&allocation_callback_address, &allocation_callback,
                  sizeof(allocation_callback_address));
      std::memcpy(&release_callback_address, &release_callback,
                  sizeof(release_callback_address));
      parameters.SetPointerValue(NVSDK_NGX_Parameter_ResourceAllocCallback,
                                 allocation_callback_address);
      parameters.SetPointerValue(NVSDK_NGX_Parameter_ResourceReleaseCallback,
                                 release_callback_address);
      std::cout << "runtime.allocation_callbacks=installed\n";
    }
    std::cout << "runtime.ngx_callback_log=ngx-callback.log\n";
    std::cout << "runtime.ngx_minimum_logging_level="
              << static_cast<int>(NVSDK_NGX_LOGGING_LEVEL_VERBOSE) << '\n';
    std::cout << "runtime.ngx_other_logging_sinks_disabled=false\n";

    const std::filesystem::path data_directory = output_directory / L"ngx-data";
    std::filesystem::create_directories(data_directory);
    WriteDirectoryInventory(output_directory / L"ngx-cache-before.json",
                            data_directory, "before");
    NVSDK_NGX_FeatureDiscoveryInfo discovery{};
    discovery.SDKVersion = static_cast<NVSDK_NGX_Version>(api_version);
    discovery.FeatureID = kFrameGenerationFeature;
    discovery.Identifier.IdentifierType =
        NVSDK_NGX_Application_Identifier_Type_Application_Id;
    discovery.Identifier.v.ApplicationId = application_id;
    discovery.ApplicationDataPath = data_directory.c_str();
    NVSDK_NGX_FeatureRequirement requirement{};
    set_trace_phase("requirements");
    const NVSDK_NGX_Result requirements_result = thunk_api.requirements(
        runtime_api.requirements, gpu.adapter.Get(), &discovery, &requirement);
    PrintNgx("runtime.requirements_result", requirements_result);
    std::cout << "runtime.requirements_support_flags=0x" << std::hex
              << std::uppercase
              << static_cast<unsigned int>(requirement.FeatureSupported)
              << std::dec << '\n';
    std::cout << "runtime.requirements_min_arch=0x" << std::hex
              << std::uppercase << requirement.MinHWArchitecture << std::dec
              << '\n';
    if (NVSDK_NGX_FAILED(requirements_result) ||
        requirement.FeatureSupported != NVSDK_NGX_FeatureSupportResult_Supported) {
      throw std::runtime_error("frame generation is unsupported on this adapter");
    }

    set_trace_phase("init");
    const NVSDK_NGX_Result init_result = thunk_api.init(
        runtime_api.init, application_id, data_directory.c_str(),
        gpu.device.Get(), static_cast<NVSDK_NGX_Version>(api_version),
        parameters.AsNgx());
    set_trace_phase("post_init");
    PrintNgx("runtime.init_result", init_result);
    if (NVSDK_NGX_FAILED(init_result)) {
      throw std::runtime_error("DLSSG runtime initialization failed");
    }
    RuntimeCleanup cleanup{&runtime_api, &thunk_api, gpu.device.Get(),
                           set_trace_phase};
    cleanup.initialized = true;

    const auto available = ReadIntegral(
        parameters, NVSDK_NGX_Parameter_FrameGeneration_Available);
    const auto needs_driver = ReadIntegral(
        parameters, NVSDK_NGX_Parameter_FrameGeneration_NeedsUpdatedDriver);
    const auto multiframe_max = ReadIntegral(
        parameters, NVSDK_NGX_DLSSG_Parameter_MultiFrameCountMax);
    std::cout << "runtime.frame_generation_available="
              << (available ? std::to_string(*available) : "unavailable")
              << '\n';
    std::cout << "runtime.needs_updated_driver="
              << (needs_driver ? std::to_string(*needs_driver) : "unavailable")
              << '\n';
    std::cout << "runtime.multiframe_count_max="
              << (multiframe_max ? std::to_string(*multiframe_max)
                                 : "unavailable")
              << '\n';
    if ((available && *available == 0) || (needs_driver && *needs_driver != 0) ||
        (multiframe_max &&
         *multiframe_max < static_cast<long long>(multiframe_count))) {
      throw std::runtime_error(
          "DLSSG capability parameters reject requested multiframe mode");
    }

    void* estimate_pointer = nullptr;
    const NVSDK_NGX_Result estimate_pointer_result = parameters.GetPointerValue(
        NVSDK_NGX_Parameter_DLSSGEstimateVRAMCallback, &estimate_pointer);
    PrintNgx("runtime.estimate_vram_callback_result", estimate_pointer_result);
    if (NVSDK_NGX_FAILED(estimate_pointer_result) ||
        estimate_pointer == nullptr) {
      throw std::runtime_error("DLSSG VRAM estimator callback unavailable");
    }
    auto estimate_vram =
        reinterpret_cast<PFN_NVSDK_NGX_DLSSG_EstimateVRAMCallback>(
            estimate_pointer);
    std::uint64_t estimated_vram = 0;
    std::size_t estimated_vram_native = 0;
    const NVSDK_NGX_Result estimate_result = estimate_vram(
        subrect_width, subrect_height, subrect_width, subrect_height,
        static_cast<std::uint32_t>(kColorFormat),
        static_cast<std::uint32_t>(kMotionFormat),
        static_cast<std::uint32_t>(kDepthFormat),
        static_cast<std::uint32_t>(use_hudless ? kColorFormat
                                               : DXGI_FORMAT_UNKNOWN),
        static_cast<std::uint32_t>(
            use_ui_alpha ? kUiAlphaFormat
                         : (use_ui_color_alpha ? kColorFormat
                                               : DXGI_FORMAT_UNKNOWN)),
        &estimated_vram_native);
    PrintNgx("runtime.estimate_vram_result", estimate_result);
    estimated_vram = estimated_vram_native;
    std::cout << "runtime.estimated_vram_bytes=" << estimated_vram << '\n';
    if (NVSDK_NGX_FAILED(estimate_result) || estimated_vram == 0) {
      throw std::runtime_error("DLSSG VRAM estimate failed");
    }

    unsigned int always_flags =
        NVSDK_NGX_DLSSG_ResourceFlags_Backbuffer |
        NVSDK_NGX_DLSSG_ResourceFlags_MVecs |
        NVSDK_NGX_DLSSG_ResourceFlags_Depth |
        NVSDK_NGX_DLSSG_ResourceFlags_OutputInterpolated;
    unsigned int never_flags = 0;
    if (use_hudless) {
      always_flags |= NVSDK_NGX_DLSSG_ResourceFlags_HUDLess;
    } else {
      never_flags |= NVSDK_NGX_DLSSG_ResourceFlags_HUDLess;
    }
    if (use_ui_alpha) {
      always_flags |= NVSDK_NGX_DLSSG_ResourceFlags_UIAlpha;
    } else {
      never_flags |= NVSDK_NGX_DLSSG_ResourceFlags_UIAlpha;
    }
    if (use_ui_color_alpha) {
      always_flags |= NVSDK_NGX_DLSSG_ResourceFlags_UI;
    } else {
      never_flags |= NVSDK_NGX_DLSSG_ResourceFlags_UI;
    }
    if (use_distortion) {
      always_flags |=
          NVSDK_NGX_DLSSG_ResourceFlags_BidirectionalDistortionField;
    } else {
      never_flags |=
          NVSDK_NGX_DLSSG_ResourceFlags_BidirectionalDistortionField;
    }
    if (use_output_real) {
      always_flags |= NVSDK_NGX_DLSSG_ResourceFlags_OutputReal;
    } else {
      never_flags |= NVSDK_NGX_DLSSG_ResourceFlags_OutputReal;
    }
    if (use_disable_output) {
      always_flags |=
          NVSDK_NGX_DLSSG_ResourceFlags_OutputDisableInterpolation;
    } else {
      never_flags |=
          NVSDK_NGX_DLSSG_ResourceFlags_OutputDisableInterpolation;
    }
    parameters.SetU32(NVSDK_NGX_Parameter_CreationNodeMask, 1);
    parameters.SetU32(NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
    const bool dynamic_resolution =
        neuralstudio::fg::SetCreateResolutionContract(
            parameters, width, height,
            {subrect_x, subrect_y, subrect_width, subrect_height});
    std::cout << "contract.dynamic_resolution_scaling="
              << (dynamic_resolution ? "true" : "false") << '\n';
    std::cout << "contract.internal_render_width="
              << (dynamic_resolution ? subrect_width : width) << '\n';
    std::cout << "contract.internal_render_height="
              << (dynamic_resolution ? subrect_height : height) << '\n';
    std::cout << "contract.resource_width=" << width << '\n';
    std::cout << "contract.resource_height=" << height << '\n';
    std::cout << "contract.feature_create_width="
              << (dynamic_resolution ? subrect_width : width) << '\n';
    std::cout << "contract.feature_create_height="
              << (dynamic_resolution ? subrect_height : height) << '\n';
    parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_BackbufferFormat,
                      static_cast<unsigned int>(kColorFormat));
    parameters.SetU32(
        NVSDK_NGX_DLSSG_Parameter_ResourceAlwaysProvided_Flags, always_flags);
    parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_ResourceNeverProvided_Flags,
                      never_flags);
    parameters.SetU32(
        NVSDK_NGX_DLSSG_Parameter_UserInterfaceRecompositionEnabled,
        (use_hudless && (use_ui_alpha || use_ui_color_alpha)) ? 1u : 0u);
    parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_AsyncCreateEnabled, 0);
    parameters.SetU32(NVSDK_NGX_Parameter_FreeMemOnReleaseFeature, 1);
    parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_FullscreenMode,
                      NVSDK_NGX_DLSSG_FullscreenMode_Windowed);
    parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_TargetFrameRate, 60.0f);

    NVSDK_NGX_Handle* feature_handle = nullptr;
    const auto create_start = std::chrono::steady_clock::now();
    set_trace_phase("create");
    const NVSDK_NGX_Result create_result = thunk_api.create(
        runtime_api.create, gpu.list.Get(), kFrameGenerationFeature,
        parameters.AsNgx(), &feature_handle);
    set_trace_phase("create.end");
    PrintNgx("runtime.create_result", create_result);
    std::cout << "runtime.feature_handle="
              << (feature_handle != nullptr ? "acquired" : "null") << '\n';
    if (NVSDK_NGX_FAILED(create_result) || feature_handle == nullptr) {
      throw std::runtime_error("DLSSG feature creation failed");
    }
    cleanup.handle = feature_handle;
    ExecuteAndReset(gpu);
    SampleLocalVideoMemory(gpu, "after_create");
    const std::filesystem::path runtime_upload_dump_directory =
        output_directory / L"runtime-upload-buffers";
    const std::filesystem::path runtime_upload_manifest =
        output_directory / L"runtime-upload-buffers-manifest.json";
    if (capture_runtime_upload_buffers) {
      if (!runtime_allocation_capture) {
        throw std::runtime_error(
            "runtime upload capture lost allocation-capture state");
      }
      runtime_allocation_capture->CaptureLiveCpuVisibleBuffers(
          runtime_upload_dump_directory, runtime_upload_manifest);
      std::cout << "runtime.upload_capture=completed\n";
      std::cout << "runtime.upload_dump_count="
                << runtime_allocation_capture->upload_dump_count() << '\n';
      std::cout << "runtime.upload_dump_bytes="
                << runtime_allocation_capture->upload_dump_bytes() << '\n';
      if (runtime_allocation_capture->upload_dump_count() !=
              expected_runtime_upload_buffer_count ||
          runtime_allocation_capture->upload_dump_bytes() !=
              expected_runtime_upload_bytes) {
        throw std::runtime_error(
            "runtime upload capture count or logical byte total mismatch");
      }
      std::cout << "runtime.upload_capture_expected_extent_match=true\n";
    }
    const std::filesystem::path runtime_default_dump_directory =
        output_directory /
        ("runtime-default-buffers-" + runtime_default_capture_slug);
    const std::filesystem::path runtime_default_manifest =
        output_directory / ("runtime-default-buffers-" +
                            runtime_default_capture_slug + ".manifest.json");
    const auto capture_default_buffers = [&]() {
      if (!runtime_allocation_capture) {
        throw std::runtime_error(
            "DEFAULT readback lost allocation-capture state");
      }
      const std::string phase =
          "capture_default_buffers." + runtime_default_capture_slug;
      set_trace_phase(phase.c_str());
      runtime_allocation_capture->PrepareDefaultBufferReadbacks(
          runtime_default_buffer_tokens, gpu.list.Get());
      ExecuteAndReset(gpu);
      const std::string complete_phase = phase + ".copy_complete";
      set_trace_phase(complete_phase.c_str());
      runtime_allocation_capture->CompleteDefaultBufferReadbacks(
          runtime_default_dump_directory, runtime_default_manifest,
          runtime_default_capture_point);
      std::cout << "runtime.default_buffer_capture=completed\n";
      std::cout << "runtime.default_buffer_dump_count="
                << runtime_allocation_capture->default_dump_count() << '\n';
      std::cout << "runtime.default_buffer_dump_bytes="
                << runtime_allocation_capture->default_dump_bytes() << '\n';
      if (runtime_allocation_capture->default_dump_count() !=
              expected_runtime_default_buffer_count ||
          runtime_allocation_capture->default_dump_bytes() !=
              expected_runtime_default_buffer_bytes) {
        throw std::runtime_error(
            "DEFAULT readback count or logical byte total mismatch");
      }
      std::cout << "runtime.default_buffer_expected_extent_match=true\n";
    };
    if (capture_runtime_default_buffers &&
        runtime_default_capture_point ==
            "post_create_fence_pre_evaluate") {
      capture_default_buffers();
    }
    const auto create_end = std::chrono::steady_clock::now();
    std::cout << std::fixed << std::setprecision(3)
              << "runtime.create_gpu_ms="
              << std::chrono::duration<double, std::milli>(create_end -
                                                            create_start)
                     .count()
              << std::defaultfloat << '\n';

    void* settings_pointer = nullptr;
    const NVSDK_NGX_Result settings_pointer_result =
        parameters.GetPointerValue(
            NVSDK_NGX_Parameter_DLSSGGetCurrentSettingsCallback,
            &settings_pointer);
    PrintNgx("runtime.current_settings_callback_result",
             settings_pointer_result);
    if (NVSDK_NGX_FAILED(settings_pointer_result) ||
        settings_pointer == nullptr) {
      throw std::runtime_error("DLSSG current-settings callback unavailable");
    }
    const auto current_settings =
        reinterpret_cast<PFN_NVSDK_NGX_DLSSG_GetCurrentSettingsCallback>(
            settings_pointer);
    const NVSDK_NGX_Result settings_result =
        current_settings(feature_handle, parameters.AsNgx());
    PrintNgx("runtime.current_settings_result", settings_result);
    if (NVSDK_NGX_FAILED(settings_result)) {
      throw std::runtime_error("DLSSG current-settings query failed");
    }

    std::vector<FrameReceipt> frame_receipts;
    std::vector<std::uint8_t> previous_input;
    bool all_generated_outputs_accepted = true;
    for (std::uint32_t frame_id = 0; frame_id < rendered_frames; ++frame_id) {
      neuralstudio::fg::SyntheticFrame synthetic;
      bool frame_reset = frame_id == 0 || reset_every_frame;
      if (input_sequence.empty()) {
        synthetic = neuralstudio::fg::MakeSyntheticFrame(width, height, frame_id);
      } else {
        const auto& input = input_sequence.at(frame_id);
        frame_reset = input.reset;
        synthetic.backbuffer_rgba8 = input.color;
        synthetic.motion_rg16f.resize(input.motion.size() / sizeof(std::uint16_t));
        synthetic.depth_r32f.resize(input.depth.size() / sizeof(float));
        std::memcpy(synthetic.motion_rg16f.data(), input.motion.data(), input.motion.size());
        std::memcpy(synthetic.depth_r32f.data(), input.depth.data(), input.depth.size());
      }
      const std::uint32_t distortion_width = std::max(1u, width / 2u);
      const std::uint32_t distortion_height = std::max(1u, height / 2u);
      std::vector<std::uint16_t> distortion_half_res(
          static_cast<std::size_t>(distortion_width) * distortion_height * 4u,
          neuralstudio::fg::FloatToHalf(0.0f));
      std::vector<std::uint8_t> output_poison(
          static_cast<std::size_t>(width) * height * 4u,
          static_cast<std::uint8_t>(output_poison_byte));
      const std::array<std::uint8_t, 4> disable_zero{{0u, 0u, 0u, 0u}};

      GpuTexture backbuffer = CreateUploadedTexture(
          gpu, kColorFormat, width, height, 4, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          synthetic.backbuffer_rgba8.data(), static_cast<std::size_t>(width) * 4u);
      std::optional<GpuTexture> hudless;
      if (use_hudless) {
        hudless = CreateUploadedTexture(
            gpu, kColorFormat, width, height, 4, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            synthetic.hudless_rgba8.data(),
            static_cast<std::size_t>(width) * 4u);
      }
      std::optional<GpuTexture> ui_alpha;
      if (use_ui_alpha) {
        ui_alpha = CreateUploadedTexture(
            gpu, kUiAlphaFormat, width, height, 1, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            synthetic.ui_alpha_r8.data(), width);
      }
      std::optional<GpuTexture> ui_color_alpha;
      if (use_ui_color_alpha) {
        ui_color_alpha = CreateUploadedTexture(
            gpu, kColorFormat, width, height, 4, D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            synthetic.ui_color_alpha_rgba8.data(),
            static_cast<std::size_t>(width) * 4u);
      }
      GpuTexture motion = CreateUploadedTexture(
          gpu, kMotionFormat, width, height, 4, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          synthetic.motion_rg16f.data(), static_cast<std::size_t>(width) * 4u);
      GpuTexture depth = CreateUploadedTexture(
          gpu, kDepthFormat, width, height, 4, D3D12_RESOURCE_FLAG_NONE,
          D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
          synthetic.depth_r32f.data(), static_cast<std::size_t>(width) * 4u);
      std::optional<GpuTexture> distortion;
      if (use_distortion) {
        distortion = CreateUploadedTexture(
            gpu, kDistortionFormat, distortion_width, distortion_height, 8,
            D3D12_RESOURCE_FLAG_NONE,
            D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
            distortion_half_res.data(),
            static_cast<std::size_t>(distortion_width) * 8u);
      }
      GpuTexture output_interpolated = CreateUploadedTexture(
          gpu, kColorFormat, width, height, 4,
          D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
          D3D12_RESOURCE_STATE_UNORDERED_ACCESS, output_poison.data(),
          static_cast<std::size_t>(width) * 4u);
      std::optional<GpuTexture> output_real;
      if (use_output_real) {
        output_real = CreateUploadedTexture(
            gpu, kColorFormat, width, height, 4,
            D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
            synthetic.backbuffer_rgba8.data(),
            static_cast<std::size_t>(width) * 4u);
      }
      std::optional<GpuBuffer> output_disable;
      if (use_disable_output) {
        output_disable = CreateUploadedBuffer(
            gpu, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
            D3D12_RESOURCE_STATE_UNORDERED_ACCESS, disable_zero.data(),
            disable_zero.size());
      }
      ExecuteAndReset(gpu);
      SampleLocalVideoMemory(gpu,
                             "after_frame_resource_upload_" +
                                 std::to_string(frame_id));

      float camera_view_to_clip[4][4];
      float clip_to_camera_view[4][4];
      float clip_to_lens_clip[4][4];
      float clip_to_prev_clip[4][4];
      float prev_clip_to_clip[4][4];
      SetIdentity(camera_view_to_clip);
      SetIdentity(clip_to_camera_view);
      SetIdentity(clip_to_lens_clip);
      SetIdentity(clip_to_prev_clip);
      SetIdentity(prev_clip_to_clip);
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_Backbuffer,
                             backbuffer.resource.Get());
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_MVecs,
                             motion.resource.Get());
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_Depth,
                             depth.resource.Get());
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_HUDLess,
                             hudless ? hudless->resource.Get() : nullptr);
      parameters.SetResource(
          NVSDK_NGX_DLSSG_Parameter_UI,
          ui_color_alpha ? ui_color_alpha->resource.Get() : nullptr);
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_UIAlpha,
                             ui_alpha ? ui_alpha->resource.Get() : nullptr);
      parameters.SetResource(
          NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField,
          distortion ? distortion->resource.Get() : nullptr);
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_OutputInterpolated,
                             output_interpolated.resource.Get());
      parameters.SetResource(NVSDK_NGX_DLSSG_Parameter_OutputReal,
                             output_real ? output_real->resource.Get()
                                         : nullptr);
      parameters.SetResource(
          NVSDK_NGX_DLSSG_Parameter_OutputDisableInterpolation,
          output_disable ? output_disable->resource.Get() : nullptr);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_MultiFrameCount,
                        multiframe_count);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex,
                        multiframe_index);
      parameters.SetU64(NVSDK_NGX_DLSSG_Parameter_BackbufferFrameID, frame_id);
      parameters.SetPointerValue(NVSDK_NGX_DLSSG_Parameter_CameraViewToClip,
                                 camera_view_to_clip);
      parameters.SetPointerValue(NVSDK_NGX_DLSSG_Parameter_ClipToCameraView,
                                 clip_to_camera_view);
      parameters.SetPointerValue(NVSDK_NGX_DLSSG_Parameter_ClipToLensClip,
                                 clip_to_lens_clip);
      parameters.SetPointerValue(NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip,
                                 clip_to_prev_clip);
      parameters.SetPointerValue(NVSDK_NGX_DLSSG_Parameter_PrevClipToClip,
                                 prev_clip_to_clip);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_MvecScaleX,
                        unit_motion_scale ? 1.0f : 1.0f / static_cast<float>(width));
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_MvecScaleY,
                        unit_motion_scale ? 1.0f : 1.0f / static_cast<float>(height));
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetX, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetY, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraPosX, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraPosY, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraPosZ, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraUpX, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraUpY, 1.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraUpZ, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraRightX, 1.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraRightY, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraRightZ, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraFwdX, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraFwdY, 0.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, 1.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraNear, 0.1f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraFar, 1000.0f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraFOV, 1.0471975512f);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio,
                        static_cast<float>(width) / static_cast<float>(height));
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR,
                        color_buffers_hdr ? 1u : 0u);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_DepthInverted, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_Reset,
                        frame_reset ? 1u : 0u);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_AutomodeOverrideReset, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_OrthoProjection, 0);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_MvecInvalidValue, 65504.0f);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_MvecDilated, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_MvecJittered, 0);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_MenuDetectionEnabled,
                        menu_detection ? 1u : 0u);
      parameters.SetF32(NVSDK_NGX_DLSSG_Parameter_LinearizedDepth_Scale, 1.0f);
      parameters.SetF32(
          NVSDK_NGX_DLSSG_Parameter_LinearizedDepth_NearFarPartition, 600.0f);
      parameters.SetF32(
          NVSDK_NGX_DLSSG_Parameter_MinRelativeLinearDepthObjectSeparation,
          40.0f);
      parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_EvalFlags,
                        NVSDK_NGX_DLSSG_EvalFlags_UpdateOnlyInsideExtents);

      const auto set_extent = [&parameters](const char* x_name,
                                            const char* y_name,
                                            const char* width_name,
                                            const char* height_name,
                                            std::uint32_t extent_x,
                                            std::uint32_t extent_y,
                                            std::uint32_t extent_width,
                                            std::uint32_t extent_height) {
        parameters.SetU32(x_name, extent_x);
        parameters.SetU32(y_name, extent_y);
        parameters.SetU32(width_name, extent_width);
        parameters.SetU32(height_name, extent_height);
      };
      neuralstudio::fg::SetBackbufferSubrectContract(
          parameters, {subrect_x, subrect_y, subrect_width, subrect_height});
      std::cout << "contract.legacy_backbuffer_subrect_mirrored=true\n";
      set_extent(NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseX,
                 NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseY,
                 NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth,
                 NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight, subrect_x,
                 subrect_y, subrect_width, subrect_height);
      set_extent(NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseX,
                 NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseY,
                 NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth,
                 NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight, subrect_x,
                 subrect_y, subrect_width, subrect_height);
      if (use_hudless) {
        set_extent(NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseX,
                   NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectBaseY,
                   NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectWidth,
                   NVSDK_NGX_DLSSG_Parameter_HUDLessSubrectHeight,
                   subrect_x, subrect_y, subrect_width, subrect_height);
      }
      if (use_ui_alpha) {
        set_extent(NVSDK_NGX_DLSSG_Parameter_UIAlphaSubrectBaseX,
                   NVSDK_NGX_DLSSG_Parameter_UIAlphaSubrectBaseY,
                   NVSDK_NGX_DLSSG_Parameter_UIAlphaSubrectWidth,
                   NVSDK_NGX_DLSSG_Parameter_UIAlphaSubrectHeight,
                   subrect_x, subrect_y, subrect_width, subrect_height);
      }
      if (use_ui_color_alpha) {
        set_extent(NVSDK_NGX_DLSSG_Parameter_UISubrectBaseX,
                   NVSDK_NGX_DLSSG_Parameter_UISubrectBaseY,
                   NVSDK_NGX_DLSSG_Parameter_UISubrectWidth,
                   NVSDK_NGX_DLSSG_Parameter_UISubrectHeight, subrect_x,
                   subrect_y, subrect_width, subrect_height);
      }
      if (use_distortion) {
        set_extent(
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionFieldSubrectBaseX,
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionFieldSubrectBaseY,
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionFieldSubrectWidth,
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionFieldSubrectHeight,
            subrect_x / 2u, subrect_y / 2u,
            std::max(1u, subrect_width / 2u),
            std::max(1u, subrect_height / 2u));
      }
      if (use_distortion) {
        parameters.SetU32(
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField_LowPrecision_IsLowPrecision,
            0);
        parameters.SetF32(
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField_LowPrecision_Bias,
            0.0f);
        parameters.SetF32(
            NVSDK_NGX_DLSSG_Parameter_BidirectionalDistortionField_LowPrecision_Scale,
            1.0f);
      }
      set_extent(NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseX,
                 NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseY,
                 NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectWidth,
                 NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectHeight,
                 subrect_x, subrect_y, subrect_width, subrect_height);
      if (use_output_real) {
        set_extent(NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseX,
                   NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectBaseY,
                   NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectWidth,
                   NVSDK_NGX_DLSSG_Parameter_OutputRealSubrectHeight,
                   subrect_x, subrect_y, subrect_width, subrect_height);
      }

      const auto evaluate_start = std::chrono::steady_clock::now();
      const std::string evaluate_phase =
          "evaluate_" + std::to_string(frame_id);
      set_trace_phase(evaluate_phase.c_str());
      const NVSDK_NGX_Result evaluate_result = thunk_api.evaluate(
          runtime_api.evaluate, gpu.list.Get(), feature_handle,
          parameters.AsNgx(), nullptr);
      const std::string evaluate_end_phase = evaluate_phase + ".end";
      set_trace_phase(evaluate_end_phase.c_str());
      PrintNgx("runtime.evaluate_result", evaluate_result);
      if (NVSDK_NGX_FAILED(evaluate_result)) {
        throw std::runtime_error("DLSSG evaluation failed");
      }
      ExecuteAndReset(gpu);
      SampleLocalVideoMemory(gpu,
                             "after_evaluate_" + std::to_string(frame_id));
      const auto evaluate_end = std::chrono::steady_clock::now();
      if (!input_sequence.empty()) {
        const auto& input = input_sequence.at(frame_id);
        for (auto pair : {std::pair{&backbuffer, &input.color}, std::pair{&motion, &input.motion}, std::pair{&depth, &input.depth}}) {
          const auto prior_state = pair.first->state;
          if (ReadTexture(gpu, *pair.first) != *pair.second) throw std::runtime_error("external GPU input mutated during evaluate");
          Transition(gpu.list.Get(), pair.first->resource.Get(), pair.first->state, prior_state);
          pair.first->state = prior_state;
          ExecuteAndReset(gpu);
        }
      }
      if (capture_runtime_default_buffers) {
        const bool post_reset_capture =
            frame_id == 0 &&
            runtime_default_capture_point ==
                "post_evaluate_0_fence_pre_evaluate_1";
        const bool post_generated_capture =
            frame_id == 1 &&
            runtime_default_capture_point ==
                "post_evaluate_1_fence_pre_evaluate_2";
        if (post_reset_capture || post_generated_capture) {
          capture_default_buffers();
        }
      }

      std::vector<std::uint8_t> output_bytes =
          ReadTexture(gpu, output_interpolated);
      std::vector<std::uint8_t> output_real_bytes;
      if (output_real) output_real_bytes = ReadTexture(gpu, *output_real);
      std::vector<std::uint8_t> disable_bytes;
      if (output_disable) disable_bytes = ReadBuffer(gpu, *output_disable);
      if (output_bytes.size() != synthetic.backbuffer_rgba8.size() ||
          (output_real &&
           output_real_bytes.size() != synthetic.backbuffer_rgba8.size()) ||
          (output_disable && disable_bytes.size() < 4u)) {
        throw std::runtime_error("DLSSG output readback size mismatch");
      }

      const std::string suffix = std::to_string(frame_id);
      WriteBinary(output_directory / ("input-backbuffer-" + suffix + ".rgba8"),
                  synthetic.backbuffer_rgba8);
      if (use_hudless) {
        WriteBinary(output_directory /
                        ("input-hudless-" + suffix + ".rgba8"),
                    synthetic.hudless_rgba8);
      }
      if (use_ui_alpha) {
        WriteBinary(output_directory / ("input-ui-alpha-" + suffix + ".r8"),
                    synthetic.ui_alpha_r8);
      }
      if (use_ui_color_alpha) {
        WriteBinary(output_directory /
                        ("input-ui-color-alpha-" + suffix + ".rgba8"),
                    synthetic.ui_color_alpha_rgba8);
      }
      WriteBinary(output_directory / ("input-motion-" + suffix + ".rg16f"),
                  synthetic.motion_rg16f);
      WriteBinary(output_directory / ("input-depth-" + suffix + ".r32f"),
                  synthetic.depth_r32f);
      if (use_distortion) {
        WriteBinary(output_directory /
                        ("input-distortion-half-" + suffix + ".rgba16f"),
                    distortion_half_res);
      }
      WriteBinary(output_directory /
                      ("output-interpolated-" + suffix + ".rgba8"),
                  output_bytes);
      if (output_real) {
        WriteBinary(output_directory / ("output-real-" + suffix + ".rgba8"),
                    output_real_bytes);
      }
      if (output_disable) {
        WriteBinary(output_directory / ("output-disable-" + suffix + ".bin"),
                    disable_bytes);
      }

      FrameReceipt receipt;
      receipt.frame_id = frame_id;
      receipt.reset = frame_reset;
      receipt.evaluate_result = evaluate_result;
      receipt.evaluate_ms =
          std::chrono::duration<double, std::milli>(evaluate_end -
                                                    evaluate_start)
              .count();
      receipt.input_sha256 = Sha256(synthetic.backbuffer_rgba8);
      receipt.output_interpolated_sha256 = Sha256(output_bytes);
      receipt.output_real_provided = output_real.has_value();
      receipt.disable_output_provided = output_disable.has_value();
      receipt.output_real_sha256 =
          output_real ? Sha256(output_real_bytes) : "not_provided";
      receipt.disable_sha256 =
          output_disable ? Sha256(disable_bytes) : "not_provided";
      receipt.disable_interpolation =
          output_disable ? disable_bytes[0] != 0 : false;
      receipt.output_nonzero_bytes = static_cast<std::size_t>(std::count_if(
          output_bytes.begin(), output_bytes.end(),
          [](std::uint8_t byte) { return byte != 0u; }));
      receipt.output_untouched_pixels = CountPixelsAllValue(
          output_bytes, static_cast<std::uint8_t>(output_poison_byte), width,
          subrect_x, subrect_y, subrect_width, subrect_height);
      receipt.differs_from_current = RegionDiffers(
          output_bytes, synthetic.backbuffer_rgba8, width, subrect_x,
          subrect_y, subrect_width, subrect_height);
      receipt.differs_from_previous =
          frame_id == 0 ||
          RegionDiffers(output_bytes, previous_input, width, subrect_x,
                        subrect_y, subrect_width, subrect_height);
      receipt.generated_output_accepted =
          receipt.reset ||
          ((!output_disable || !receipt.disable_interpolation) &&
           RegionContainsNonzero(output_bytes, width, subrect_x, subrect_y,
                                 subrect_width, subrect_height) &&
           !RegionAllValue(output_bytes,
                           static_cast<std::uint8_t>(output_poison_byte), width,
                           subrect_x, subrect_y, subrect_width,
                           subrect_height) &&
           receipt.differs_from_previous && receipt.differs_from_current);
      if (!receipt.reset && !receipt.generated_output_accepted) {
        all_generated_outputs_accepted = false;
      }
      std::cout << "frame.id=" << frame_id << '\n';
      std::cout << "frame.reset=" << (receipt.reset ? "true" : "false")
                << '\n';
      std::cout << "frame.input_sha256=" << receipt.input_sha256 << '\n';
      std::cout << "frame.output_interpolated_sha256="
                << receipt.output_interpolated_sha256 << '\n';
      std::cout << "frame.output_real_sha256=" << receipt.output_real_sha256
                << '\n';
      std::cout << "frame.disable_sha256=" << receipt.disable_sha256 << '\n';
      std::cout << "frame.disable_output_provided="
                << (receipt.disable_output_provided ? "true" : "false")
                << '\n';
      std::cout << "frame.disable_interpolation="
                << (receipt.disable_interpolation ? "true" : "false")
                << '\n';
      std::cout << "frame.generated_output_accepted="
                << (receipt.generated_output_accepted ? "true" : "false")
                << '\n';
      std::cout << "frame.output_untouched_pixels="
                << receipt.output_untouched_pixels << '\n';
      previous_input = synthetic.backbuffer_rgba8;
      frame_receipts.push_back(std::move(receipt));
    }

    AllocationCaptureReceipt allocation_capture_receipt;
    allocation_capture_receipt.enabled = capture_runtime_allocations;
    allocation_capture_receipt.upload_capture_enabled =
        capture_runtime_upload_buffers;
    allocation_capture_receipt.expected_upload_dump_count =
        expected_runtime_upload_buffer_count;
    allocation_capture_receipt.expected_upload_dump_bytes =
        expected_runtime_upload_bytes;
    allocation_capture_receipt.default_capture_enabled =
        capture_runtime_default_buffers;
    allocation_capture_receipt.default_capture_point =
        runtime_default_capture_point;
    allocation_capture_receipt.default_manifest =
        capture_runtime_default_buffers
            ? WideToUtf8(runtime_default_manifest.filename().native())
            : "not_requested";
    allocation_capture_receipt.expected_default_dump_count =
        expected_runtime_default_buffer_count;
    allocation_capture_receipt.expected_default_dump_bytes =
        expected_runtime_default_buffer_bytes;
    allocation_capture_receipt.expected_baseline_sha256 =
        expected_baseline_output_sha256;
    if (runtime_allocation_capture) {
      if (frame_receipts.size() < 2u) {
        throw std::runtime_error(
            "allocation capture produced no generated-frame receipt");
      }
      allocation_capture_receipt.baseline_output_match =
          frame_receipts[1].output_interpolated_sha256 ==
          expected_baseline_output_sha256;
      std::cout << "runtime.allocation_capture_baseline_output_match="
                << (allocation_capture_receipt.baseline_output_match ? "true"
                                                                     : "false")
                << '\n';
    }

    WriteLoadedModuleInventory(output_directory / L"module-identities.json");
    cleanup.Finish();
    if (runtime_allocation_capture) {
      runtime_allocation_capture->Finalize();
      allocation_capture_receipt.healthy =
          runtime_allocation_capture->healthy();
      allocation_capture_receipt.allocations =
          runtime_allocation_capture->allocation_count();
      allocation_capture_receipt.releases =
          runtime_allocation_capture->release_count();
      allocation_capture_receipt.leaks =
          runtime_allocation_capture->leak_count();
      allocation_capture_receipt.semantic_ledger_sha256 =
          Sha256(ReadFile(allocation_semantic_path));
      allocation_capture_receipt.volatile_ledger_sha256 =
          Sha256(ReadFile(allocation_volatile_path));
      if (capture_runtime_upload_buffers) {
        allocation_capture_receipt.upload_dump_count =
            runtime_allocation_capture->upload_dump_count();
        allocation_capture_receipt.upload_dump_bytes =
            runtime_allocation_capture->upload_dump_bytes();
        allocation_capture_receipt.upload_manifest_sha256 =
            Sha256(ReadFile(runtime_upload_manifest));
        if (allocation_capture_receipt.upload_dump_count == 0) {
          allocation_capture_receipt.healthy = false;
        }
      }
      if (capture_runtime_default_buffers) {
        allocation_capture_receipt.default_dump_count =
            runtime_allocation_capture->default_dump_count();
        allocation_capture_receipt.default_dump_bytes =
            runtime_allocation_capture->default_dump_bytes();
        allocation_capture_receipt.default_manifest_sha256 =
            Sha256(ReadFile(runtime_default_manifest));
        if (allocation_capture_receipt.default_dump_count == 0) {
          allocation_capture_receipt.healthy = false;
        }
      }
      std::cout << "runtime.allocation_capture_healthy="
                << (allocation_capture_receipt.healthy ? "true" : "false")
                << '\n';
      std::cout << "runtime.allocation_count="
                << allocation_capture_receipt.allocations << '\n';
      std::cout << "runtime.allocation_release_count="
                << allocation_capture_receipt.releases << '\n';
      std::cout << "runtime.allocation_leak_count="
                << allocation_capture_receipt.leaks << '\n';
    }
    ngx_log_sink.Close();
    const bool ngx_log_write_failed = ngx_log_sink.write_failed();
    const std::string ngx_callback_log_sha256 =
        Sha256(ReadFile(ngx_log_sink.path()));
    std::cout << "runtime.ngx_callback_log_sha256="
              << ngx_callback_log_sha256 << '\n';
    std::cout << "runtime.ngx_callback_log_write_failed="
              << (ngx_log_write_failed ? "true" : "false") << '\n';
    WriteDirectoryInventory(output_directory / L"ngx-cache-after.json",
                            data_directory, "after");
    const std::uint64_t severe_messages =
        CountSevereDebugMessages(gpu.device.Get());
    const bool accepted = all_generated_outputs_accepted &&
                          severe_messages == 0 &&
                          !ngx_log_write_failed &&
                          allocation_capture_receipt.healthy &&
                          allocation_capture_receipt.baseline_output_match &&
                          NVSDK_NGX_SUCCEED(cleanup.release_result) &&
                          NVSDK_NGX_SUCCEED(cleanup.shutdown_result);
    WriteScenarioSnapshot(
        output_directory / L"scenario-after.json", "after", width, height,
        rendered_frames, subrect_x, subrect_y, subrect_width, subrect_height,
        multiframe_count, multiframe_index, resource_profile,
        output_poison_byte, accepted);
    WriteReceipt(output_directory / L"fg-run-receipt.json", runtime_path,
                 thunk_path, runtime_sha256, thunk_sha256, run_nonce,
                 queue_nonce, driver_version, loaded_nvapi_proxy_path_utf8,
                 loaded_nvapi_proxy_sha256_value, ngx_callback_log_sha256,
                 gpu, application_id, api_version, snippet_version, width,
                 height, subrect_x,
                 subrect_y, subrect_width, subrect_height, multiframe_count,
                 multiframe_index, resource_profile, output_poison_byte,
                 estimated_vram,
                 create_result, cleanup.release_result,
                 cleanup.shutdown_result, frame_receipts,
                 allocation_capture_receipt, severe_messages, accepted,
                 input_sequence_sha256, motion_scale_contract);
    std::cout << "receipt.path=\""
              << WideToUtf8((output_directory / L"fg-run-receipt.json").native())
              << "\"\n";
    std::cout << "result=" << (accepted ? "observed" : "rejected") << '\n';
    std::cout << "claim=standalone_runtime_execution_only\n";
    return accepted ? 0 : 20;
  } catch (const std::exception& error) {
    std::cerr << "error=fg_host_failed message=\""
              << JsonEscape(error.what()) << "\"\n";
    std::cerr << "result=rejected\n";
    std::cerr << "claim=none\n";
    return 10;
  }
}
