#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <bit>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#if defined(NEURALSTUDIO_CUDA_DETOURS)
#include <detours.h>
#endif

// CUDA's redistributable headers are deliberately not required here. This
// proxy covers only the 30 Driver API symbols selected by the hash-pinned
// DLSSNR runtime and preserves their x64 ABI while forwarding to System32.
using CUdeviceptr = uint64_t;
using CUresult = int;

constexpr CUresult kCudaSuccess = 0;
constexpr CUresult kCudaErrorInvalidValue = 1;
constexpr CUresult kCudaErrorNotFound = 500;
constexpr size_t kMaxAllocations = 64;
constexpr size_t kMaxModules = 256;
constexpr size_t kMaxFunctions = 512;
constexpr size_t kMaxArrays = 64;
constexpr size_t kDeltaPageBytes = 64 * 1024;

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
      CUdeviceptr device_pointer;
      int format;
      unsigned int channels;
      size_t size_bytes;
    } linear;
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

struct CudaMemcpy2D {
  size_t source_x_bytes;
  size_t source_y;
  int source_memory_type;
  const void* source_host;
  CUdeviceptr source_device;
  void* source_array;
  size_t source_pitch;
  size_t destination_x_bytes;
  size_t destination_y;
  int destination_memory_type;
  void* destination_host;
  CUdeviceptr destination_device;
  void* destination_array;
  size_t destination_pitch;
  size_t width_bytes;
  size_t height;
};

struct AllocationRecord {
  CUdeviceptr pointer;
  size_t bytes;
  LONG id;
  bool active;
};

struct ModuleRecord {
  void* handle;
  LONG id;
  bool active;
};

struct FunctionRecord {
  void* handle;
  LONG id;
  LONG module_id;
  char name[256];
  bool active;
};

struct ArrayRecord {
  void* handle;
  LONG id;
  size_t width;
  size_t height;
  int format;
  unsigned int channels;
  bool active;
};

extern "C" {
__declspec(dllexport) CUresult WINAPI cuArrayCreate(void**, const void*);
__declspec(dllexport) CUresult WINAPI cuArray3DCreate(void**, const void*);
__declspec(dllexport) CUresult WINAPI cuArrayDestroy(void*);
__declspec(dllexport) CUresult WINAPI cuArrayGetDescriptor(void*, void*);
__declspec(dllexport) CUresult WINAPI cuCtxGetDevice(int*);
__declspec(dllexport) CUresult WINAPI cuCtxSetCurrent(void*);
__declspec(dllexport) CUresult WINAPI cuCtxSynchronize();
__declspec(dllexport) CUresult WINAPI cuDestroyExternalMemory(void*);
__declspec(dllexport) CUresult WINAPI cuDeviceGetAttribute(int*, int, int);
__declspec(dllexport) CUresult WINAPI cuDeviceGetLuid(char*, unsigned int*, int);
__declspec(dllexport) CUresult WINAPI cuDeviceGetUuid(void*, int);
__declspec(dllexport) CUresult WINAPI cuFuncSetAttribute(void*, int, int);
__declspec(dllexport) CUresult WINAPI cuGetErrorString(CUresult, const char**);
__declspec(dllexport) CUresult WINAPI cuLaunchKernel(
    void*, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, unsigned int, unsigned int, void*, void**, void**);
__declspec(dllexport) CUresult WINAPI cuMemAlloc(CUdeviceptr*, size_t);
__declspec(dllexport) CUresult WINAPI cuMemAllocHost(void**, size_t);
__declspec(dllexport) CUresult WINAPI cuMemFree(CUdeviceptr);
__declspec(dllexport) CUresult WINAPI cuMemFreeHost(void*);
__declspec(dllexport) CUresult WINAPI cuMemcpy2D(const void*);
__declspec(dllexport) CUresult WINAPI cuMemcpyHtoDAsync(
    CUdeviceptr, const void*, size_t, void*);
__declspec(dllexport) CUresult WINAPI cuMemsetD32Async(
    CUdeviceptr, unsigned int, size_t, void*);
__declspec(dllexport) CUresult WINAPI cuMipmappedArrayDestroy(void*);
__declspec(dllexport) CUresult WINAPI cuModuleGetFunction(
    void**, void*, const char*);
__declspec(dllexport) CUresult WINAPI cuModuleLoadData(void**, const void*);
__declspec(dllexport) CUresult WINAPI cuModuleUnload(void*);
__declspec(dllexport) CUresult WINAPI cuSurfObjectCreate(uint64_t*, const void*);
__declspec(dllexport) CUresult WINAPI cuSurfObjectDestroy(uint64_t);
__declspec(dllexport) CUresult WINAPI cuSurfObjectGetResourceDesc(
    void*, uint64_t);
__declspec(dllexport) CUresult WINAPI cuTexObjectCreate(
    uint64_t*, const void*, const void*, const void*);
__declspec(dllexport) CUresult WINAPI cuTexObjectDestroy(uint64_t);
__declspec(dllexport) CUresult WINAPI cuTexObjectGetResourceDesc(
    void*, uint64_t);
__declspec(dllexport) CUresult WINAPI cuGetProcAddress(
    const char*, void**, int, uint64_t, void*);
__declspec(dllexport) CUresult WINAPI cuInit(unsigned int);
__declspec(dllexport) CUresult WINAPI cuDeviceGet(int*, int);
__declspec(dllexport) CUresult WINAPI cuDevicePrimaryCtxRetain(void**, int);
__declspec(dllexport) CUresult WINAPI cuDevicePrimaryCtxRelease(int);
__declspec(dllexport) void WINAPI NeuralStudioCudaTraceSetPhase(const char*);
__declspec(dllexport) BOOL WINAPI NeuralStudioInstallCudaHooks();
__declspec(dllexport) BOOL WINAPI NeuralStudioRemoveCudaHooks();
}

namespace {

struct ForwardEntry {
  const char* name;
  FARPROC wrapper;
  FARPROC real;
};

template <typename Destination, typename Source>
Destination PointerCast(Source source) {
  static_assert(sizeof(Destination) == sizeof(Source));
  Destination destination{};
  memcpy(&destination, &source, sizeof(destination));
  return destination;
}

#define FORWARD_ENTRY(name) {#name, PointerCast<FARPROC>(&name), nullptr}
ForwardEntry g_entries[] = {
    FORWARD_ENTRY(cuArrayCreate),
    FORWARD_ENTRY(cuArray3DCreate),
    FORWARD_ENTRY(cuArrayDestroy),
    FORWARD_ENTRY(cuArrayGetDescriptor),
    FORWARD_ENTRY(cuCtxGetDevice),
    FORWARD_ENTRY(cuCtxSetCurrent),
    FORWARD_ENTRY(cuCtxSynchronize),
    FORWARD_ENTRY(cuDestroyExternalMemory),
    FORWARD_ENTRY(cuDeviceGetAttribute),
    FORWARD_ENTRY(cuDeviceGetLuid),
    FORWARD_ENTRY(cuDeviceGetUuid),
    FORWARD_ENTRY(cuFuncSetAttribute),
    FORWARD_ENTRY(cuGetErrorString),
    FORWARD_ENTRY(cuLaunchKernel),
    FORWARD_ENTRY(cuMemAlloc),
    FORWARD_ENTRY(cuMemAllocHost),
    FORWARD_ENTRY(cuMemFree),
    FORWARD_ENTRY(cuMemFreeHost),
    FORWARD_ENTRY(cuMemcpy2D),
    FORWARD_ENTRY(cuMemcpyHtoDAsync),
    FORWARD_ENTRY(cuMemsetD32Async),
    FORWARD_ENTRY(cuMipmappedArrayDestroy),
    FORWARD_ENTRY(cuModuleGetFunction),
    FORWARD_ENTRY(cuModuleLoadData),
    FORWARD_ENTRY(cuModuleUnload),
    FORWARD_ENTRY(cuSurfObjectCreate),
    FORWARD_ENTRY(cuSurfObjectDestroy),
    FORWARD_ENTRY(cuSurfObjectGetResourceDesc),
    FORWARD_ENTRY(cuTexObjectCreate),
    FORWARD_ENTRY(cuTexObjectDestroy),
    FORWARD_ENTRY(cuTexObjectGetResourceDesc),
};
#undef FORWARD_ENTRY

HMODULE g_self = nullptr;
HMODULE g_real_driver = nullptr;
INIT_ONCE g_driver_once = INIT_ONCE_STATIC_INIT;
CRITICAL_SECTION g_lock;
LONG g_next_allocation_id = 0;
LONG g_next_module_id = 0;
LONG g_next_function_id = 0;
LONG g_next_array_id = 0;
LONG g_next_transfer_id = 0;
LONG g_next_launch_id = 0;
AllocationRecord g_allocations[kMaxAllocations]{};
ModuleRecord g_modules[kMaxModules]{};
FunctionRecord g_functions[kMaxFunctions]{};
ArrayRecord g_arrays[kMaxArrays]{};
unsigned char* g_snapshot_current = nullptr;
unsigned char* g_snapshot_previous = nullptr;
size_t g_snapshot_bytes = 0;
LONG g_snapshot_allocation_id = 0;
using RealGetProcAddress = CUresult(WINAPI*)(
    const char*, void**, int, uint64_t, void*);
using WinGetProcAddress = FARPROC(WINAPI*)(HMODULE, LPCSTR);
RealGetProcAddress g_real_get_proc_address = nullptr;
WinGetProcAddress g_real_win_get_proc_address = nullptr;
bool g_cuda_hooks_installed = false;
struct IatPatch {
  PVOID* slot;
  PVOID original;
};
IatPatch g_iat_patches[8]{};
size_t g_iat_patch_count = 0;
HMODULE g_nvngx_core = nullptr;

bool EnvironmentEnabled(const char* name) {
  char value[8]{};
  const DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
  return length > 0 && length < sizeof(value) && value[0] == '1';
}

void TraceLine(const char* line) {
  char path[MAX_PATH]{};
  const DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_CUDA_CAPTURE_TRACE", path, static_cast<DWORD>(sizeof(path)));
  if (length == 0 || length >= sizeof(path)) return;
  EnterCriticalSection(&g_lock);
  FILE* output = nullptr;
  if (fopen_s(&output, path, "ab") == 0 && output != nullptr) {
    fputs(line, output);
    fputc('\n', output);
    fclose(output);
  }
  LeaveCriticalSection(&g_lock);
}

void TraceCall(const char* call, const char* fields) {
  char line[1400]{};
  if (fields != nullptr && fields[0] != '\0') {
    sprintf_s(line, "{\"call\":\"%s\",%s}", call, fields);
  } else {
    sprintf_s(line, "{\"call\":\"%s\"}", call);
  }
  TraceLine(line);
}

void EnsureDirectoryFor(const char* environment_name, char* output,
                        size_t output_size) {
  const DWORD length = GetEnvironmentVariableA(
      environment_name, output, static_cast<DWORD>(output_size));
  if (length == 0 || length >= output_size) {
    output[0] = '\0';
    return;
  }
  CreateDirectoryA(output, nullptr);
}

void DumpBytes(const char* kind, LONG id, const void* data, size_t bytes) {
  if (data == nullptr || bytes == 0) return;
  char directory[MAX_PATH]{};
  EnsureDirectoryFor("NEURALSTUDIO_CUDA_CAPTURE_DUMP_DIR", directory,
                     sizeof(directory));
  if (directory[0] == '\0') return;
  char path[MAX_PATH]{};
  sprintf_s(path, "%s\\%s-%04ld.bin", directory, kind, id);
  FILE* output = nullptr;
  if (fopen_s(&output, path, "wb") == 0 && output != nullptr) {
    fwrite(data, 1, bytes, output);
    fclose(output);
  }
}

BOOL CALLBACK LoadRealDriver(PINIT_ONCE, PVOID, PVOID*) {
  wchar_t path[MAX_PATH]{};
  DWORD length = GetEnvironmentVariableW(
      L"NEURALSTUDIO_REAL_NVCUDA", path,
      static_cast<DWORD>(sizeof(path) / sizeof(path[0])));
  if (length == 0 || length >= sizeof(path) / sizeof(path[0])) {
    length = GetSystemDirectoryW(path,
        static_cast<UINT>(sizeof(path) / sizeof(path[0])));
    if (length == 0 || length + 12 >= sizeof(path) / sizeof(path[0])) {
      return TRUE;
    }
    wcscat_s(path, L"\\nvcuda.dll");
  }
  g_real_driver = LoadLibraryW(path);
  if (g_real_driver == g_self) g_real_driver = nullptr;
  return TRUE;
}

bool EnsureRealDriver() {
  InitOnceExecuteOnce(&g_driver_once, LoadRealDriver, nullptr, nullptr);
  return g_real_driver != nullptr;
}

ForwardEntry* FindEntry(const char* name) {
  if (name == nullptr) return nullptr;
  for (auto& entry : g_entries) {
    if (strcmp(entry.name, name) == 0) return &entry;
  }
  return nullptr;
}

ForwardEntry* FindVersionedEntry(const char* name) {
  ForwardEntry* exact = FindEntry(name);
  if (exact != nullptr) return exact;
  static constexpr const char* suffixes[] = {
      "_v2", "_v3", "_ptsz", "_ptds", "_v2_ptsz", "_v2_ptds"};
  char candidate[160]{};
  for (auto& entry : g_entries) {
    for (const char* suffix : suffixes) {
      sprintf_s(candidate, "%s%s", entry.name, suffix);
      if (strcmp(candidate, name) == 0) return &entry;
    }
  }
  return nullptr;
}

FARPROC OriginalGetProcAddress(HMODULE module, const char* name) {
  return g_real_win_get_proc_address != nullptr
      ? g_real_win_get_proc_address(module, name)
      : GetProcAddress(module, name);
}

FARPROC WINAPI HookWinGetProcAddress(HMODULE module, LPCSTR name) {
  if (name == nullptr || reinterpret_cast<uintptr_t>(name) <= 0xFFFF) {
    return g_real_win_get_proc_address(module, name);
  }
  FARPROC result = g_real_win_get_proc_address(module, name);
  if (module != g_real_driver) return result;
  ForwardEntry* entry = FindVersionedEntry(name);
  char fields[640]{};
  sprintf_s(fields,
            "\"symbol\":\"%s\",\"intercepted\":%s,"
            "\"resolved\":%s",
            name, entry != nullptr ? "true" : "false",
            result != nullptr ? "true" : "false");
  TraceCall("win_get_proc_address", fields);
  if (entry != nullptr && result != nullptr) {
    entry->real = result;
    return entry->wrapper;
  }
  if ((strcmp(name, "cuGetProcAddress") == 0 ||
       strcmp(name, "cuGetProcAddress_v2") == 0) &&
      result != nullptr) {
    return PointerCast<FARPROC>(&cuGetProcAddress);
  }
  return result;
}

bool PatchModuleGetProcAddress(HMODULE module) {
  if (module == nullptr) return false;
  auto* image = reinterpret_cast<unsigned char*>(module);
  auto* dos = reinterpret_cast<IMAGE_DOS_HEADER*>(image);
  if (dos->e_magic != IMAGE_DOS_SIGNATURE) return false;
  auto* nt = reinterpret_cast<IMAGE_NT_HEADERS64*>(image + dos->e_lfanew);
  if (nt->Signature != IMAGE_NT_SIGNATURE ||
      nt->OptionalHeader.Magic != IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
    return false;
  }
  const IMAGE_DATA_DIRECTORY imports_directory =
      nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
  if (imports_directory.VirtualAddress == 0) return false;
  auto* descriptor = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(
      image + imports_directory.VirtualAddress);
  for (; descriptor->Name != 0; ++descriptor) {
    if (descriptor->OriginalFirstThunk == 0 || descriptor->FirstThunk == 0) {
      continue;
    }
    auto* names = reinterpret_cast<IMAGE_THUNK_DATA64*>(
        image + descriptor->OriginalFirstThunk);
    auto* addresses = reinterpret_cast<IMAGE_THUNK_DATA64*>(
        image + descriptor->FirstThunk);
    for (; names->u1.AddressOfData != 0; ++names, ++addresses) {
      if (IMAGE_SNAP_BY_ORDINAL64(names->u1.Ordinal)) continue;
      auto* import = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(
          image + names->u1.AddressOfData);
      if (strcmp(reinterpret_cast<const char*>(import->Name),
                 "GetProcAddress") != 0) {
        continue;
      }
      if (g_iat_patch_count >=
          sizeof(g_iat_patches) / sizeof(g_iat_patches[0])) {
        return false;
      }
      auto** slot = reinterpret_cast<PVOID*>(&addresses->u1.Function);
      for (size_t index = 0; index < g_iat_patch_count; ++index) {
        if (g_iat_patches[index].slot == slot) return true;
      }
      DWORD previous = 0;
      if (!VirtualProtect(slot, sizeof(*slot), PAGE_READWRITE, &previous)) {
        return false;
      }
      IatPatch& patch = g_iat_patches[g_iat_patch_count++];
      patch.slot = slot;
      patch.original = *slot;
      *slot = PointerCast<PVOID>(&HookWinGetProcAddress);
      DWORD ignored = 0;
      VirtualProtect(slot, sizeof(*slot), previous, &ignored);
      FlushInstructionCache(GetCurrentProcess(), slot, sizeof(*slot));
      return true;
    }
  }
  return false;
}

bool RestoreIatPatches() {
  bool result = true;
  while (g_iat_patch_count > 0) {
    IatPatch& patch = g_iat_patches[--g_iat_patch_count];
    DWORD previous = 0;
    if (!VirtualProtect(patch.slot, sizeof(*patch.slot), PAGE_READWRITE,
                        &previous)) {
      result = false;
      continue;
    }
    *patch.slot = patch.original;
    DWORD ignored = 0;
    VirtualProtect(patch.slot, sizeof(*patch.slot), previous, &ignored);
    FlushInstructionCache(GetCurrentProcess(), patch.slot,
                          sizeof(*patch.slot));
  }
  return result;
}

FARPROC RealFunction(const char* name) {
  ForwardEntry* entry = FindEntry(name);
  if (entry != nullptr && entry->real != nullptr) return entry->real;
  if (!EnsureRealDriver()) return nullptr;
  FARPROC result = OriginalGetProcAddress(g_real_driver, name);
  if (result == nullptr) {
    char versioned[128]{};
    sprintf_s(versioned, "%s_v2", name);
    result = OriginalGetProcAddress(g_real_driver, versioned);
  }
  if (entry != nullptr) entry->real = result;
  return result;
}

template <typename Function>
Function Real(const char* name) {
  return PointerCast<Function>(RealFunction(name));
}

AllocationRecord* FindAllocation(CUdeviceptr pointer) {
  for (auto& allocation : g_allocations) {
    if (allocation.active && allocation.pointer == pointer) return &allocation;
  }
  return nullptr;
}

AllocationRecord* AddAllocation(CUdeviceptr pointer, size_t bytes) {
  for (auto& allocation : g_allocations) {
    if (!allocation.active) {
      allocation.pointer = pointer;
      allocation.bytes = bytes;
      allocation.id = InterlockedIncrement(&g_next_allocation_id);
      allocation.active = true;
      return &allocation;
    }
  }
  return nullptr;
}

ModuleRecord* FindModule(void* handle) {
  for (auto& module : g_modules) {
    if (module.active && module.handle == handle) return &module;
  }
  return nullptr;
}

ModuleRecord* AddModule(void* handle) {
  for (auto& module : g_modules) {
    if (!module.active) {
      module.handle = handle;
      module.id = InterlockedIncrement(&g_next_module_id);
      module.active = true;
      return &module;
    }
  }
  return nullptr;
}

FunctionRecord* FindFunction(void* handle) {
  for (auto& function : g_functions) {
    if (function.active && function.handle == handle) return &function;
  }
  return nullptr;
}

FunctionRecord* AddFunction(void* handle, LONG module_id, const char* name) {
  for (auto& function : g_functions) {
    if (!function.active) {
      function.handle = handle;
      function.id = InterlockedIncrement(&g_next_function_id);
      function.module_id = module_id;
      strncpy_s(function.name, name, _TRUNCATE);
      function.active = true;
      return &function;
    }
  }
  return nullptr;
}

ArrayRecord* FindArray(void* handle) {
  for (auto& array : g_arrays) {
    if (array.active && array.handle == handle) return &array;
  }
  return nullptr;
}

ArrayRecord* AddArray(void* handle, const CudaArrayDescriptor* descriptor) {
  for (auto& array : g_arrays) {
    if (!array.active) {
      array.handle = handle;
      array.id = InterlockedIncrement(&g_next_array_id);
      array.width = descriptor->width;
      array.height = descriptor->height;
      array.format = descriptor->format;
      array.channels = descriptor->channels;
      array.active = true;
      return &array;
    }
  }
  return nullptr;
}

size_t FatbinBytes(const void* image) {
  if (image == nullptr) return 0;
  const auto* bytes = static_cast<const unsigned char*>(image);
  uint32_t magic = 0;
  uint16_t header_bytes = 0;
  uint64_t payload_bytes = 0;
  memcpy(&magic, bytes, sizeof(magic));
  memcpy(&header_bytes, bytes + 6, sizeof(header_bytes));
  memcpy(&payload_bytes, bytes + 8, sizeof(payload_bytes));
  if (magic != 0xBA55ED50u || header_bytes < 16 ||
      payload_bytes > 64ull * 1024ull * 1024ull) {
    return 0;
  }
  return static_cast<size_t>(header_bytes + payload_bytes);
}

AllocationRecord* SnapshotTarget() {
  char requested_text[32]{};
  const DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_CUDA_SNAPSHOT_ALLOCATION", requested_text,
      static_cast<DWORD>(sizeof(requested_text)));
  const LONG requested = length > 0 && length < sizeof(requested_text)
      ? strtol(requested_text, nullptr, 10)
      : 0;
  AllocationRecord* best = nullptr;
  for (auto& allocation : g_allocations) {
    if (!allocation.active) continue;
    if (requested > 0 && allocation.id == requested) return &allocation;
    if (requested == 0 && (best == nullptr || allocation.bytes > best->bytes)) {
      best = &allocation;
    }
  }
  return requested > 0 ? nullptr : best;
}

void ResetSnapshotBuffers(size_t bytes, LONG allocation_id) {
  if (g_snapshot_bytes == bytes &&
      g_snapshot_allocation_id == allocation_id &&
      g_snapshot_current != nullptr && g_snapshot_previous != nullptr) {
    return;
  }
  if (g_snapshot_current != nullptr) VirtualFree(g_snapshot_current, 0, MEM_RELEASE);
  if (g_snapshot_previous != nullptr) VirtualFree(g_snapshot_previous, 0, MEM_RELEASE);
  g_snapshot_current = static_cast<unsigned char*>(VirtualAlloc(
      nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  g_snapshot_previous = static_cast<unsigned char*>(VirtualAlloc(
      nullptr, bytes, MEM_RESERVE | MEM_COMMIT, PAGE_READWRITE));
  g_snapshot_bytes =
      g_snapshot_current != nullptr && g_snapshot_previous != nullptr ? bytes : 0;
  g_snapshot_allocation_id = g_snapshot_bytes != 0 ? allocation_id : 0;
}

void CaptureSnapshotDelta(LONG launch_id) {
  if (!EnvironmentEnabled("NEURALSTUDIO_CUDA_CAPTURE_SNAPSHOTS")) return;
  char directory[MAX_PATH]{};
  EnsureDirectoryFor("NEURALSTUDIO_CUDA_SNAPSHOT_DIR", directory,
                     sizeof(directory));
  if (directory[0] == '\0') return;
  AllocationRecord* target = SnapshotTarget();
  if (target == nullptr || target->bytes == 0) return;
  ResetSnapshotBuffers(target->bytes, target->id);
  if (g_snapshot_bytes == 0) {
    TraceCall("snapshot_error", "\"reason\":\"host_allocation_failed\"");
    return;
  }

  using Synchronize = CUresult(WINAPI*)();
  using CopyDtoH = CUresult(WINAPI*)(void*, CUdeviceptr, size_t);
  const Synchronize synchronize = Real<Synchronize>("cuCtxSynchronize");
  CopyDtoH copy = nullptr;
  if (EnsureRealDriver()) {
    copy = PointerCast<CopyDtoH>(
        GetProcAddress(g_real_driver, "cuMemcpyDtoH_v2"));
    if (copy == nullptr) {
      copy = PointerCast<CopyDtoH>(
          GetProcAddress(g_real_driver, "cuMemcpyDtoH"));
    }
  }
  const CUresult synchronize_result =
      synchronize != nullptr ? synchronize() : kCudaErrorNotFound;
  const CUresult copy_result =
      copy != nullptr && synchronize_result == kCudaSuccess
          ? copy(g_snapshot_current, target->pointer, target->bytes)
          : kCudaErrorNotFound;
  if (synchronize_result != kCudaSuccess || copy_result != kCudaSuccess) {
    char fields[192]{};
    sprintf_s(fields,
              "\"reason\":\"device_copy_failed\","
              "\"synchronize_result\":%d,\"copy_result\":%d",
              synchronize_result, copy_result);
    TraceCall("snapshot_error", fields);
    return;
  }

  uint32_t changed_pages = 0;
  const uint32_t page_count = static_cast<uint32_t>(
      (target->bytes + kDeltaPageBytes - 1) / kDeltaPageBytes);
  for (uint32_t page = 0; page < page_count; ++page) {
    const size_t offset = static_cast<size_t>(page) * kDeltaPageBytes;
    const size_t bytes = target->bytes - offset < kDeltaPageBytes
        ? target->bytes - offset
        : kDeltaPageBytes;
    if (memcmp(g_snapshot_current + offset, g_snapshot_previous + offset,
               bytes) != 0) {
      ++changed_pages;
    }
  }

  char path[MAX_PATH]{};
  sprintf_s(path, "%s\\snapshot-%04ld.delta", directory, launch_id);
  FILE* output = nullptr;
  if (fopen_s(&output, path, "wb") != 0 || output == nullptr) {
    TraceCall("snapshot_error", "\"reason\":\"delta_open_failed\"");
    return;
  }
  const unsigned char magic[8] = {'N', 'S', 'D', 'L', 'T', 'A', '1', 0};
  const uint32_t launch = static_cast<uint32_t>(launch_id);
  const uint32_t allocation = static_cast<uint32_t>(target->id);
  const uint64_t allocation_bytes = static_cast<uint64_t>(target->bytes);
  const uint32_t page_bytes = static_cast<uint32_t>(kDeltaPageBytes);
  fwrite(magic, 1, sizeof(magic), output);
  fwrite(&launch, sizeof(launch), 1, output);
  fwrite(&allocation, sizeof(allocation), 1, output);
  fwrite(&allocation_bytes, sizeof(allocation_bytes), 1, output);
  fwrite(&page_bytes, sizeof(page_bytes), 1, output);
  fwrite(&changed_pages, sizeof(changed_pages), 1, output);
  uint64_t changed_bytes = 0;
  for (uint32_t page = 0; page < page_count; ++page) {
    const size_t offset = static_cast<size_t>(page) * kDeltaPageBytes;
    const uint32_t bytes = static_cast<uint32_t>(
        target->bytes - offset < kDeltaPageBytes
            ? target->bytes - offset
            : kDeltaPageBytes);
    if (memcmp(g_snapshot_current + offset, g_snapshot_previous + offset,
               bytes) == 0) {
      continue;
    }
    fwrite(&page, sizeof(page), 1, output);
    fwrite(&bytes, sizeof(bytes), 1, output);
    fwrite(g_snapshot_current + offset, 1, bytes, output);
    memcpy(g_snapshot_previous + offset, g_snapshot_current + offset, bytes);
    changed_bytes += bytes;
  }
  fclose(output);
  char fields[320]{};
  sprintf_s(fields,
            "\"launch_id\":%ld,\"allocation_id\":%ld,"
            "\"allocation_bytes\":%llu,\"page_bytes\":%u,"
            "\"changed_pages\":%u,\"changed_bytes\":%llu,"
            "\"delta\":\"snapshot-%04ld.delta\"",
            launch_id, target->id,
            static_cast<unsigned long long>(target->bytes), page_bytes,
            changed_pages, static_cast<unsigned long long>(changed_bytes),
            launch_id);
  TraceCall("snapshot_delta", fields);
}

size_t ArrayFormatBytes(int format) {
  switch (format) {
    case 1:
    case 8:
      return 1;
    case 2:
    case 9:
    case 16:
      return 2;
    case 3:
    case 10:
    case 32:
      return 4;
    default:
      return 0;
  }
}

void CaptureArraySnapshots(LONG launch_id) {
  char directory[MAX_PATH]{};
  EnsureDirectoryFor("NEURALSTUDIO_CUDA_ARRAY_SNAPSHOT_DIR", directory,
                     sizeof(directory));
  if (directory[0] == '\0') return;

  using Synchronize = CUresult(WINAPI*)();
  using Copy2D = CUresult(WINAPI*)(const void*);
  const Synchronize synchronize = Real<Synchronize>("cuCtxSynchronize");
  const Copy2D copy = Real<Copy2D>("cuMemcpy2D_v2");
  const CUresult synchronize_result =
      synchronize != nullptr ? synchronize() : kCudaErrorNotFound;
  if (synchronize_result != kCudaSuccess || copy == nullptr) {
    char fields[192]{};
    sprintf_s(fields,
              "\"launch_id\":%ld,\"reason\":\"setup_failed\","
              "\"synchronize_result\":%d",
              launch_id, synchronize_result);
    TraceCall("array_snapshot_error", fields);
    return;
  }

  for (auto& array : g_arrays) {
    if (!array.active) continue;
    const size_t element_bytes = ArrayFormatBytes(array.format);
    if (element_bytes == 0 || array.channels == 0 || array.width == 0 ||
        array.height == 0 ||
        array.width > SIZE_MAX / array.channels / element_bytes) {
      char fields[160]{};
      sprintf_s(fields,
                "\"launch_id\":%ld,\"array_id\":%ld,"
                "\"reason\":\"unsupported_geometry\"",
                launch_id, array.id);
      TraceCall("array_snapshot_error", fields);
      continue;
    }
    const size_t row_bytes = array.width * array.channels * element_bytes;
    if (array.height > SIZE_MAX / row_bytes) {
      TraceCall("array_snapshot_error", "\"reason\":\"size_overflow\"");
      continue;
    }
    const size_t byte_count = row_bytes * array.height;
    void* host = VirtualAlloc(nullptr, byte_count, MEM_COMMIT | MEM_RESERVE,
                              PAGE_READWRITE);
    if (host == nullptr) {
      TraceCall("array_snapshot_error", "\"reason\":\"host_alloc_failed\"");
      continue;
    }
    CudaMemcpy2D descriptor{};
    descriptor.source_memory_type = 3;
    descriptor.source_array = array.handle;
    descriptor.destination_memory_type = 1;
    descriptor.destination_host = host;
    descriptor.destination_pitch = row_bytes;
    descriptor.width_bytes = row_bytes;
    descriptor.height = array.height;
    const CUresult result = copy(&descriptor);
    char filename[96]{};
    sprintf_s(filename, "array-launch-%04ld-array-%04ld.bin", launch_id,
              array.id);
    bool written = false;
    if (result == kCudaSuccess) {
      char path[MAX_PATH]{};
      sprintf_s(path, "%s\\%s", directory, filename);
      FILE* output = nullptr;
      if (fopen_s(&output, path, "wb") == 0 && output != nullptr) {
        written = fwrite(host, 1, byte_count, output) == byte_count;
        fclose(output);
      }
    }
    VirtualFree(host, 0, MEM_RELEASE);
    char fields[320]{};
    sprintf_s(fields,
              "\"launch_id\":%ld,\"array_id\":%ld,"
              "\"width\":%llu,\"height\":%llu,\"format\":%d,"
              "\"channels\":%u,\"bytes\":%llu,\"file\":\"%s\","
              "\"written\":%s,\"result\":%d",
              launch_id, array.id,
              static_cast<unsigned long long>(array.width),
              static_cast<unsigned long long>(array.height), array.format,
              array.channels, static_cast<unsigned long long>(byte_count),
              filename, written ? "true" : "false", result);
    TraceCall(written ? "array_snapshot" : "array_snapshot_error", fields);
  }
}

}  // namespace

extern "C" __declspec(dllexport) BOOL WINAPI NeuralStudioInstallCudaHooks() {
  if (g_cuda_hooks_installed) return TRUE;
  if (!EnsureRealDriver()) return FALSE;
  g_real_win_get_proc_address =
      PointerCast<WinGetProcAddress>(&GetProcAddress);
  g_real_get_proc_address = PointerCast<RealGetProcAddress>(
      OriginalGetProcAddress(g_real_driver, "cuGetProcAddress_v2"));
  if (g_real_get_proc_address == nullptr) return FALSE;
  wchar_t core_path[MAX_PATH]{};
  const DWORD core_path_length = GetEnvironmentVariableW(
      L"NEURALSTUDIO_NVNGX_CORE", core_path,
      static_cast<DWORD>(sizeof(core_path) / sizeof(core_path[0])));
  if (core_path_length == 0 ||
      core_path_length >= sizeof(core_path) / sizeof(core_path[0])) {
    return FALSE;
  }
  g_nvngx_core = LoadLibraryW(core_path);
  HMODULE runtime = GetModuleHandleW(L"nvngx_dlssnr.dll");
  if (runtime == nullptr) runtime = GetModuleHandleW(L"nvngx_dlss.dll");
  if (g_nvngx_core == nullptr || runtime == nullptr ||
      !PatchModuleGetProcAddress(g_nvngx_core) ||
      !PatchModuleGetProcAddress(runtime)) {
    RestoreIatPatches();
    return FALSE;
  }
  g_cuda_hooks_installed = true;
  char fields[160]{};
  sprintf_s(fields,
            "{\"event\":\"nvcuda_driver_hooks_installed\","
            "\"driver_export_wrappers\":31,"
            "\"iat_get_proc_address_hooks\":2,"
            "\"get_proc_address_export\":\"cuGetProcAddress_v2\"}");
  TraceLine(fields);
  return TRUE;
}

extern "C" __declspec(dllexport) BOOL WINAPI NeuralStudioRemoveCudaHooks() {
  if (!g_cuda_hooks_installed) return TRUE;
  if (!RestoreIatPatches()) return FALSE;
  g_cuda_hooks_installed = false;
  TraceLine("{\"event\":\"nvcuda_driver_hooks_removed\"}");
  return TRUE;
}

extern "C" __declspec(dllexport) void WINAPI NeuralStudioCudaTraceSetPhase(
    const char* phase) {
  char fields[128]{};
  sprintf_s(fields, "\"phase\":\"%s\"", phase != nullptr ? phase : "");
  TraceCall("NeuralStudioCudaTraceSetPhase", fields);
}

extern "C" __declspec(dllexport) CUresult WINAPI cuGetProcAddress(
    const char* symbol, void** function, int cuda_version, uint64_t flags,
    void* query_status) {
  if (symbol == nullptr || function == nullptr || !EnsureRealDriver()) {
    return kCudaErrorInvalidValue;
  }
  const auto real_get = g_real_get_proc_address != nullptr
      ? g_real_get_proc_address
      : PointerCast<RealGetProcAddress>(
            GetProcAddress(g_real_driver, "cuGetProcAddress"));
  if (real_get == nullptr) return kCudaErrorNotFound;
  void* real_function = nullptr;
  const CUresult result = real_get(symbol, &real_function, cuda_version, flags,
                                   query_status);
  ForwardEntry* entry = FindEntry(symbol);
  if (result == kCudaSuccess && real_function != nullptr && entry != nullptr) {
    entry->real = PointerCast<FARPROC>(real_function);
    *function = PointerCast<void*>(entry->wrapper);
  } else {
    *function = real_function;
  }
  char fields[640]{};
  sprintf_s(fields,
            "\"symbol\":\"%s\",\"cuda_version\":%d,"
            "\"flags\":%llu,\"intercepted\":%s,\"result\":%d",
            symbol, cuda_version, static_cast<unsigned long long>(flags),
            entry != nullptr ? "true" : "false", result);
  TraceCall("cuGetProcAddress", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuInit(unsigned int flags) {
  using Function = CUresult(WINAPI*)(unsigned int);
  const Function real = Real<Function>("cuInit");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(flags);
  char fields[96]{};
  sprintf_s(fields, "\"flags\":%u,\"result\":%d", flags, result);
  TraceCall("cuInit", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDeviceGet(
    int* device, int ordinal) {
  using Function = CUresult(WINAPI*)(int*, int);
  const Function real = Real<Function>("cuDeviceGet");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(device, ordinal);
  char fields[128]{};
  sprintf_s(fields, "\"ordinal\":%d,\"device\":%d,\"result\":%d",
            ordinal, device != nullptr ? *device : -1, result);
  TraceCall("cuDeviceGet", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDevicePrimaryCtxRetain(
    void** context, int device) {
  using Function = CUresult(WINAPI*)(void**, int);
  const Function real = Real<Function>("cuDevicePrimaryCtxRetain");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(context, device);
  char fields[128]{};
  sprintf_s(fields, "\"device\":%d,\"context\":\"%p\",\"result\":%d",
            device, context != nullptr ? *context : nullptr, result);
  TraceCall("cuDevicePrimaryCtxRetain", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDevicePrimaryCtxRelease(
    int device) {
  using Function = CUresult(WINAPI*)(int);
  const Function real = Real<Function>("cuDevicePrimaryCtxRelease");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(device);
  char fields[96]{};
  sprintf_s(fields, "\"device\":%d,\"result\":%d", device, result);
  TraceCall("cuDevicePrimaryCtxRelease", fields);
  return result;
}

#define SIMPLE_FORWARD0(name)                                                   \
  extern "C" __declspec(dllexport) CUresult WINAPI name() {                    \
    using Function = CUresult(WINAPI*)();                                       \
    const Function real = Real<Function>(#name);                                \
    if (real == nullptr) return kCudaErrorNotFound;                              \
    const CUresult result = real();                                              \
    TraceCall(#name, nullptr);                                                   \
    return result;                                                               \
  }

SIMPLE_FORWARD0(cuCtxSynchronize)

extern "C" __declspec(dllexport) CUresult WINAPI cuCtxGetDevice(int* device) {
  using Function = CUresult(WINAPI*)(int*);
  const Function real = Real<Function>("cuCtxGetDevice");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(device);
  char fields[96]{};
  sprintf_s(fields, "\"device\":%d,\"result\":%d",
            device != nullptr ? *device : -1, result);
  TraceCall("cuCtxGetDevice", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuCtxSetCurrent(void* context) {
  using Function = CUresult(WINAPI*)(void*);
  const Function real = Real<Function>("cuCtxSetCurrent");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(context);
  char fields[128]{};
  sprintf_s(fields, "\"context\":\"%p\",\"result\":%d", context, result);
  TraceCall("cuCtxSetCurrent", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuGetErrorString(
    CUresult error, const char** text) {
  using Function = CUresult(WINAPI*)(CUresult, const char**);
  const Function real = Real<Function>("cuGetErrorString");
  return real != nullptr ? real(error, text) : kCudaErrorNotFound;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDeviceGetAttribute(
    int* value, int attribute, int device) {
  using Function = CUresult(WINAPI*)(int*, int, int);
  const Function real = Real<Function>("cuDeviceGetAttribute");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(value, attribute, device);
  char fields[160]{};
  sprintf_s(fields,
            "\"attribute\":%d,\"device\":%d,\"value\":%d,"
            "\"result\":%d",
            attribute, device, value != nullptr ? *value : 0, result);
  TraceCall("cuDeviceGetAttribute", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDeviceGetLuid(
    char* luid, unsigned int* node_mask, int device) {
  using Function = CUresult(WINAPI*)(char*, unsigned int*, int);
  const Function real = Real<Function>("cuDeviceGetLuid");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(luid, node_mask, device);
  char fields[128]{};
  sprintf_s(fields, "\"device\":%d,\"node_mask\":%u,\"result\":%d",
            device, node_mask != nullptr ? *node_mask : 0, result);
  TraceCall("cuDeviceGetLuid", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuDeviceGetUuid(
    void* uuid, int device) {
  using Function = CUresult(WINAPI*)(void*, int);
  const Function real = Real<Function>("cuDeviceGetUuid");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(uuid, device);
  char fields[96]{};
  sprintf_s(fields, "\"device\":%d,\"result\":%d", device, result);
  TraceCall("cuDeviceGetUuid", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemAlloc(
    CUdeviceptr* pointer, size_t bytes) {
  using Function = CUresult(WINAPI*)(CUdeviceptr*, size_t);
  const Function real = Real<Function>("cuMemAlloc");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(pointer, bytes);
  AllocationRecord* record = nullptr;
  if (result == kCudaSuccess && pointer != nullptr) record = AddAllocation(*pointer, bytes);
  char fields[192]{};
  sprintf_s(fields,
            "\"allocation_id\":%ld,\"pointer\":\"0x%llx\","
            "\"bytes\":%llu,\"result\":%d",
            record != nullptr ? record->id : 0,
            pointer != nullptr ? static_cast<unsigned long long>(*pointer) : 0,
            static_cast<unsigned long long>(bytes), result);
  TraceCall("cuMemAlloc", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemFree(CUdeviceptr pointer) {
  using Function = CUresult(WINAPI*)(CUdeviceptr);
  const Function real = Real<Function>("cuMemFree");
  if (real == nullptr) return kCudaErrorNotFound;
  AllocationRecord* record = FindAllocation(pointer);
  const CUresult result = real(pointer);
  char fields[128]{};
  sprintf_s(fields, "\"allocation_id\":%ld,\"result\":%d",
            record != nullptr ? record->id : 0, result);
  TraceCall("cuMemFree", fields);
  if (result == kCudaSuccess && record != nullptr) record->active = false;
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemAllocHost(
    void** pointer, size_t bytes) {
  using Function = CUresult(WINAPI*)(void**, size_t);
  const Function real = Real<Function>("cuMemAllocHost");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(pointer, bytes);
  char fields[160]{};
  sprintf_s(fields, "\"pointer\":\"%p\",\"bytes\":%llu,\"result\":%d",
            pointer != nullptr ? *pointer : nullptr,
            static_cast<unsigned long long>(bytes), result);
  TraceCall("cuMemAllocHost", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemFreeHost(void* pointer) {
  using Function = CUresult(WINAPI*)(void*);
  const Function real = Real<Function>("cuMemFreeHost");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(pointer);
  TraceCall("cuMemFreeHost", nullptr);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemcpyHtoDAsync(
    CUdeviceptr destination, const void* source, size_t bytes, void* stream) {
  using Function = CUresult(WINAPI*)(CUdeviceptr, const void*, size_t, void*);
  const Function real = Real<Function>("cuMemcpyHtoDAsync");
  if (real == nullptr) return kCudaErrorNotFound;
  const LONG transfer_id = InterlockedIncrement(&g_next_transfer_id);
  DumpBytes("transfer", transfer_id, source, bytes);
  const CUresult result = real(destination, source, bytes, stream);
  AllocationRecord* allocation = nullptr;
  size_t offset = 0;
  for (auto& candidate : g_allocations) {
    if (candidate.active && destination >= candidate.pointer &&
        destination + bytes >= destination &&
        destination + bytes <= candidate.pointer + candidate.bytes) {
      allocation = &candidate;
      offset = static_cast<size_t>(destination - candidate.pointer);
      break;
    }
  }
  char fields[256]{};
  sprintf_s(fields,
            "\"transfer_id\":%ld,\"allocation_id\":%ld,"
            "\"byte_offset\":%llu,\"bytes\":%llu,\"result\":%d",
            transfer_id, allocation != nullptr ? allocation->id : 0,
            static_cast<unsigned long long>(offset),
            static_cast<unsigned long long>(bytes), result);
  TraceCall("cuMemcpyHtoDAsync", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuMemsetD32Async(
    CUdeviceptr destination, unsigned int value, size_t count, void* stream) {
  using Function = CUresult(WINAPI*)(CUdeviceptr, unsigned int, size_t, void*);
  const Function real = Real<Function>("cuMemsetD32Async");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(destination, value, count, stream);
  char fields[224]{};
  sprintf_s(fields,
            "\"destination\":\"0x%llx\",\"value\":%u,"
            "\"count\":%llu,\"result\":%d",
            static_cast<unsigned long long>(destination), value,
            static_cast<unsigned long long>(count), result);
  TraceCall("cuMemsetD32Async", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuModuleLoadData(
    void** module, const void* image) {
  using Function = CUresult(WINAPI*)(void**, const void*);
  const Function real = Real<Function>("cuModuleLoadData");
  if (real == nullptr) return kCudaErrorNotFound;
  const size_t image_bytes = FatbinBytes(image);
  const CUresult result = real(module, image);
  ModuleRecord* record = nullptr;
  if (result == kCudaSuccess && module != nullptr) record = AddModule(*module);
  if (record != nullptr) DumpBytes("module", record->id, image, image_bytes);
  char fields[256]{};
  sprintf_s(fields,
            "\"module_id\":%ld,\"bytes\":%llu,"
            "\"dump\":\"module-%04ld.bin\",\"result\":%d",
            record != nullptr ? record->id : 0,
            static_cast<unsigned long long>(image_bytes),
            record != nullptr ? record->id : 0, result);
  TraceCall("cuModuleLoadData", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuModuleUnload(void* module) {
  using Function = CUresult(WINAPI*)(void*);
  const Function real = Real<Function>("cuModuleUnload");
  if (real == nullptr) return kCudaErrorNotFound;
  ModuleRecord* record = FindModule(module);
  const CUresult result = real(module);
  char fields[128]{};
  sprintf_s(fields, "\"module_id\":%ld,\"result\":%d",
            record != nullptr ? record->id : 0, result);
  TraceCall("cuModuleUnload", fields);
  if (result == kCudaSuccess && record != nullptr) record->active = false;
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuModuleGetFunction(
    void** function, void* module, const char* name) {
  using Function = CUresult(WINAPI*)(void**, void*, const char*);
  const Function real = Real<Function>("cuModuleGetFunction");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(function, module, name);
  ModuleRecord* module_record = FindModule(module);
  FunctionRecord* record = nullptr;
  if (result == kCudaSuccess && function != nullptr && name != nullptr) {
    record = AddFunction(*function,
                         module_record != nullptr ? module_record->id : 0,
                         name);
  }
  char fields[640]{};
  sprintf_s(fields,
            "\"function_id\":%ld,\"module_id\":%ld,"
            "\"name\":\"%s\",\"result\":%d",
            record != nullptr ? record->id : 0,
            module_record != nullptr ? module_record->id : 0,
            name != nullptr ? name : "", result);
  TraceCall("cuModuleGetFunction", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuFuncSetAttribute(
    void* function, int attribute, int value) {
  using Function = CUresult(WINAPI*)(void*, int, int);
  const Function real = Real<Function>("cuFuncSetAttribute");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(function, attribute, value);
  FunctionRecord* record = FindFunction(function);
  char fields[192]{};
  sprintf_s(fields,
            "\"function_id\":%ld,\"attribute\":%d,"
            "\"value\":%d,\"result\":%d",
            record != nullptr ? record->id : 0, attribute, value, result);
  TraceCall("cuFuncSetAttribute", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuLaunchKernel(
    void* function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_bytes, void* stream,
    void** parameters, void** extra) {
  using Function = CUresult(WINAPI*)(
      void*, unsigned int, unsigned int, unsigned int, unsigned int,
      unsigned int, unsigned int, unsigned int, void*, void**, void**);
  const Function real = Real<Function>("cuLaunchKernel");
  if (real == nullptr) return kCudaErrorNotFound;
  const LONG launch_id = InterlockedIncrement(&g_next_launch_id);
  const unsigned char* argument_buffer = nullptr;
  size_t argument_bytes = 0;
  if (extra != nullptr) {
    for (size_t index = 0; index < 16; index += 2) {
      if (extra[index] == nullptr) break;
      if (extra[index] == reinterpret_cast<void*>(0x1)) {
        argument_buffer = static_cast<const unsigned char*>(extra[index + 1]);
      } else if (extra[index] == reinterpret_cast<void*>(0x2) &&
                 extra[index + 1] != nullptr) {
        argument_bytes = *static_cast<const size_t*>(extra[index + 1]);
      }
    }
  }
  if (argument_bytes > 1024 * 1024) argument_bytes = 0;
  DumpBytes("arguments", launch_id, argument_buffer, argument_bytes);
  const CUresult result = real(function, grid_x, grid_y, grid_z, block_x,
                               block_y, block_z, shared_bytes, stream,
                               parameters, extra);
  FunctionRecord* record = FindFunction(function);
  char fields[900]{};
  sprintf_s(fields,
            "\"launch_id\":%ld,\"function_id\":%ld,"
            "\"name\":\"%s\",\"grid\":[%u,%u,%u],"
            "\"block\":[%u,%u,%u],\"shared_bytes\":%u,"
            "\"argument_bytes\":%llu,"
            "\"argument_dump\":\"arguments-%04ld.bin\","
            "\"result\":%d",
            launch_id, record != nullptr ? record->id : 0,
            record != nullptr ? record->name : "", grid_x, grid_y, grid_z,
            block_x, block_y, block_z, shared_bytes,
            static_cast<unsigned long long>(argument_bytes), launch_id,
            result);
  TraceCall("cuLaunchKernel", fields);
  if (result == kCudaSuccess) {
    CaptureSnapshotDelta(launch_id);
    CaptureArraySnapshots(launch_id);
  }
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuArrayCreate(
    void** array, const void* descriptor) {
  using Function = CUresult(WINAPI*)(void**, const void*);
  const Function real = Real<Function>("cuArrayCreate");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(array, descriptor);
  ArrayRecord* record = nullptr;
  const auto* typed = static_cast<const CudaArrayDescriptor*>(descriptor);
  if (result == kCudaSuccess && array != nullptr && typed != nullptr) {
    record = AddArray(*array, typed);
  }
  char fields[256]{};
  sprintf_s(fields,
            "\"array_id\":%ld,\"pointer\":\"%p\","
            "\"width\":%llu,\"height\":%llu,"
            "\"format\":%d,\"channels\":%u,\"result\":%d",
            record != nullptr ? record->id : 0,
            array != nullptr ? *array : nullptr,
            typed != nullptr ? static_cast<unsigned long long>(typed->width) : 0,
            typed != nullptr ? static_cast<unsigned long long>(typed->height) : 0,
            typed != nullptr ? typed->format : 0,
            typed != nullptr ? typed->channels : 0, result);
  TraceCall("cuArrayCreate", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuArray3DCreate(
    void** array, const void* descriptor) {
  using Function = CUresult(WINAPI*)(void**, const void*);
  const Function real = Real<Function>("cuArray3DCreate_v2");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(array, descriptor);
  ArrayRecord* record = nullptr;
  const auto* typed = static_cast<const CudaArray3DDescriptor*>(descriptor);
  CudaArrayDescriptor projected{};
  if (typed != nullptr) {
    projected.width = typed->width;
    projected.height = typed->height;
    projected.format = typed->format;
    projected.channels = typed->channels;
  }
  if (result == kCudaSuccess && array != nullptr && typed != nullptr) {
    record = AddArray(*array, &projected);
  }
  char fields[320]{};
  sprintf_s(fields,
            "\"array_id\":%ld,\"pointer\":\"%p\","
            "\"width\":%llu,\"height\":%llu,"
            "\"depth\":%llu,\"format\":%d,\"channels\":%u,"
            "\"flags\":%u,\"result\":%d",
            record != nullptr ? record->id : 0,
            array != nullptr ? *array : nullptr,
            typed != nullptr ? static_cast<unsigned long long>(typed->width) : 0,
            typed != nullptr ? static_cast<unsigned long long>(typed->height) : 0,
            typed != nullptr ? static_cast<unsigned long long>(typed->depth) : 0,
            typed != nullptr ? typed->format : 0,
            typed != nullptr ? typed->channels : 0,
            typed != nullptr ? typed->flags : 0, result);
  TraceCall("cuArray3DCreate", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuArrayDestroy(void* array) {
  using Function = CUresult(WINAPI*)(void*);
  const Function real = Real<Function>("cuArrayDestroy");
  if (real == nullptr) return kCudaErrorNotFound;
  ArrayRecord* record = FindArray(array);
  const CUresult result = real(array);
  char fields[128]{};
  sprintf_s(fields, "\"array_id\":%ld,\"result\":%d",
            record != nullptr ? record->id : 0, result);
  TraceCall("cuArrayDestroy", fields);
  if (result == kCudaSuccess && record != nullptr) record->active = false;
  return result;
}

#define SIMPLE_FORWARD2(name, type1, type2)                                     \
  extern "C" __declspec(dllexport) CUresult WINAPI name(type1 a, type2 b) {     \
    using Function = CUresult(WINAPI*)(type1, type2);                            \
    const Function real = Real<Function>(#name);                                \
    if (real == nullptr) return kCudaErrorNotFound;                              \
    const CUresult result = real(a, b);                                          \
    TraceCall(#name, nullptr);                                                   \
    return result;                                                               \
  }

SIMPLE_FORWARD2(cuArrayGetDescriptor, void*, void*)
SIMPLE_FORWARD2(cuSurfObjectGetResourceDesc, void*, uint64_t)
SIMPLE_FORWARD2(cuTexObjectGetResourceDesc, void*, uint64_t)

#define SIMPLE_FORWARD1(name, type1)                                            \
  extern "C" __declspec(dllexport) CUresult WINAPI name(type1 a) {             \
    using Function = CUresult(WINAPI*)(type1);                                   \
    const Function real = Real<Function>(#name);                                \
    if (real == nullptr) return kCudaErrorNotFound;                              \
    const CUresult result = real(a);                                             \
    TraceCall(#name, nullptr);                                                   \
    return result;                                                               \
  }

SIMPLE_FORWARD1(cuDestroyExternalMemory, void*)
SIMPLE_FORWARD1(cuMipmappedArrayDestroy, void*)
SIMPLE_FORWARD1(cuSurfObjectDestroy, uint64_t)
SIMPLE_FORWARD1(cuTexObjectDestroy, uint64_t)

extern "C" __declspec(dllexport) CUresult WINAPI cuMemcpy2D(
    const void* descriptor) {
  using Function = CUresult(WINAPI*)(const void*);
  const Function real = Real<Function>("cuMemcpy2D");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(descriptor);
  const auto* typed = static_cast<const CudaMemcpy2D*>(descriptor);
  ArrayRecord* source_array =
      typed != nullptr ? FindArray(typed->source_array) : nullptr;
  ArrayRecord* destination_array =
      typed != nullptr ? FindArray(typed->destination_array) : nullptr;
  char fields[320]{};
  sprintf_s(fields,
            "\"source_memory_type\":%d,\"source_array_id\":%ld,"
            "\"destination_memory_type\":%d,"
            "\"destination_array_id\":%ld,\"width_bytes\":%llu,"
            "\"height\":%llu,\"result\":%d",
            typed != nullptr ? typed->source_memory_type : 0,
            source_array != nullptr ? source_array->id : 0,
            typed != nullptr ? typed->destination_memory_type : 0,
            destination_array != nullptr ? destination_array->id : 0,
            typed != nullptr
                ? static_cast<unsigned long long>(typed->width_bytes)
                : 0,
            typed != nullptr ? static_cast<unsigned long long>(typed->height)
                             : 0,
            result);
  TraceCall("cuMemcpy2D", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuSurfObjectCreate(
    uint64_t* object, const void* resource) {
  using Function = CUresult(WINAPI*)(uint64_t*, const void*);
  const Function real = Real<Function>("cuSurfObjectCreate");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(object, resource);
  const auto* typed = static_cast<const CudaResourceDesc*>(resource);
  ArrayRecord* array = typed != nullptr ? FindArray(typed->resource.array.array) : nullptr;
  char fields[192]{};
  sprintf_s(fields, "\"object\":%llu,\"array_id\":%ld,\"result\":%d",
            object != nullptr ? static_cast<unsigned long long>(*object) : 0,
            array != nullptr ? array->id : 0,
            result);
  TraceCall("cuSurfObjectCreate", fields);
  return result;
}

extern "C" __declspec(dllexport) CUresult WINAPI cuTexObjectCreate(
    uint64_t* object, const void* resource, const void* texture,
    const void* view) {
  using Function = CUresult(WINAPI*)(
      uint64_t*, const void*, const void*, const void*);
  const Function real = Real<Function>("cuTexObjectCreate");
  if (real == nullptr) return kCudaErrorNotFound;
  const CUresult result = real(object, resource, texture, view);
  const auto* typed = static_cast<const CudaResourceDesc*>(resource);
  const auto* typed_texture = static_cast<const CudaTextureDesc*>(texture);
  ArrayRecord* array = typed != nullptr ? FindArray(typed->resource.array.array) : nullptr;
  char fields[512]{};
  sprintf_s(fields,
            "\"object\":%llu,\"array_id\":%ld,"
            "\"address_mode\":[%d,%d,%d],\"filter_mode\":%d,"
            "\"texture_flags\":%u,\"max_anisotropy\":%u,"
            "\"mipmap_filter_mode\":%d,\"mipmap_level_bias_bits\":%u,"
            "\"min_mipmap_level_clamp_bits\":%u,"
            "\"max_mipmap_level_clamp_bits\":%u,\"result\":%d",
            object != nullptr ? static_cast<unsigned long long>(*object) : 0,
            array != nullptr ? array->id : 0,
            typed_texture != nullptr ? typed_texture->address_mode[0] : 0,
            typed_texture != nullptr ? typed_texture->address_mode[1] : 0,
            typed_texture != nullptr ? typed_texture->address_mode[2] : 0,
            typed_texture != nullptr ? typed_texture->filter_mode : 0,
            typed_texture != nullptr ? typed_texture->flags : 0,
            typed_texture != nullptr ? typed_texture->max_anisotropy : 0,
            typed_texture != nullptr ? typed_texture->mipmap_filter_mode : 0,
            typed_texture != nullptr
                ? std::bit_cast<unsigned int>(typed_texture->mipmap_level_bias)
                : 0,
            typed_texture != nullptr
                ? std::bit_cast<unsigned int>(typed_texture->min_mipmap_level_clamp)
                : 0,
            typed_texture != nullptr
                ? std::bit_cast<unsigned int>(typed_texture->max_mipmap_level_clamp)
                : 0,
            result);
  TraceCall("cuTexObjectCreate", fields);
  return result;
}

extern "C" BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID) {
  if (reason == DLL_PROCESS_ATTACH) {
    g_self = instance;
    InitializeCriticalSection(&g_lock);
    TraceLine("{\"event\":\"nvcuda_forward_trace_loaded\","
              "\"inference_proven\":false}");
  } else if (reason == DLL_PROCESS_DETACH) {
    if (g_snapshot_current != nullptr) VirtualFree(g_snapshot_current, 0, MEM_RELEASE);
    if (g_snapshot_previous != nullptr) VirtualFree(g_snapshot_previous, 0, MEM_RELEASE);
    TraceLine("{\"event\":\"nvcuda_forward_trace_unloaded\"}");
    DeleteCriticalSection(&g_lock);
  }
  return TRUE;
}
