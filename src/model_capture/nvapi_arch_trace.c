#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define NVAPI_OK 0
#define NVAPI_INVALID_ARGUMENT -5
#define NVAPI_MAX_PHYSICAL_GPUS 64

#define ID_INITIALIZE 0x0150E828u
#define ID_UNLOAD 0xD22BDD7Eu
#define ID_GET_ERROR_MESSAGE 0x6C2D048Cu
#define ID_ENUM_PHYSICAL_GPUS 0xE5AC921Fu
#define ID_GET_LOGICAL_FROM_PHYSICAL 0xADD604D1u
#define ID_GPU_GET_LOGICAL_INFO 0x842B066Eu
#define ID_GPU_GET_ARCH_INFO 0xD8265D24u

typedef void* NvPhysicalGpuHandle;
typedef void* NvLogicalGpuHandle;
typedef int NvAPI_Status;

typedef struct NvLogicalGpuData {
  uint32_t version;
  void* os_adapter_id;
  uint32_t physical_gpu_count;
  NvPhysicalGpuHandle physical_gpu_handles[NVAPI_MAX_PHYSICAL_GPUS];
  uint32_t reserved[8];
} NvLogicalGpuData;

typedef struct NvGpuArchInfo {
  uint32_t version;
  uint32_t architecture;
  uint32_t implementation;
  uint32_t revision;
} NvGpuArchInfo;

static const uintptr_t kPhysicalHandle = 0x1B000001u;
static const uintptr_t kLogicalHandle = 0x1B000002u;
static CRITICAL_SECTION g_log_lock;

static void trace_query(const char* dispatcher, uint32_t id, int resolved) {
  char path[MAX_PATH];
  DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_NVAPI_TRACE", path, (DWORD)sizeof(path));
  if (length == 0 || length >= sizeof(path)) {
    strcpy(path, "nvapi-arch-trace.jsonl");
  }
  EnterCriticalSection(&g_log_lock);
  FILE* output = fopen(path, "ab");
  if (output != NULL) {
    fprintf(output,
            "{\"dispatcher\":\"%s\",\"id\":\"0x%08X\","
            "\"resolved\":%s}\n",
            dispatcher, id, resolved ? "true" : "false");
    fclose(output);
  }
  LeaveCriticalSection(&g_log_lock);
}

static NvAPI_Status WINAPI trace_initialize(void) { return NVAPI_OK; }
static NvAPI_Status WINAPI trace_unload(void) { return NVAPI_OK; }

static NvAPI_Status WINAPI trace_get_error_message(NvAPI_Status status,
                                                    char* message) {
  if (message == NULL) return NVAPI_INVALID_ARGUMENT;
  _snprintf(message, 64, "NeuralStudio NVAPI architecture trace status %d",
            status);
  message[63] = '\0';
  return NVAPI_OK;
}

static NvAPI_Status WINAPI trace_enum_physical_gpus(
    NvPhysicalGpuHandle* handles, uint32_t* count) {
  if (handles == NULL || count == NULL) return NVAPI_INVALID_ARGUMENT;
  memset(handles, 0, sizeof(NvPhysicalGpuHandle) * NVAPI_MAX_PHYSICAL_GPUS);
  handles[0] = (NvPhysicalGpuHandle)kPhysicalHandle;
  *count = 1;
  return NVAPI_OK;
}

static NvAPI_Status WINAPI trace_get_logical_from_physical(
    NvPhysicalGpuHandle physical, NvLogicalGpuHandle* logical) {
  if (physical != (NvPhysicalGpuHandle)kPhysicalHandle || logical == NULL) {
    return NVAPI_INVALID_ARGUMENT;
  }
  *logical = (NvLogicalGpuHandle)kLogicalHandle;
  return NVAPI_OK;
}

static NvAPI_Status WINAPI trace_gpu_get_logical_info(
    NvLogicalGpuHandle logical, NvLogicalGpuData* data) {
  if (logical != (NvLogicalGpuHandle)kLogicalHandle || data == NULL ||
      data->os_adapter_id == NULL) {
    return NVAPI_INVALID_ARGUMENT;
  }
  memset(data->os_adapter_id, 0x42, 8);
  data->physical_gpu_count = 1;
  data->physical_gpu_handles[0] = (NvPhysicalGpuHandle)kPhysicalHandle;
  memset(data->reserved, 0, sizeof(data->reserved));
  return NVAPI_OK;
}

static NvAPI_Status WINAPI trace_gpu_get_arch_info(
    NvPhysicalGpuHandle physical, NvGpuArchInfo* info) {
  if (physical != (NvPhysicalGpuHandle)kPhysicalHandle || info == NULL) {
    return NVAPI_INVALID_ARGUMENT;
  }
  info->architecture = 0x1B0u;
  info->implementation = 0x2u;
  info->revision = 0x11u;
  return NVAPI_OK;
}

static void* resolve_interface(uint32_t id) {
  switch (id) {
    case ID_INITIALIZE: return (void*)trace_initialize;
    case ID_UNLOAD: return (void*)trace_unload;
    case ID_GET_ERROR_MESSAGE: return (void*)trace_get_error_message;
    case ID_ENUM_PHYSICAL_GPUS: return (void*)trace_enum_physical_gpus;
    case ID_GET_LOGICAL_FROM_PHYSICAL:
      return (void*)trace_get_logical_from_physical;
    case ID_GPU_GET_LOGICAL_INFO: return (void*)trace_gpu_get_logical_info;
    case ID_GPU_GET_ARCH_INFO: return (void*)trace_gpu_get_arch_info;
    default: return NULL;
  }
}

__declspec(dllexport) void* __cdecl nvapi_QueryInterface(uint32_t id) {
  void* result = resolve_interface(id);
  trace_query("nvapi_QueryInterface", id, result != NULL);
  return result;
}

__declspec(dllexport) void* __cdecl nvapi_Direct_GetMethod(uint32_t id) {
  void* result = resolve_interface(id);
  trace_query("nvapi_Direct_GetMethod", id, result != NULL);
  return result;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
  (void)instance;
  (void)reserved;
  if (reason == DLL_PROCESS_ATTACH) {
    InitializeCriticalSection(&g_log_lock);
  } else if (reason == DLL_PROCESS_DETACH) {
    DeleteCriticalSection(&g_log_lock);
  }
  return TRUE;
}
