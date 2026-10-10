#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_FOUND 500

typedef uint64_t CUdeviceptr;
typedef int CUresult;

typedef struct TraceModule {
  LONG id;
  const void* image;
} TraceModule;

typedef struct TraceFunction {
  LONG id;
  TraceModule* module;
  char name[256];
} TraceFunction;

typedef struct TraceArray {
  LONG id;
  size_t width;
  size_t height;
  int format;
  unsigned int channels;
} TraceArray;

typedef struct CudaArrayDescriptor {
  size_t width;
  size_t height;
  int format;
  unsigned int channels;
} CudaArrayDescriptor;

typedef struct CudaArray3DDescriptor {
  size_t width;
  size_t height;
  size_t depth;
  int format;
  unsigned int channels;
  unsigned int flags;
} CudaArray3DDescriptor;

typedef struct CudaResourceDesc {
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
} CudaResourceDesc;

typedef struct TraceObject {
  uint64_t id;
  CudaResourceDesc resource;
  int active;
} TraceObject;

static HMODULE g_module;
static CRITICAL_SECTION g_log_lock;
static LONG g_next_id;
static LONG g_next_transfer_id;
static LONG g_next_launch_id;
static LONG g_next_array_id;
static TraceObject g_texture_objects[4096];
static TraceObject g_surface_objects[4096];
static char g_phase[32] = "load";

static void trace_line(const char* line) {
  char path[MAX_PATH];
  DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_CUDA_TRACE", path, (DWORD)sizeof(path));
  if (length == 0 || length >= sizeof(path)) {
    strcpy(path, "nvcuda-trace.jsonl");
  }
  EnterCriticalSection(&g_log_lock);
  FILE* output = fopen(path, "ab");
  if (output != NULL) {
    fputs(line, output);
    fputc('\n', output);
    fclose(output);
  }
  LeaveCriticalSection(&g_log_lock);
}

static void trace_call(const char* name, const char* fields) {
  char line[1152];
  if (fields != NULL && fields[0] != '\0') {
    _snprintf(line, sizeof(line),
              "{\"call\":\"%s\",\"phase\":\"%s\",%s}",
              name, g_phase, fields);
  } else {
    _snprintf(line, sizeof(line),
              "{\"call\":\"%s\",\"phase\":\"%s\"}", name,
              g_phase);
  }
  line[sizeof(line) - 1] = '\0';
  trace_line(line);
}

__declspec(dllexport) void WINAPI NeuralStudioCudaTraceSetPhase(
    const char* phase) {
  if (phase == NULL || phase[0] == '\0') return;
  EnterCriticalSection(&g_log_lock);
  strncpy(g_phase, phase, sizeof(g_phase) - 1);
  g_phase[sizeof(g_phase) - 1] = '\0';
  LeaveCriticalSection(&g_log_lock);
  trace_call("NeuralStudioCudaTraceSetPhase", NULL);
}

static void dump_bytes(const char* kind, LONG id, const void* data,
                       size_t size) {
  char directory[MAX_PATH];
  DWORD length = GetEnvironmentVariableA(
      "NEURALSTUDIO_CUDA_DUMP_DIR", directory, (DWORD)sizeof(directory));
  if (length == 0 || length >= sizeof(directory) || data == NULL) return;
  CreateDirectoryA(directory, NULL);
  char path[MAX_PATH];
  _snprintf(path, sizeof(path), "%s\\%s-%04ld.bin", directory, kind, id);
  path[sizeof(path) - 1] = '\0';
  FILE* output = fopen(path, "wb");
  if (output != NULL) {
    fwrite(data, 1, size, output);
    fclose(output);
  }
}

static void* allocate_zeroed(size_t size) {
  return HeapAlloc(GetProcessHeap(), HEAP_ZERO_MEMORY, size == 0 ? 1 : size);
}

static void release_allocation(void* pointer) {
  if (pointer != NULL) HeapFree(GetProcessHeap(), 0, pointer);
}

static TraceObject* find_object(TraceObject* objects, uint64_t id) {
  size_t index;
  for (index = 0; index < 4096; ++index) {
    if (objects[index].active && objects[index].id == id) return &objects[index];
  }
  return NULL;
}

static TraceObject* add_object(TraceObject* objects, uint64_t id,
                               const CudaResourceDesc* resource) {
  size_t index;
  for (index = 0; index < 4096; ++index) {
    if (!objects[index].active) {
      objects[index].id = id;
      objects[index].resource = *resource;
      objects[index].active = 1;
      return &objects[index];
    }
  }
  return NULL;
}

static FARPROC resolve_symbol(const char* symbol) {
  FARPROC result = GetProcAddress(g_module, symbol);
  if (result != NULL) return result;

  char base[256];
  size_t length = strlen(symbol);
  if (length >= sizeof(base)) return NULL;
  memcpy(base, symbol, length + 1);
  char* suffix = strstr(base, "_v2");
  if (suffix == NULL) suffix = strstr(base, "_v3");
  if (suffix == NULL) suffix = strstr(base, "_ptsz");
  if (suffix != NULL) *suffix = '\0';
  return GetProcAddress(g_module, base);
}

__declspec(dllexport) CUresult WINAPI cuGetProcAddress(
    const char* symbol, void** function, int cuda_version,
    uint64_t flags, void* unused_query_status) {
  (void)unused_query_status;
  if (symbol == NULL || function == NULL) return CUDA_ERROR_INVALID_VALUE;
  FARPROC address = resolve_symbol(symbol);
  *function = (void*)address;
  char fields[640];
  _snprintf(fields, sizeof(fields),
            "\"symbol\":\"%s\",\"cuda_version\":%d,"
            "\"flags\":%llu,\"resolved\":%s",
            symbol, cuda_version, (unsigned long long)flags,
            address != NULL ? "true" : "false");
  fields[sizeof(fields) - 1] = '\0';
  trace_call("cuGetProcAddress", fields);
  return address != NULL ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

__declspec(dllexport) CUresult WINAPI cuGetErrorString(CUresult error,
                                                       const char** text) {
  static const char success[] = "CUDA_SUCCESS (NeuralStudio trace device)";
  static const char failure[] = "CUDA trace-device error";
  if (text != NULL) *text = error == CUDA_SUCCESS ? success : failure;
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuCtxGetDevice(int* device) {
  if (device == NULL) return CUDA_ERROR_INVALID_VALUE;
  *device = 0;
  trace_call("cuCtxGetDevice", "\"device\":0");
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuCtxSetCurrent(void* context) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"context\":\"%p\"", context);
  trace_call("cuCtxSetCurrent", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuCtxSynchronize(void) {
  trace_call("cuCtxSynchronize", NULL);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuDeviceGetAttribute(int* value,
                                                           int attribute,
                                                           int device) {
  (void)device;
  if (value == NULL) return CUDA_ERROR_INVALID_VALUE;
  switch (attribute) {
    case 1: *value = 1024; break;
    case 2: *value = 1024; break;
    case 3: *value = 1024; break;
    case 4: *value = 64; break;
    case 5: *value = 2147483647; break;
    case 6: *value = 65535; break;
    case 7: *value = 65535; break;
    case 8: *value = 101376; break;
    case 10: *value = 32; break;
    case 12: *value = 65536; break;
    case 13: *value = 2400000; break;
    case 16: *value = 170; break;
    case 17: *value = 1; break;
    case 18: *value = 0; break;
    case 19: *value = 1; break;
    case 20: *value = 0; break;
    case 21: *value = 131072; break;
    case 22: *value = 131072; break;
    case 23: *value = 65536; break;
    case 33: *value = 1; break;
    case 34: *value = 0; break;
    case 35: *value = 0; break;
    case 36: *value = 14000000; break;
    case 37: *value = 512; break;
    case 38: *value = 100663296; break;
    case 39: *value = 1536; break;
    case 40: *value = 5; break;
    case 41: *value = 1; break;
    case 75: *value = 12; break;
    case 76: *value = 0; break;
    case 81: *value = 233472; break;
    case 83: *value = 1; break;
    case 89: *value = 1; break;
    default: *value = 1; break;
  }
  char fields[128];
  _snprintf(fields, sizeof(fields),
            "\"attribute\":%d,\"value\":%d", attribute, *value);
  trace_call("cuDeviceGetAttribute", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuDeviceGetLuid(char* luid,
                                                      unsigned int* node_mask,
                                                      int device) {
  (void)device;
  if (luid == NULL || node_mask == NULL) return CUDA_ERROR_INVALID_VALUE;
  memset(luid, 0x42, 8);
  *node_mask = 1;
  trace_call("cuDeviceGetLuid", "\"node_mask\":1");
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuDeviceGetUuid(void* uuid, int device) {
  (void)device;
  if (uuid == NULL) return CUDA_ERROR_INVALID_VALUE;
  memset(uuid, 0x24, 16);
  trace_call("cuDeviceGetUuid", NULL);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemAlloc(CUdeviceptr* pointer,
                                                 size_t size) {
  if (pointer == NULL) return CUDA_ERROR_INVALID_VALUE;
  void* allocation = allocate_zeroed(size);
  if (allocation == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  *pointer = (CUdeviceptr)(uintptr_t)allocation;
  char fields[160];
  _snprintf(fields, sizeof(fields),
            "\"pointer\":\"0x%llx\",\"bytes\":%llu",
            (unsigned long long)*pointer, (unsigned long long)size);
  trace_call("cuMemAlloc", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemFree(CUdeviceptr pointer) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"pointer\":\"0x%llx\"",
            (unsigned long long)pointer);
  trace_call("cuMemFree", fields);
  release_allocation((void*)(uintptr_t)pointer);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemAllocHost(void** pointer,
                                                     size_t size) {
  if (pointer == NULL) return CUDA_ERROR_INVALID_VALUE;
  *pointer = allocate_zeroed(size);
  if (*pointer == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  char fields[160];
  _snprintf(fields, sizeof(fields),
            "\"pointer\":\"%p\",\"bytes\":%llu", *pointer,
            (unsigned long long)size);
  trace_call("cuMemAllocHost", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemFreeHost(void* pointer) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"pointer\":\"%p\"", pointer);
  trace_call("cuMemFreeHost", fields);
  release_allocation(pointer);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemcpyHtoDAsync(
    CUdeviceptr destination, const void* source, size_t bytes, void* stream) {
  if (destination == 0 || source == NULL) return CUDA_ERROR_INVALID_VALUE;
  memcpy((void*)(uintptr_t)destination, source, bytes);
  const LONG transfer_id = InterlockedIncrement(&g_next_transfer_id);
  dump_bytes("transfer", transfer_id, source, bytes);
  char fields[224];
  _snprintf(fields, sizeof(fields),
            "\"transfer_id\":%ld,\"destination\":\"0x%llx\","
            "\"source\":\"%p\","
            "\"bytes\":%llu,\"stream\":\"%p\"",
            transfer_id, (unsigned long long)destination, source,
            (unsigned long long)bytes, stream);
  trace_call("cuMemcpyHtoDAsync", fields);
  return CUDA_SUCCESS;
}

// The signed runtime does not request this entry through cuGetProcAddress.
// It exists only so the real-driver forwarding proxy can exercise its
// post-launch snapshot path against this CPU-backed trace device.
__declspec(dllexport) CUresult WINAPI cuMemcpyDtoH(
    void* destination, CUdeviceptr source, size_t bytes) {
  if (destination == NULL || source == 0) return CUDA_ERROR_INVALID_VALUE;
  memcpy(destination, (const void*)(uintptr_t)source, bytes);
  char fields[160];
  _snprintf(fields, sizeof(fields),
            "\"source\":\"0x%llx\",\"bytes\":%llu",
            (unsigned long long)source, (unsigned long long)bytes);
  trace_call("cuMemcpyDtoH", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemsetD32Async(
    CUdeviceptr destination, unsigned int value, size_t count, void* stream) {
  if (destination == 0) return CUDA_ERROR_INVALID_VALUE;
  unsigned int* output = (unsigned int*)(uintptr_t)destination;
  for (size_t index = 0; index < count; ++index) output[index] = value;
  char fields[224];
  _snprintf(fields, sizeof(fields),
            "\"destination\":\"0x%llx\",\"value\":%u,"
            "\"count\":%llu,\"stream\":\"%p\"",
            (unsigned long long)destination, value,
            (unsigned long long)count, stream);
  trace_call("cuMemsetD32Async", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMemcpy2D(const void* descriptor) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"descriptor\":\"%p\"", descriptor);
  trace_call("cuMemcpy2D", fields);
  return descriptor != NULL ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

__declspec(dllexport) CUresult WINAPI cuModuleLoadData(void** module,
                                                       const void* image) {
  if (module == NULL || image == NULL) return CUDA_ERROR_INVALID_VALUE;
  TraceModule* result = (TraceModule*)allocate_zeroed(sizeof(*result));
  if (result == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  result->id = InterlockedIncrement(&g_next_id);
  result->image = image;
  *module = result;
  const unsigned char* bytes = (const unsigned char*)image;
  size_t image_size = 0;
  if (*(const uint32_t*)bytes == 0xBA55ED50u) {
    const uint16_t header_size = *(const uint16_t*)(bytes + 6);
    const uint64_t payload_size = *(const uint64_t*)(bytes + 8);
    if (header_size >= 16 && payload_size <= 64ull * 1024ull * 1024ull) {
      image_size = (size_t)header_size + (size_t)payload_size;
    }
  }
  dump_bytes("module", result->id, image, image_size);
  char fields[256];
  _snprintf(fields, sizeof(fields),
            "\"module_id\":%ld,\"image\":\"%p\","
            "\"magic\":\"%02x%02x%02x%02x\",\"bytes\":%llu,"
            "\"dump\":\"module-%04ld.bin\"",
            result->id, image, bytes[0], bytes[1], bytes[2], bytes[3],
            (unsigned long long)image_size, result->id);
  trace_call("cuModuleLoadData", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuModuleUnload(void* module) {
  TraceModule* typed = (TraceModule*)module;
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"module_id\":%ld",
            typed != NULL ? typed->id : 0);
  trace_call("cuModuleUnload", fields);
  release_allocation(module);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuModuleGetFunction(
    void** function, void* module, const char* name) {
  if (function == NULL || module == NULL || name == NULL) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  TraceFunction* result = (TraceFunction*)allocate_zeroed(sizeof(*result));
  if (result == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  result->id = InterlockedIncrement(&g_next_id);
  result->module = (TraceModule*)module;
  strncpy(result->name, name, sizeof(result->name) - 1);
  *function = result;
  char fields[640];
  _snprintf(fields, sizeof(fields),
            "\"function_id\":%ld,\"module_id\":%ld,\"name\":\"%s\"",
            result->id, result->module->id, result->name);
  trace_call("cuModuleGetFunction", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuFuncSetAttribute(
    void* function, int attribute, int value) {
  TraceFunction* typed = (TraceFunction*)function;
  char fields[192];
  _snprintf(fields, sizeof(fields),
            "\"function_id\":%ld,\"attribute\":%d,\"value\":%d",
            typed != NULL ? typed->id : 0, attribute, value);
  trace_call("cuFuncSetAttribute", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuLaunchKernel(
    void* function, unsigned int grid_x, unsigned int grid_y,
    unsigned int grid_z, unsigned int block_x, unsigned int block_y,
    unsigned int block_z, unsigned int shared_bytes, void* stream,
    void** parameters, void** extra) {
  TraceFunction* typed = (TraceFunction*)function;
  const LONG launch_id = InterlockedIncrement(&g_next_launch_id);
  const unsigned char* argument_buffer = NULL;
  size_t argument_size = 0;
  if (extra != NULL) {
    size_t index;
    for (index = 0; index < 16; index += 2) {
      void* selector = extra[index];
      if (selector == NULL) break;
      if (selector == (void*)0x1) {
        argument_buffer = (const unsigned char*)extra[index + 1];
      } else if (selector == (void*)0x2 && extra[index + 1] != NULL) {
        argument_size = *(const size_t*)extra[index + 1];
      }
    }
  }
  if (argument_size > 1024 * 1024) argument_size = 0;
  dump_bytes("arguments", launch_id, argument_buffer, argument_size);
  char fields[896];
  _snprintf(
      fields, sizeof(fields),
      "\"launch_id\":%ld,\"function_id\":%ld,\"name\":\"%s\","
      "\"grid\":[%u,%u,%u],\"block\":[%u,%u,%u],"
      "\"shared_bytes\":%u,\"stream\":\"%p\","
      "\"parameters\":\"%p\",\"extra\":\"%p\","
      "\"argument_bytes\":%llu,\"argument_dump\":"
      "\"arguments-%04ld.bin\"",
      launch_id, typed != NULL ? typed->id : 0,
      typed != NULL ? typed->name : "<null>", grid_x, grid_y, grid_z,
      block_x, block_y, block_z, shared_bytes, stream, parameters, extra,
      (unsigned long long)argument_size, launch_id);
  trace_call("cuLaunchKernel", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuArrayCreate(void** array,
                                                    const void* descriptor) {
  if (array == NULL || descriptor == NULL) return CUDA_ERROR_INVALID_VALUE;
  const CudaArrayDescriptor* input = (const CudaArrayDescriptor*)descriptor;
  TraceArray* result = (TraceArray*)allocate_zeroed(sizeof(*result));
  if (result == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  result->id = InterlockedIncrement(&g_next_array_id);
  result->width = input->width;
  result->height = input->height;
  result->format = input->format;
  result->channels = input->channels;
  *array = result;
  char fields[192];
  _snprintf(fields, sizeof(fields),
            "\"array_id\":%ld,\"pointer\":\"%p\","
            "\"width\":%llu,\"height\":%llu,\"format\":%d,"
            "\"channels\":%u",
            result->id, result,
            (unsigned long long)result->width,
            (unsigned long long)result->height, result->format,
            result->channels);
  trace_call("cuArrayCreate", fields);
  return *array != NULL ? CUDA_SUCCESS : CUDA_ERROR_OUT_OF_MEMORY;
}

__declspec(dllexport) CUresult WINAPI cuArray3DCreate(void** array,
                                                      const void* descriptor) {
  if (array == NULL || descriptor == NULL) return CUDA_ERROR_INVALID_VALUE;
  const CudaArray3DDescriptor* input =
      (const CudaArray3DDescriptor*)descriptor;
  TraceArray* result = (TraceArray*)allocate_zeroed(sizeof(*result));
  if (result == NULL) return CUDA_ERROR_OUT_OF_MEMORY;
  result->id = InterlockedIncrement(&g_next_array_id);
  result->width = input->width;
  result->height = input->height;
  result->format = input->format;
  result->channels = input->channels;
  *array = result;
  char fields[256];
  _snprintf(fields, sizeof(fields),
            "\"array_id\":%ld,\"pointer\":\"%p\","
            "\"width\":%llu,\"height\":%llu,\"depth\":%llu,"
            "\"format\":%d,\"channels\":%u,\"flags\":%u",
            result->id, result, (unsigned long long)result->width,
            (unsigned long long)result->height,
            (unsigned long long)input->depth, result->format,
            result->channels, input->flags);
  trace_call("cuArray3DCreate", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuArrayDestroy(void* array) {
  TraceArray* typed = (TraceArray*)array;
  char fields[128];
  _snprintf(fields, sizeof(fields),
            "\"array_id\":%ld,\"pointer\":\"%p\"",
            typed != NULL ? typed->id : 0, array);
  trace_call("cuArrayDestroy", fields);
  release_allocation(array);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuArrayGetDescriptor(void* descriptor,
                                                           void* array) {
  if (descriptor == NULL || array == NULL) return CUDA_ERROR_INVALID_VALUE;
  const TraceArray* input = (const TraceArray*)array;
  CudaArrayDescriptor* output = (CudaArrayDescriptor*)descriptor;
  output->width = input->width;
  output->height = input->height;
  output->format = input->format;
  output->channels = input->channels;
  trace_call("cuArrayGetDescriptor", NULL);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuMipmappedArrayDestroy(void* array) {
  trace_call("cuMipmappedArrayDestroy", NULL);
  release_allocation(array);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuTexObjectCreate(
    uint64_t* object, const void* resource, const void* texture,
    const void* view) {
  (void)texture;
  (void)view;
  if (object == NULL || resource == NULL) return CUDA_ERROR_INVALID_VALUE;
  *object = (uint64_t)InterlockedIncrement(&g_next_id);
  if (add_object(g_texture_objects, *object,
                 (const CudaResourceDesc*)resource) == NULL) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  const CudaResourceDesc* typed_resource =
      (const CudaResourceDesc*)resource;
  char fields[192];
  _snprintf(fields, sizeof(fields),
            "\"object\":%llu,\"resource_type\":%d,"
            "\"array_pointer\":\"%p\"",
            (unsigned long long)*object, typed_resource->resource_type,
            typed_resource->resource.array.array);
  trace_call("cuTexObjectCreate", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuTexObjectDestroy(uint64_t object) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"object\":%llu",
            (unsigned long long)object);
  trace_call("cuTexObjectDestroy", fields);
  TraceObject* found = find_object(g_texture_objects, object);
  if (found != NULL) found->active = 0;
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuTexObjectGetResourceDesc(
    void* descriptor, uint64_t object) {
  if (descriptor == NULL) return CUDA_ERROR_INVALID_VALUE;
  TraceObject* found = find_object(g_texture_objects, object);
  if (found == NULL) return CUDA_ERROR_INVALID_VALUE;
  memcpy(descriptor, &found->resource, sizeof(found->resource));
  trace_call("cuTexObjectGetResourceDesc", NULL);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuSurfObjectCreate(
    uint64_t* object, const void* resource) {
  if (object == NULL || resource == NULL) return CUDA_ERROR_INVALID_VALUE;
  *object = (uint64_t)InterlockedIncrement(&g_next_id);
  if (add_object(g_surface_objects, *object,
                 (const CudaResourceDesc*)resource) == NULL) {
    return CUDA_ERROR_OUT_OF_MEMORY;
  }
  const CudaResourceDesc* typed_resource =
      (const CudaResourceDesc*)resource;
  char fields[192];
  _snprintf(fields, sizeof(fields),
            "\"object\":%llu,\"resource_type\":%d,"
            "\"array_pointer\":\"%p\"",
            (unsigned long long)*object, typed_resource->resource_type,
            typed_resource->resource.array.array);
  trace_call("cuSurfObjectCreate", fields);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuSurfObjectDestroy(uint64_t object) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"object\":%llu",
            (unsigned long long)object);
  trace_call("cuSurfObjectDestroy", fields);
  TraceObject* found = find_object(g_surface_objects, object);
  if (found != NULL) found->active = 0;
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuSurfObjectGetResourceDesc(
    void* descriptor, uint64_t object) {
  if (descriptor == NULL) return CUDA_ERROR_INVALID_VALUE;
  TraceObject* found = find_object(g_surface_objects, object);
  if (found == NULL) return CUDA_ERROR_INVALID_VALUE;
  memcpy(descriptor, &found->resource, sizeof(found->resource));
  trace_call("cuSurfObjectGetResourceDesc", NULL);
  return CUDA_SUCCESS;
}

__declspec(dllexport) CUresult WINAPI cuDestroyExternalMemory(void* memory) {
  char fields[96];
  _snprintf(fields, sizeof(fields), "\"memory\":\"%p\"", memory);
  trace_call("cuDestroyExternalMemory", fields);
  return CUDA_SUCCESS;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
  (void)reserved;
  if (reason == DLL_PROCESS_ATTACH) {
    g_module = instance;
    InitializeCriticalSection(&g_log_lock);
    trace_line("{\"event\":\"nvcuda_trace_loaded\",\"device\":\"sm_120\"}");
  } else if (reason == DLL_PROCESS_DETACH) {
    trace_line("{\"event\":\"nvcuda_trace_unloaded\"}");
    DeleteCriticalSection(&g_log_lock);
  }
  return TRUE;
}
