#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <stdint.h>

typedef uint32_t(WINAPI* InitCuda)(uint64_t application_id,
                                   const wchar_t* application_data_path,
                                   void* cuda_device,
                                   uint32_t sdk_version,
                                   const void* parameters);
typedef uint32_t(WINAPI* ShutdownCuda)(void* cuda_device);
typedef uint32_t(WINAPI* PopulateParameters)(void* parameters);
typedef uint32_t(WINAPI* CreateCuda)(void* cuda_device, int feature,
                                     const void* parameters, void** handle);
typedef uint32_t(WINAPI* ReleaseFeature)(void* handle);
typedef uint32_t(WINAPI* RequirementsCuda)(int cuda_device,
                                           const void* discovery,
                                           void* requirement);
typedef uint32_t(WINAPI* EvaluateCuda)(const void* handle,
                                       const void* parameters,
                                       void* progress_callback);

/* Older DLSS runtimes expose the pre-device forms of three of these. They are not renames: each of
   the "1" variants takes an NVSDK_NGX_CUDADevice* in a different position - inserted at three for
   init, prepended for create, added for shutdown - so they need their own signatures here and their
   own call sites, not a symbol fallback. */
typedef uint32_t(WINAPI* InitCudaLegacy)(uint64_t application_id,
                                         const wchar_t* application_data_path,
                                         uint32_t sdk_version,
                                         const void* parameters);
typedef uint32_t(WINAPI* ShutdownCudaLegacy)(void);
typedef uint32_t(WINAPI* CreateCudaLegacy)(int feature, const void* parameters,
                                           void** handle);

/* Returned when this shim rejects its own arguments before calling the runtime.
   Deliberately outside NGX's result space: NGX failures are 0xBAD00000|n, so the value this used to
   return - 0xBAD00005 - was indistinguishable from NVSDK_NGX_Result_FAIL_InvalidParameter coming
   back from the runtime itself. That ambiguity cost real time diagnosing an older runtime, where
   the question was precisely whether the call had reached it. */
#define NEURALSTUDIO_THUNK_BAD_ARGUMENT 0xC0DE0001u

#if defined(__GNUC__)
#define NEURALSTUDIO_NOINLINE __attribute__((noinline))
#else
#define NEURALSTUDIO_NOINLINE __declspec(noinline)
#endif

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI NeuralStudioInitCUDA(
    InitCuda init, uint64_t application_id,
    const wchar_t* application_data_path, void* cuda_device,
    uint32_t sdk_version, const void* parameters) {
  if (init == NULL || application_data_path == NULL || cuda_device == NULL) {
    return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  }
  volatile uint32_t result = init(application_id, application_data_path,
                                  cuda_device, sdk_version, parameters);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI NeuralStudioShutdownCUDA(
    ShutdownCuda shutdown, void* cuda_device) {
  if (shutdown == NULL || cuda_device == NULL) return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  volatile uint32_t result = shutdown(cuda_device);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioInitCUDALegacy(InitCudaLegacy init, uint64_t application_id,
                           const wchar_t* application_data_path,
                           uint32_t sdk_version, const void* parameters) {
  /* No cuda_device to null-check here - that is the whole difference. */
  if (init == NULL || application_data_path == NULL) return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  volatile uint32_t result =
      init(application_id, application_data_path, sdk_version, parameters);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioShutdownCUDALegacy(ShutdownCudaLegacy shutdown) {
  if (shutdown == NULL) return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  volatile uint32_t result = shutdown();
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioCreateCUDALegacy(CreateCudaLegacy create, int feature,
                             const void* parameters, void** handle) {
  if (create == NULL || parameters == NULL || handle == NULL) {
    return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  }
  volatile uint32_t result = create(feature, parameters, handle);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioPopulateCUDA(PopulateParameters populate, void* parameters) {
  if (populate == NULL || parameters == NULL) return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  volatile uint32_t result = populate(parameters);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioCreateCUDA(CreateCuda create, void* cuda_device, int feature,
                       const void* parameters, void** handle) {
  if (create == NULL || cuda_device == NULL || parameters == NULL ||
      handle == NULL) {
    return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  }
  volatile uint32_t result =
      create(cuda_device, feature, parameters, handle);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioReleaseCUDA(ReleaseFeature release, void* handle) {
  if (release == NULL || handle == NULL) return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  volatile uint32_t result = release(handle);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioRequirementsCUDA(RequirementsCuda requirements, int cuda_device,
                             const void* discovery, void* requirement) {
  if (requirements == NULL || discovery == NULL || requirement == NULL) {
    return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  }
  volatile uint32_t result =
      requirements(cuda_device, discovery, requirement);
  return result;
}

__declspec(dllexport) NEURALSTUDIO_NOINLINE uint32_t WINAPI
NeuralStudioEvaluateCUDA(EvaluateCuda evaluate, const void* handle,
                         const void* parameters, void* progress_callback) {
  if (evaluate == NULL || handle == NULL || parameters == NULL) {
    return NEURALSTUDIO_THUNK_BAD_ARGUMENT;
  }
  volatile uint32_t result =
      evaluate(handle, parameters, progress_callback);
  return result;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
  (void)instance;
  (void)reason;
  (void)reserved;
  return TRUE;
}
