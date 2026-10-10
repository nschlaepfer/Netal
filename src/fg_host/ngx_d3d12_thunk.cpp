#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <nvsdk_ngx.h>
#include <nvsdk_ngx_params.h>

using Populate =
    NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Parameter* parameters);
using Init = NVSDK_NGX_Result(NVSDK_CONV*)(
    unsigned long long application_id, const wchar_t* application_data_path,
    ID3D12Device* device, NVSDK_NGX_Version sdk_version,
    const NVSDK_NGX_Parameter* parameters);
using Create = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList* command_list, NVSDK_NGX_Feature feature,
    const NVSDK_NGX_Parameter* parameters, NVSDK_NGX_Handle** handle);
using Evaluate = NVSDK_NGX_Result(NVSDK_CONV*)(
    ID3D12GraphicsCommandList* command_list, const NVSDK_NGX_Handle* handle,
    const NVSDK_NGX_Parameter* parameters,
    PFN_NVSDK_NGX_ProgressCallback progress);
using Release = NVSDK_NGX_Result(NVSDK_CONV*)(NVSDK_NGX_Handle* handle);
using Shutdown =
    NVSDK_NGX_Result(NVSDK_CONV*)(ID3D12Device* device);
using Requirements = NVSDK_NGX_Result(NVSDK_CONV*)(
    IDXGIAdapter* adapter,
    const NVSDK_NGX_FeatureDiscoveryInfo* discovery,
    NVSDK_NGX_FeatureRequirement* requirement);

// The signed snippet verifies parts of its call chain. This module is staged as
// nvngx.dll and performs only typed forwarding; it does not emulate NGX core.
extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioPopulateD3D12(Populate populate, NVSDK_NGX_Parameter* parameters) {
  if (populate == nullptr || parameters == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return populate(parameters);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioInitD3D12(Init init, unsigned long long application_id,
                      const wchar_t* application_data_path,
                      ID3D12Device* device, NVSDK_NGX_Version sdk_version,
                      const NVSDK_NGX_Parameter* parameters) {
  if (init == nullptr || application_data_path == nullptr || device == nullptr ||
      parameters == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return init(application_id, application_data_path, device, sdk_version,
              parameters);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioCreateD3D12(Create create,
                        ID3D12GraphicsCommandList* command_list,
                        NVSDK_NGX_Feature feature,
                        const NVSDK_NGX_Parameter* parameters,
                        NVSDK_NGX_Handle** handle) {
  if (create == nullptr || command_list == nullptr || parameters == nullptr ||
      handle == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return create(command_list, feature, parameters, handle);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioEvaluateD3D12(Evaluate evaluate,
                          ID3D12GraphicsCommandList* command_list,
                          const NVSDK_NGX_Handle* handle,
                          const NVSDK_NGX_Parameter* parameters,
                          PFN_NVSDK_NGX_ProgressCallback progress) {
  if (evaluate == nullptr || command_list == nullptr || handle == nullptr ||
      parameters == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return evaluate(command_list, handle, parameters, progress);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioReleaseD3D12(Release release, NVSDK_NGX_Handle* handle) {
  if (release == nullptr || handle == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return release(handle);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioShutdownD3D12(Shutdown shutdown, ID3D12Device* device) {
  if (shutdown == nullptr || device == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return shutdown(device);
}

extern "C" __declspec(dllexport) NVSDK_NGX_Result NVSDK_CONV
NeuralStudioRequirementsD3D12(
    Requirements requirements, IDXGIAdapter* adapter,
    const NVSDK_NGX_FeatureDiscoveryInfo* discovery,
    NVSDK_NGX_FeatureRequirement* requirement) {
  if (requirements == nullptr || adapter == nullptr || discovery == nullptr ||
      requirement == nullptr) {
    return NVSDK_NGX_Result_FAIL_InvalidParameter;
  }
  return requirements(adapter, discovery, requirement);
}

BOOL WINAPI DllMain(HINSTANCE, DWORD, LPVOID) { return TRUE; }
