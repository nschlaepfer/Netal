#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "fg_parameter_contract.h"

#include <cassert>
#include <climits>
#include <cstdint>
#include <iostream>

int main() {
  neuralstudio::fg::ParameterStore parameters;

  parameters.SetU32("u32_small", 17u);
  int signed_value = -1;
  assert(parameters.AsNgx()->Get("u32_small", &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 17);

  parameters.SetU32("u32_int_max", static_cast<unsigned int>(INT_MAX));
  signed_value = -1;
  assert(parameters.AsNgx()->Get("u32_int_max", &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == INT_MAX);

  parameters.SetU32("u32_too_large",
                    static_cast<unsigned int>(INT_MAX) + 1u);
  signed_value = -1;
  assert(parameters.AsNgx()->Get("u32_too_large", &signed_value) ==
         NVSDK_NGX_Result_FAIL_MissingInput);
  assert(signed_value == 0);

  parameters.SetI32("i32_zero", 0);
  unsigned int unsigned_value = 99u;
  assert(parameters.AsNgx()->Get("i32_zero", &unsigned_value) ==
         NVSDK_NGX_Result_Success);
  assert(unsigned_value == 0u);

  parameters.SetI32("i32_positive", 31);
  unsigned_value = 0u;
  assert(parameters.AsNgx()->Get("i32_positive", &unsigned_value) ==
         NVSDK_NGX_Result_Success);
  assert(unsigned_value == 31u);

  parameters.SetI32("i32_negative", -1);
  unsigned_value = 99u;
  assert(parameters.AsNgx()->Get("i32_negative", &unsigned_value) ==
         NVSDK_NGX_Result_FAIL_MissingInput);
  assert(unsigned_value == 0u);

  parameters.SetI32("typed_precedence", -7);
  parameters.SetU32("typed_precedence", 7u);
  signed_value = 0;
  unsigned_value = 0;
  assert(parameters.AsNgx()->Get("typed_precedence", &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(parameters.AsNgx()->Get("typed_precedence", &unsigned_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == -7);
  assert(unsigned_value == 7u);

  assert(parameters.AsNgx()->Get("missing", &signed_value) ==
         NVSDK_NGX_Result_FAIL_MissingInput);
  assert(parameters.AsNgx()->Get("missing", &unsigned_value) ==
         NVSDK_NGX_Result_FAIL_MissingInput);
  assert(parameters.AsNgx()->Get(nullptr, &signed_value) ==
         NVSDK_NGX_Result_FAIL_InvalidParameter);
  assert(parameters.AsNgx()->Get("invalid", static_cast<int*>(nullptr)) ==
         NVSDK_NGX_Result_FAIL_InvalidParameter);

  neuralstudio::fg::SetBackbufferSubrectContract(parameters,
                                                  {64u, 36u, 512u, 288u});
  struct ExpectedParameter {
    const char* name;
    int value;
  };
  const ExpectedParameter subrect_parameters[] = {
      {NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseX, 64},
      {NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseY, 36},
      {NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectWidth, 512},
      {NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectHeight, 288},
      {NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectBaseX, 64},
      {NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectBaseY, 36},
      {NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectWidth, 512},
      {NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectHeight, 288},
  };
  for (const ExpectedParameter& expected : subrect_parameters) {
    signed_value = -1;
    assert(parameters.AsNgx()->Get(expected.name, &signed_value) ==
           NVSDK_NGX_Result_Success);
    assert(signed_value == expected.value);
  }

  const bool full_frame_dynamic =
      neuralstudio::fg::SetCreateResolutionContract(parameters, 640u, 360u,
                                                     {0u, 0u, 640u, 360u});
  assert(!full_frame_dynamic);
  signed_value = -1;
  assert(parameters.AsNgx()->Get(
             NVSDK_NGX_DLSSG_Parameter_DynamicResolution, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 0);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 640);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 360);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_Parameter_Width, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 640);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_Parameter_Height, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 360);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_Width,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 640);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_Height,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 360);

  const bool subrect_dynamic =
      neuralstudio::fg::SetCreateResolutionContract(parameters, 640u, 360u,
                                                     {64u, 36u, 512u, 288u});
  assert(subrect_dynamic);
  signed_value = -1;
  assert(parameters.AsNgx()->Get(
             NVSDK_NGX_DLSSG_Parameter_DynamicResolution, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 1);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 512);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 288);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_Parameter_Width, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 512);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_Parameter_Height, &signed_value) ==
         NVSDK_NGX_Result_Success);
  assert(signed_value == 288);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_Width,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 512);
  assert(parameters.AsNgx()->Get(NVSDK_NGX_DLSSG_Parameter_Height,
                                 &signed_value) == NVSDK_NGX_Result_Success);
  assert(signed_value == 288);

  std::cout << "parameter_store.numeric_cross_type_contract=passed\n";
  std::cout << "parameter_store.legacy_backbuffer_subrect_contract=passed\n";
  std::cout << "parameter_store.dynamic_resolution_contract=passed\n";
  return 0;
}
