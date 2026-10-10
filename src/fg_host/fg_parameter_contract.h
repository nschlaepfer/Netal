#pragma once

#include <nvsdk_ngx_defs_dlssg.h>

#include "ngx_parameter_store.h"

#include <cstdint>

namespace neuralstudio::fg {

struct SubrectContract {
  std::uint32_t x;
  std::uint32_t y;
  std::uint32_t width;
  std::uint32_t height;
};

inline void SetExtent(ParameterStore& parameters, const char* x_name,
                      const char* y_name, const char* width_name,
                      const char* height_name,
                      const SubrectContract& extent) {
  parameters.SetU32(x_name, extent.x);
  parameters.SetU32(y_name, extent.y);
  parameters.SetU32(width_name, extent.width);
  parameters.SetU32(height_name, extent.height);
}

inline void SetBackbufferSubrectContract(ParameterStore& parameters,
                                         const SubrectContract& extent) {
  SetExtent(parameters,
            NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseX,
            NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseY,
            NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectWidth,
            NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectHeight, extent);

  // Runtime 310.8 still queries this deprecated quartet even when the modern
  // input/output-specific extents are present. NVIDIA documents the legacy
  // values as applying to both input and output backbuffers, so mirror the
  // exact selected extent rather than allowing an implicit full-frame default.
  SetExtent(parameters, NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectBaseX,
            NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectBaseY,
            NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectWidth,
            NVSDK_NGX_DLSSG_Parameter_BackbufferSubrectHeight, extent);
}

inline bool SetCreateResolutionContract(ParameterStore& parameters,
                                        std::uint32_t frame_width,
                                        std::uint32_t frame_height,
                                        const SubrectContract& extent) {
  const bool dynamic_resolution =
      extent.x != 0u || extent.y != 0u || extent.width != frame_width ||
      extent.height != frame_height;
  const std::uint32_t feature_width =
      dynamic_resolution ? extent.width : frame_width;
  const std::uint32_t feature_height =
      dynamic_resolution ? extent.height : frame_height;
  parameters.SetU32(NVSDK_NGX_Parameter_Width, feature_width);
  parameters.SetU32(NVSDK_NGX_Parameter_Height, feature_height);
  parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_Width, feature_width);
  parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_Height, feature_height);
  parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_InternalWidth,
                    feature_width);
  parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_InternalHeight,
                    feature_height);
  parameters.SetU32(NVSDK_NGX_DLSSG_Parameter_DynamicResolution,
                    dynamic_resolution ? 1u : 0u);
  return dynamic_resolution;
}

}  // namespace neuralstudio::fg
