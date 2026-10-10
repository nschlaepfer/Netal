#pragma once

#include <cstddef>
#include <cstdint>

namespace fg_mid_l55 {

inline constexpr std::uint32_t kPhysicalWidth = 640;
inline constexpr std::uint32_t kPhysicalHeight = 368;
inline constexpr std::uint32_t kActiveWidth = 640;
inline constexpr std::uint32_t kActiveHeight = 360;
inline constexpr std::uint32_t kCandidateWidth = 320;
inline constexpr std::uint32_t kCandidateHeight = 180;
inline constexpr std::uint32_t kAccumulationMode = 1;
inline constexpr std::uint32_t kOptionalUiEnabled = 0;
inline constexpr std::uint32_t kGridX = 40;
inline constexpr std::uint32_t kGridY = 23;
inline constexpr std::uint32_t kBlockX = 256;
inline constexpr std::size_t kPackedParameterBytes = 240;
inline constexpr std::size_t kRgbaTextureBytes =
    std::size_t{kActiveWidth} * kActiveHeight * 8u;
inline constexpr std::size_t kRgTextureBytes =
    std::size_t{kCandidateWidth} * kCandidateHeight * 4u;
inline constexpr std::size_t kCandidateBufferBytes =
    std::size_t{kCandidateWidth} * kCandidateHeight * 8u;
inline constexpr std::size_t kLinearRgbaBytes =
    std::size_t{kPhysicalWidth} * kPhysicalHeight * 8u;
inline constexpr std::size_t kLinearMaskBytes =
    std::size_t{kPhysicalWidth} * kPhysicalHeight * 2u;
inline constexpr std::size_t kSurfaceRgbaBytes =
    std::size_t{kActiveWidth} * kActiveHeight * 8u;
inline constexpr std::size_t kSurfaceMaskBytes =
    std::size_t{kActiveWidth} * kActiveHeight;
inline constexpr std::size_t kGuardBytes = 256;
inline constexpr std::uint8_t kGuardValue = 0xc7;
inline constexpr std::uint64_t kInvalidCandidate = 0x7fffffff7fffffffull;

inline constexpr char kModuleSha256[] =
    "66a9e6bda87ec8b63c5e10752e84d07435dc4392bbe42ec142e78b0d0b624e37";
inline constexpr std::size_t kModuleBytes = 43880;
inline constexpr char kRawParameterSha256[] =
    "f46662e2dadbf89180e182202dad123a674ae94b08f9289b8d00e80f8d9f66ff";
inline constexpr char kAddressFreeParameterSha256[] =
    "6e05e0cc5b2005a313d8b4cb03e1bc495028603d224a84e709a48351238766f4";
inline constexpr char kPtxSha256[] =
    "5f1217226a5818846d655fee23cfbc05b08cd1d6d4c85d428b379b39b612f964";
inline constexpr char kDescriptorRootSha256[] =
    "d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733";
inline constexpr char kRcpRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
inline constexpr std::size_t kRcpRomBytes = 2097152;
inline constexpr char kGenerator[] =
    "splitmix64-fg-mid-l55-independent-resources-v1";

struct MetalParameters {
  std::uint32_t physical_width;
  std::uint32_t physical_height;
  std::uint32_t accumulation_mode;
  std::uint32_t candidate_width;
  std::uint32_t candidate_height;
  std::uint32_t secondary_width;
  std::uint32_t secondary_height;
  std::uint32_t active_width;
  std::uint32_t active_height;
  std::uint32_t optional_ui_enabled;
};
static_assert(sizeof(MetalParameters) == 40);

}  // namespace fg_mid_l55
