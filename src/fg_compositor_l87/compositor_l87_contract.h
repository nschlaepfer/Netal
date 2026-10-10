#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace fg_compositor_l87 {

inline constexpr std::uint32_t kWidth = 640;
inline constexpr std::uint32_t kHeight = 360;
inline constexpr std::uint32_t kPyramidWidth = 320;
inline constexpr std::uint32_t kPyramidHeight = 180;
inline constexpr std::uint32_t kSelectedLevel = 10;
inline constexpr std::uint32_t kGridX = 40;
inline constexpr std::uint32_t kGridY = 23;
inline constexpr std::uint32_t kBlockX = 256;
inline constexpr std::size_t kPyramid0Bytes =
    std::size_t{kPyramidWidth} * kPyramidHeight * 8u;
inline constexpr std::array<std::size_t, 4> kPyramidBytes = {
    460800u, 115200u, 28800u, 7360u};
inline constexpr std::size_t kMaskBytes = std::size_t{kWidth} * kHeight;
inline constexpr std::size_t kOutputBytes =
    std::size_t{kWidth} * kHeight * 4u;
inline constexpr std::size_t kPackedParameterBytes = 72;
inline constexpr std::size_t kGuardBytes = 256;
inline constexpr std::uint8_t kGuardValue = 0xc7;

inline constexpr char kModuleSha256[] =
    "6cc983a76decaed1a8c7f207a95531bece5a1ad092c4db3a0159b590c331c24b";
inline constexpr std::size_t kModuleBytes = 26024;
inline constexpr char kRawParameterSha256[] =
    "b4237dac2bf46edd1c9069d8b9e89b71f21f109c144b1b39cf1120bda1a8f54e";
inline constexpr char kAddressFreeParameterSha256[] =
    "85a388275a5df0aae3b463f2af4fee7c5cc6135c0405127e5261b38f2ab55a01";
inline constexpr char kRcpRomSha256[] =
    "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750";
inline constexpr std::size_t kRcpRomBytes = 2097152;
inline constexpr char kGenerator[] =
    "splitmix64-fg-compositor-l87-half4-mask-v1";

struct MetalParameters {
  std::uint32_t level;
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t reserved;
};
static_assert(sizeof(MetalParameters) == 16);

}  // namespace fg_compositor_l87
