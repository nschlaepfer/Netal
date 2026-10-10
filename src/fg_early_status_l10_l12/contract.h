#pragma once

#include <cstddef>
#include <cstdint>

namespace fg_early_status_l10_l12 {

inline constexpr std::size_t kModuleBytes = 4024;
inline constexpr char kModuleSha256[] =
    "c8b8c082139005c1bf059f156827c4e331f9a6579ded1b2c9646198b1cc2abf7";
inline constexpr std::uint32_t kCount = 1;
inline constexpr std::uint32_t kResetValue = 1;
inline constexpr std::uint32_t kGeneratedValue = 0;
inline constexpr char kResetTraceRawParameterSha256[] =
    "7d61d31336110db4d7ca50f8ed86e5a512bf414641011c84616c87d6dcb02331";
inline constexpr char kGeneratedTraceRawParameterSha256[] =
    "45ea95d049547e4afa55733c880dd3f2681c9fb161b7d47202c736de0ee8aa3d";
inline constexpr char kResetAddressFreeParameterSha256[] =
    "b342c83d7060c5852f741e8e009d375103df2e1f7c76fbc512e575cd86d7b03d";
inline constexpr char kGeneratedAddressFreeParameterSha256[] =
    "9d34149fbd1fe777eb238799054c8cbfbce372255f219f8740838def9bfd02db";
inline constexpr char kInputGenerator[] =
    "no-tensor-input-status-u32-v1";

}  // namespace fg_early_status_l10_l12
