#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_conv0_c8 {
inline constexpr std::uint32_t kInputWidth=320,kInputHeight=184,kInputChannels=32,kOutputWidth=160,kOutputHeight=92,kOutputChannels=32;
inline constexpr std::size_t kInputBytes=std::size_t{kInputWidth}*kInputHeight*kInputChannels*2,kOutputBytes=std::size_t{kOutputWidth}*kOutputHeight*kOutputChannels*2,kWeightBytes=18432,kBiasBytes=64,kModuleBytes=5384;
inline constexpr char kModuleSha256[]="8b0501ef1e90ee5c0621138406982cb06822de06de930184ce40f7f4d6e1709e",kWeightSha256[]="be4a2cdb5d93c3b3ee3ff5ee8663d27017488664ca8682313740ff99aec4b829",kBiasSha256[]="097bd45ed222a90e9311fd89bbb8bca8db92760ec932380d3c44e29eb3870b69",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
inline std::uint64_t SplitMix64(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30u))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27u))*0x94D049BB133111EBull;return v^(v>>31u);}
inline std::vector<std::uint8_t> GenerateInput(){constexpr std::array<std::uint16_t,11> values={0x0000,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t> r(kInputBytes);constexpr std::uint64_t seed=0x13198A2E03707344ull,step=0x8EBC6AF09C88C6E3ull;for(std::size_t i=0;i<r.size()/2;++i){auto bits=values[SplitMix64(seed+std::uint64_t{i}*step)%values.size()];r[i*2]=std::uint8_t(bits);r[i*2+1]=std::uint8_t(bits>>8u);}return r;}
}
