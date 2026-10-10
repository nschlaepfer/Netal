#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_conv1 {
inline constexpr std::uint32_t kInputWidth=160,kInputHeight=92,kInputChannels=32,kOutputWidth=80,kOutputHeight=46,kOutputChannels=64;
inline constexpr std::size_t kInputBytes=std::size_t{kInputWidth}*kInputHeight*kInputChannels*2,kOutputBytes=std::size_t{kOutputWidth}*kOutputHeight*kOutputChannels*2,kWeightBytes=36864,kBiasBytes=128,kModuleBytes=5240;
inline constexpr char kModuleSha256[]="c36bedc4fe275c9891dbf35fc79f3377e71f91e7150a2d6b2ae7eb7f0d0548f7",kWeightSha256[]="c314eefc0c076af12dc246cbffbcf32fe78b50640b451567aa85832eff367092",kBiasSha256[]="9718b129b9bf59ee1a683c6001880cfb910a5e95e9b2338e4fcfc742d7f83be7",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
inline std::uint64_t SplitMix64(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30u))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27u))*0x94D049BB133111EBull;return v^(v>>31u);}
inline std::vector<std::uint8_t> GenerateInput(){constexpr std::array<std::uint16_t,11> values={0x0000,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t> r(kInputBytes);constexpr std::uint64_t seed=0xA4093822299F31D0ull,step=0x9E3779B97F4A7C15ull;for(std::size_t i=0;i<r.size()/2;++i){auto bits=values[SplitMix64(seed+std::uint64_t{i}*step)%values.size()];r[i*2]=std::uint8_t(bits);r[i*2+1]=std::uint8_t(bits>>8u);}return r;}
}
