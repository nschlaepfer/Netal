#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_bot0{inline constexpr std::size_t kInputBytes=46u*80u*64u*2u,kOutputPlaneBytes=92u*160u*32u*2u,kOutputBytes=3u*kOutputPlaneBytes,kWeightBytes=110592,kBiasBytes=192,kModuleBytes=4840;inline constexpr char kModuleSha256[]="3acb96881f20a50e575e2b2959f0aa433e03e3da20fbf624ae6bb183d77a3e48",kWeightSha256[]="6b59f16f1be43617ebf0c94ff29cf7c8b7a2f2c7b7bfb41df8f9ede585e70684",kBiasSha256[]="c3e301df2e15c48e0c2fb35fc636b944653c6f91071a101d71b256b740224d29",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";inline std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27))*0x94D049BB133111EBull;return v^(v>>31);}inline std::vector<std::uint8_t>Input(){constexpr std::array<std::uint16_t,11>v={0,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t>r(kInputBytes);for(std::size_t i=0;i<r.size()/2;++i){auto b=v[Mix(0xC0AC29B7C97C50DDull+std::uint64_t{i}*0x9E3779B97F4A7C15ull)%v.size()];r[2*i]=std::uint8_t(b);r[2*i+1]=std::uint8_t(b>>8);}return r;}}
