#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_block1_conv0{
inline constexpr std::uint32_t kInputWidth=640,kInputHeight=368,kInputChannels=18,kOutputWidth=320,kOutputHeight=184,kOutputChannels=16;inline constexpr std::size_t kInputBytes=std::size_t{kInputWidth}*kInputHeight*kInputChannels*2,kOutputBytes=std::size_t{kOutputWidth}*kOutputHeight*kOutputChannels*2,kWeightBytes=5184,kBiasBytes=32,kModuleBytes=5096;inline constexpr char kModuleSha256[]="5a33f08403b49bbabd4dfb04f476daaac8f9fa0964befc752dc4e7fd6d2471d6",kWeightSha256[]="255701f510034a1ec801207b0542c2d49919d131c35d1337a60ad16fdc9004e6",kBiasSha256[]="8ad9d15f85e8cea4e0a5b67792f39e170e75c7db010f727c7894e2c23be5d4c1",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
inline std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27))*0x94D049BB133111EBull;return v^(v>>31);}
inline std::vector<std::uint8_t>Input(){constexpr std::array<std::uint16_t,11>v={0,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t>r(kInputBytes);for(std::size_t i=0;i<r.size()/2;++i){auto b=v[Mix(0xBE5466CF34E90C6Cull+std::uint64_t{i}*0xD1B54A32D192ED03ull)%v.size()];r[2*i]=std::uint8_t(b);r[2*i+1]=std::uint8_t(b>>8);}return r;}}
