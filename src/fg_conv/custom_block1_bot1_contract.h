#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_block1_bot1 {
inline constexpr std::size_t kLatentBytes=184u*320u*16u*2u,kSkip0Bytes=368u*640u*4u*2u,kSkip1Bytes=368u*640u*1u*2u,kSkip2Bytes=368u*640u*3u*2u,kOutputBytes=368u*640u*8u*2u,kWeightBytes=2304,kBiasBytes=16,kModuleBytes=6224;
inline constexpr char kModuleSha256[]="6f82303a09e3d7f5da446047e0b9667d8da04cfd96f6b48d38c460a95c55953a",kWeightSha256[]="1514d5a5c51a81d8436de6e8e538275895dafc2e072512dbdcbe640871ec48f0",kBiasSha256[]="89f7b17382bc6f08bbbc9d416da1c4c5b72abaf0c4faf870966a2585f912b493",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
inline std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27))*0x94D049BB133111EBull;return v^(v>>31);}
inline std::vector<std::uint8_t> Input(std::size_t bytes,std::uint64_t seed){constexpr std::array<std::uint16_t,11>v={0,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t>r(bytes);for(std::size_t i=0;i<r.size()/2;++i){auto b=v[Mix(seed+std::uint64_t{i}*0xD1B54A32D192ED03ull)%v.size()];r[2*i]=std::uint8_t(b);r[2*i+1]=std::uint8_t(b>>8);}return r;}
inline std::array<std::vector<std::uint8_t>,6> Inputs(){return {Input(kLatentBytes,0x13198A2E03707344ull),Input(kLatentBytes,0x243F6A8885A308D3ull),Input(kLatentBytes,0xA4093822299F31D0ull),Input(kSkip0Bytes,0x082EFA98EC4E6C89ull),Input(kSkip1Bytes,0x452821E638D01377ull),Input(kSkip2Bytes,0xBE5466CF34E90C6Cull)};}
}
