#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>
namespace fg_custom_block1_bot0 {
inline constexpr std::size_t kInputBytes=92u*160u*32u*2u,kOutputPlaneBytes=184u*320u*16u*2u,kOutputBytes=3u*kOutputPlaneBytes,kWeightBytes=27648,kBiasBytes=96,kModuleBytes=5264;
inline constexpr char kModuleSha256[]="0f80c161e89ca4354f932ea6c3f7a0d3206e0a4fff7530d1da0fe4b162102faa",kWeightSha256[]="4ab572a084310d9f79b7878aefd73a356900ea8428320b861c6de28c114977d6",kBiasSha256[]="3ae35c7933b00a52114c7a75f0184bd480c5c814d2204271bd46da8f88772913",kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
inline std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27))*0x94D049BB133111EBull;return v^(v>>31);}
inline std::vector<std::uint8_t> Input(){constexpr std::array<std::uint16_t,11>v={0,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3800,0xb800};std::vector<std::uint8_t>r(kInputBytes);for(std::size_t i=0;i<r.size()/2;++i){auto b=v[Mix(0xA4093822299F31D0ull+std::uint64_t{i}*0xD1B54A32D192ED03ull)%v.size()];r[2*i]=std::uint8_t(b);r[2*i+1]=std::uint8_t(b>>8);}return r;}
}
