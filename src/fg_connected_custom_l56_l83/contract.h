#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
namespace fg_connected_custom {
inline constexpr std::uint32_t kWidth=640,kHeight=368;
inline constexpr std::size_t kPixels=std::size_t{kWidth}*kHeight;
inline constexpr std::size_t kInputABytes=kPixels*8,kInputBBytes=kPixels*8,kInputCBytes=kPixels*2;
inline constexpr std::size_t kL56Bytes=kPixels*20,kL57Bytes=184u*320u*32u*2u,kL58Bytes=92u*160u*32u*2u,kLow64Bytes=46u*80u*64u*2u;
inline constexpr std::size_t kL68PlaneBytes=92u*160u*32u*2u,kL69Out0Bytes=184u*320u*4u*2u,kL69Out1Bytes=184u*320u*2u,kL69Out2Bytes=184u*320u*3u*2u;
inline constexpr std::size_t kL70Out0Bytes=kPixels*4u*2u,kL70Out1Bytes=kPixels*2u,kL70Out2Bytes=kPixels*3u*2u,kL71Bytes=kPixels*18u*2u;
inline constexpr char kManifestSha256[]="2a1179d4a25be030f772cd90ca7d017ed6eac9291917f7d43ea9b0cd5ac6a294";
struct ModelSpec{int weight_token;std::size_t weight_bytes;const char*weight_sha;int bias_token;std::size_t bias_bytes;const char*bias_sha;};
inline constexpr std::array<ModelSpec,13> kModels={{
{106,9216,"14058e1781a3e362cd9f0106cf5c3fcd985ede67f1391ff3dfc73bceeceeae38",108,64,"349bbe2fcd8fc31407bfe238f4889871f0d64dd593b4146b59eca837fcc9a77f"},
{110,18432,"be4a2cdb5d93c3b3ee3ff5ee8663d27017488664ca8682313740ff99aec4b829",112,64,"097bd45ed222a90e9311fd89bbb8bca8db92760ec932380d3c44e29eb3870b69"},
{114,36864,"c314eefc0c076af12dc246cbffbcf32fe78b50640b451567aa85832eff367092",116,128,"9718b129b9bf59ee1a683c6001880cfb910a5e95e9b2338e4fcfc742d7f83be7"},
{118,73728,"819fa5692e109843574dd0608312014a32abc7e7499784066142a4d82c486985",120,128,"ba63d29aecb3781452154a32c8e771471f49ff4cd1d7f56d06a1cd89081cd461"},
{122,73728,"5f4ac25a208ceb920b34fae48b7cee7107cd3e8da565e0a689a2cab93fe7ce93",124,128,"a7548519b77593645b6d5c46777597c0eba48255cd9f079f77d166da4c406bb5"},
{126,73728,"42a3ff7432377294d3c1fbdbe3c9876d13b989afb3952dd8ce964c531eae47a3",128,128,"f0dcd605c4a628cd940630048aeca237e2f563ec5219f265d93d382abef569ff"},
{130,73728,"672b9d32f4510335489894e12379bdc53a82ba27474c4484f6a58e1a18657af3",132,128,"591d8e9cb8e6c383b40636a814e714e676857deecfea6331792d84131edbf618"},
{134,73728,"54356c51efe3b2a4aa798275b73e1f16857d1976eaa99bfffe32c1d6f9905eae",136,128,"8f2b5996e111aa7d5d9711f0c58cea678de5974e3a733ae6a1a8bf1e0ec5840c"},
{138,73728,"7323eede30ef99e54156437d5f0430c266c1bf076018ad942afd48e7a62eca47",140,128,"7b48edd449f5442b4af26a7ac5d9e610aa0f087422347d7acfe6fa2d46b573d8"},
{142,73728,"971c7195720f720d7ce59564d9ee85259adfd0f66b2f2c426ca7f43c9290877c",144,128,"755ff7524a6855c81f50da913721a63f8fe39aa9f8106e577f14a0575a24633e"},
{146,73728,"145a29061230186864b92a934c6dcafa00ddede91381002ce7086e42c13b9490",148,128,"89e683bf9d2d108add777b4fb521d9d72940c66270cf96d1735943bf423bb93d"},
{150,110592,"6b59f16f1be43617ebf0c94ff29cf7c8b7a2f2c7b7bfb41df8f9ede585e70684",152,192,"c3e301df2e15c48e0c2fb35fc636b944653c6f91071a101d71b256b740224d29"},
{154,4608,"4d477655f6ec78e0e6a0d3c41e0848e537091c0c311fb04d986ad8891823e4e8",156,16,"0a9a85630034bb3dc5442cd52381424c1e7b9c676014b764477341005cb2f998"}}};
struct ModuleSpec{int token;std::size_t bytes;const char*sha;const char*function;};
inline constexpr std::array<ModuleSpec,9> kModules={{
{53,912,"63b7e8105e00b04049e1004a8df23998074fb04b2b6637171da379b4a967d8c3","k_initial_merge"},
{54,4192,"a371abef8e41995064c25e171d0f2f9119a24ad4b671041be16c7b639a7c68a2","custom_block0_convPre_kernel"},
{55,5384,"8b0501ef1e90ee5c0621138406982cb06822de06de930184ce40f7f4d6e1709e","custom_block0_conv0_c8_kernel"},
{56,5240,"c36bedc4fe275c9891dbf35fc79f3377e71f91e7150a2d6b2ae7eb7f0d0548f7","custom_block0_conv1_kernel"},
{57,5352,"d2b6ef4d2533614364bffab475f98a32f3c2cacbe02e0fa5825588dc5dfa3b4d","custom_block0_conv2_kernel"},
{58,4840,"3acb96881f20a50e575e2b2959f0aa433e03e3da20fbf624ae6bb183d77a3e48","custom_block0_conv_bot0_hf_kernel"},
{59,6976,"4d35017769f6338b6f8ba3ac3079756c32410eeacca7f817e48ad5d7dc59eb3c","custom_block0_conv_bot1_hf_kernel"},
{60,2112,"1d19bfddd2affc5587ec19829259b070404e44c28a6f0c570a4b7769fd0dbba6","custom_upsample_hf_kernel"},
{61,5752,"184542107e98be3dc6229251c19cae4ecc44a7cc1b78cbf741b7fb3b8d81f060","k_central_block"}}};
inline std::uint64_t Mix(std::uint64_t v){v+=0x9E3779B97F4A7C15ull;v=(v^(v>>30))*0xBF58476D1CE4E5B9ull;v=(v^(v>>27))*0x94D049BB133111EBull;return v^(v>>31);}
inline std::vector<std::uint8_t> External(std::size_t bytes,std::uint64_t seed){constexpr std::array<std::uint16_t,16>v={0,0x8000,0x2800,0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400,0xb400,0x3555,0xb555,0x3800,0xb800,0x3a00,0xba00};std::vector<std::uint8_t>r(bytes);for(std::size_t i=0;i<bytes/2;++i){auto b=v[Mix(seed+std::uint64_t{i}*0xD1B54A32D192ED03ull)%v.size()];r[2*i]=std::uint8_t(b);r[2*i+1]=std::uint8_t(b>>8);}return r;}
struct ExternalInputs{std::vector<std::uint8_t>a,b,c;};
inline ExternalInputs Generate(){return {External(kInputABytes,0x243F6A8885A308D3ull),External(kInputBBytes,0x13198A2E03707344ull),External(kInputCBytes,0xA4093822299F31D0ull)};}
}
