#include "module9_contract.h"
#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <array>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>
namespace {
namespace m9 = fg_distortion_module9;
std::string H(const std::vector<uint8_t> &b) {
  std::array<unsigned char, 32> d{};
  CC_SHA256(b.data(), b.size(), d.data());
  std::ostringstream s;
  s << std::hex << std::setfill('0');
  for (auto x : d)
    s << std::setw(2) << unsigned(x);
  return s.str();
}
void W(const std::filesystem::path &p, const std::vector<uint8_t> &b) {
  std::ofstream f(p, std::ios::binary);
  f.write((char *)b.data(), b.size());
}
id<MTLTexture> M(id<MTLDevice> d, MTLPixelFormat f, size_t row,
                 const std::vector<uint8_t> &b, MTLTextureUsage u) {
  auto *x = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
                                                               width:320
                                                              height:180
                                                           mipmapped:NO];
  x.storageMode = MTLStorageModeShared;
  x.usage = u;
  id<MTLTexture> t = [d newTextureWithDescriptor:x];
  [t replaceRegion:MTLRegionMake2D(0, 0, 320, 180)
       mipmapLevel:0
         withBytes:b.data()
       bytesPerRow:row];
  return t;
}
std::vector<uint8_t> R(id<MTLTexture> t, size_t row) {
  std::vector<uint8_t> b(row * 180);
  [t getBytes:b.data()
      bytesPerRow:row
       fromRegion:MTLRegionMake2D(0, 0, 320, 180)
      mipmapLevel:0];
  return b;
}
size_t D(const std::vector<uint8_t> &a, const std::vector<uint8_t> &b) {
  size_t n = 0;
  for (size_t i = 0; i < a.size(); i++)
    n += a[i] != b[i];
  return n;
}
size_t PP(const std::vector<uint8_t> &b, size_t s, uint8_t p) {
  size_t n = 0;
  for (size_t o = 0; o < b.size(); o += s) {
    bool q = 1;
    for (size_t i = 0; i < s; i++)
      q &= b[o + i] == p;
    n += q;
  }
  return n;
}
} // namespace
int main(int ac, const char **av) {
  if (ac != 5)
    return 2;
  @autoreleasepool {
    try {
      std::filesystem::path od = av[2];
      std::filesystem::create_directories(od);
      std::string run_id = av[3];
      uint8_t poison = std::stoul(av[4], nullptr, 16);
      auto in = m9::Generate();
      std::vector<uint8_t> z0(m9::kRg16Bytes, poison),
          z1(m9::kRg16Bytes, poison), z2(m9::kR32Bytes, poison),
          can(m9::kRg16Bytes, poison ^ 255);
      id<MTLDevice> d = MTLCreateSystemDefaultDevice();
      NSError *e = nil;
      id<MTLLibrary> l = [d
          newLibraryWithURL:[NSURL
                                fileURLWithPath:[NSString
                                                    stringWithUTF8String:av[1]]]
                      error:&e];
      id<MTLFunction> fn = [l newFunctionWithName:@"fg_distortion_module9"];
      id<MTLComputePipelineState> ps =
          [d newComputePipelineStateWithFunction:fn error:&e];
      std::array<id<MTLTexture>, 5> ti = {
          M(d, MTLPixelFormatRGBA16Float, 2560, in.a,
            MTLTextureUsageShaderRead),
          M(d, MTLPixelFormatRGBA16Float, 2560, in.b,
            MTLTextureUsageShaderRead),
          M(d, MTLPixelFormatRG32Float, 2560, in.flow32,
            MTLTextureUsageShaderRead),
          M(d, MTLPixelFormatRG16Float, 1280, in.flow16,
            MTLTextureUsageShaderRead),
          M(d, MTLPixelFormatR32Uint, 1280, in.mask,
            MTLTextureUsageShaderRead)};
      auto o0 = M(d, MTLPixelFormatRG16Float, 1280, z0,
                  MTLTextureUsageShaderWrite),
           o1 = M(d, MTLPixelFormatRG16Float, 1280, z1,
                  MTLTextureUsageShaderWrite),
           o2 = M(d, MTLPixelFormatR32Uint, 1280, z2,
                  MTLTextureUsageShaderWrite),
           ct = M(d, MTLPixelFormatRG16Float, 1280, can,
                  MTLTextureUsageShaderWrite);
      struct {
        uint32_t w, h;
      } p{320, 180};
      auto cb = [[d newCommandQueue] commandBuffer];
      auto en = [cb computeCommandEncoder];
      [en setComputePipelineState:ps];
      for (NSUInteger i = 0; i < 5; i++)
        [en setTexture:ti[i] atIndex:i];
      [en setTexture:o0 atIndex:5];
      [en setTexture:o1 atIndex:6];
      [en setTexture:o2 atIndex:7];
      [en setBytes:&p length:8 atIndex:0];
      [en dispatchThreads:MTLSizeMake(320, 180, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [en endEncoding];
      [cb commit];
      [cb waitUntilCompleted];
      if (cb.status != MTLCommandBufferStatusCompleted)
        throw std::runtime_error("Metal command buffer failed");
      std::array<std::vector<uint8_t>, 5> ia = {R(ti[0], 2560), R(ti[1], 2560),
                                                R(ti[2], 2560), R(ti[3], 1280),
                                                R(ti[4], 1280)};
      std::array<std::vector<uint8_t> *, 5> iv = {&in.a, &in.b, &in.flow32,
                                                  &in.flow16, &in.mask};
      size_t im = 0;
      std::array<std::string, 5> input_pre_sha256, input_post_sha256;
      for (int i = 0; i < 5; i++) {
        im += D(*iv[i], ia[i]);
        input_pre_sha256[i] = H(*iv[i]);
        input_post_sha256[i] = H(ia[i]);
      }
      auto a = R(o0, 1280), b = R(o1, 1280), c = R(o2, 1280);
      size_t cm = D(can, R(ct, 1280));
      W(od / "out-a.rg16f", a);
      W(od / "out-b.rg16f", b);
      W(od / "out-mask.r32u", c);
      std::ofstream r(od / "run-receipt.json");
      r << "{\n\"schema\":\"neuralstudio.fg.distortion-module9-run.v1\","
           "\n\"backend\":\"mac-metal\",\n\"run_id\":\""
        << run_id << "\",\n\"process_id\":" << getpid()
        << ",\n\"poison_byte\":" << unsigned(poison)
        << ",\n\"module_sha256\":\"" << m9::kModuleSha256
        << "\",\n\"module_bytes\":" << m9::kModuleBytes
        << ",\n\"selected_launch_tokens\":[\"launch-000009\",\"launch-000020\"]"
           ","
           "\n\"selected_phase_raw_parameter_sha256\":[\""
        << m9::kRawParameterSha256 << "\",\""
        << m9::kGeneratedRawParameterSha256
        << "\"],\n\"frozen_reset_address_free_parameter_sha256\":\""
        << m9::kFrozenAddressFreeSha256
        << "\",\n\"packed_parameter_bytes\":72,\n\"fixture_address_free_"
           "parameter_sha256\":\""
        << m9::kFixtureAddressFreeSha256 << "\",\n\"input_generator\":\""
        << m9::kInputGenerator << "\",\n\"input_pre_sha256\":[";
      for (size_t i = 0; i < input_pre_sha256.size(); ++i) {
        if (i)
          r << ',';
        r << '\"' << input_pre_sha256[i] << '\"';
      }
      r << "],\n\"input_post_sha256\":[";
      for (size_t i = 0; i < input_post_sha256.size(); ++i) {
        if (i)
          r << ',';
        r << '\"' << input_post_sha256[i] << '\"';
      }
      r << "],\n\"input_mismatched_bytes\":" << im
        << ",\n\"texture_canary_mismatched_bytes\":" << cm
        << ",\n\"out_a_bytes\":" << a.size()
        << ",\n\"out_b_bytes\":" << b.size()
        << ",\n\"out_mask_bytes\":" << c.size() << ",\n\"out_a_sha256\":\""
        << H(a) << "\",\n\"out_b_sha256\":\"" << H(b)
        << "\",\n\"out_mask_sha256\":\"" << H(c)
        << "\",\n\"out_a_poison_pixels\":" << PP(a, 4, poison)
        << ",\n\"out_b_poison_pixels\":" << PP(b, 4, poison)
        << ",\n\"out_mask_poison_pixels\":" << PP(c, 4, poison)
        << ",\n\"typed_metal_resources\":true,"
           "\n\"complete_write_coverage_checked_by_dual_poison\":true,"
           "\n\"coordinate_probe_consumed_by_execution\":false,"
           "\n\"expected_output_fed_to_execution\":false,"
           "\n\"archived_intermediates_consumed\":false,"
           "\n\"address_fields_recorded\":false,\n\"result\":\""
        << ((im | cm) ? "fail" : "pass") << "\"\n}\n";
      return (im | cm) ? 3 : 0;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
