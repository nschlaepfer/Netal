#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"
#include "contract.h"
namespace {
namespace c = fg_dynamic_module25;
std::vector<std::uint8_t> RF(const std::filesystem::path &p) {
  std::ifstream s(p, std::ios::binary | std::ios::ate);
  auto n = s.tellg();
  std::vector<std::uint8_t> b((std::size_t)n);
  s.seekg(0);
  s.read((char *)b.data(), n);
  return b;
}
id<MTLTexture> T(id<MTLDevice> d, MTLPixelFormat f, uint w, uint h,
                 MTLTextureUsage u, const std::vector<std::uint8_t> &b,
                 std::size_t row) {
  auto x = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f
                                                              width:w
                                                             height:h
                                                          mipmapped:NO];
  x.storageMode = MTLStorageModeShared;
  x.usage = u;
  auto t = [d newTextureWithDescriptor:x];
  [t replaceRegion:MTLRegionMake2D(0, 0, w, h)
       mipmapLevel:0
         withBytes:b.data()
       bytesPerRow:row];
  return t;
}
std::vector<std::uint8_t> TR(id<MTLTexture> t, std::size_t row) {
  std::vector<std::uint8_t> b(row * t.height);
  [t getBytes:b.data()
      bytesPerRow:row
       fromRegion:MTLRegionMake2D(0, 0, t.width, t.height)
      mipmapLevel:0];
  return b;
}
} // namespace
int main(int n, const char *v[]) {
  @autoreleasepool {
    if (n != 9)
      return 2;
    try {
      auto rom = RF(v[2]);
      auto z = std::uint8_t(strtoul(v[5], nullptr, 16));
      auto &r = c::Find(v[6]);
      bool g = std::string(v[7]) == "generated",
           art = std::string(v[8]) == "full";
      std::filesystem::path d = v[3];
      std::filesystem::create_directories(d);
      auto in0 = c::Input(r, g);
      std::vector<std::uint8_t> out0(std::size_t(r.w) * (r.h + 1), z);
      id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
      NSError *er = nil;
      auto lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:@(v[1])]
                                  error:&er];
      auto fn = [lib newFunctionWithName:@"fg_early_l8_l18"];
      auto pipe = [dev newComputePipelineStateWithFunction:fn error:&er];
      auto input = T(dev, MTLPixelFormatRG16Float, r.w, r.h,
                     MTLTextureUsageShaderRead, in0, r.w * 4u);
      auto output = T(dev, MTLPixelFormatR8Unorm, r.w, r.h + 1,
                      MTLTextureUsageShaderWrite, out0, r.w);
      auto rb = Guarded(dev, &rom, rom.size(), 0);
      struct P {
        uint w, h, mode;
      } p{r.w, r.h, 0};
      auto cmd = [[dev newCommandQueue] commandBuffer];
      auto enc = [cmd computeCommandEncoder];
      [enc setComputePipelineState:pipe];
      [enc setTexture:input atIndex:0];
      [enc setTexture:output atIndex:1];
      [enc setBuffer:rb offset:kGuardBytes atIndex:0];
      [enc setBytes:&p length:sizeof(p) atIndex:1];
      [enc dispatchThreads:MTLSizeMake(r.w, r.h, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [enc endEncoding];
      [cmd commit];
      [cmd waitUntilCompleted];
      auto in1 = TR(input, r.w * 4u);
      auto og = TR(output, r.w);
      std::vector<std::uint8_t> o(og.begin(),
                                  og.begin() + std::size_t(r.w) * r.h),
          cov(o.size());
      std::size_t bad = 0, written = 0, guard = 0;
      for (std::size_t i = 0; i < o.size(); ++i) {
        if (o[i] == 0) {
          cov[i] = 1;
          ++written;
        } else if (o[i] != z)
          ++bad;
      }
      for (std::size_t i = o.size(); i < og.size(); ++i)
        guard += og[i] != z;
      auto mm = Mismatches(in0, in1) + Mismatches(rom, Payload(rb, rom.size()));
      if (art) {
        Write(d / "output.bin", o);
        Write(d / "coverage.bin", cov);
      }
      std::ofstream s(d / "run-receipt.json");
      s << "{\n  \"schema\": \"neuralstudio.fg.dynamic-module25-run.v1\",\n  "
           "\"backend\": \"mac-metal\",\n  \"run_id\": \""
        << v[4] << "\",\n  \"process_id\": " << getpid()
        << ",\n  \"resolution\": \"" << r.name << "\",\n  \"phase\": \""
        << (g ? "generated" : "reset")
        << "\",\n  \"poison_byte\": " << unsigned(z)
        << ",\n  \"module_sha256\": \"" << c::kModuleSha
        << "\",\n  \"extent\": [" << r.w << ", " << r.h << "],\n  \"grid\": ["
        << r.gx << ", " << r.gy
        << ", 1],\n  \"mode\": 0,\n  \"raw_output_sha256\": \"" << Sha256(o)
        << "\",\n  \"coverage_sha256\": \"" << Sha256(cov)
        << "\",\n  \"written_pixels\": " << written
        << ",\n  \"caller_owned_pixels\": " << (o.size() - written)
        << ",\n  \"invalid_pixels\": " << bad
        << ",\n  \"input_mismatched_bytes\": " << mm
        << ",\n  \"guard_row_mismatched_bytes\": " << guard
        << ",\n  \"artifacts_written\": " << (art ? "true" : "false")
        << ",\n  \"expected_or_captured_output_ingress\": false,\n  "
           "\"result\": \""
        << ((!bad && !mm && !guard) ? "pass" : "fail") << "\"\n}\n";
      return bad || mm || guard ? 3 : 0;
    } catch (const std::exception &e) {
      std::cerr << e.what() << '\n';
      return 1;
    }
  }
}
