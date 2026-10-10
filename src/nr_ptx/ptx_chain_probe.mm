// GPU-resident chained runner for the transliterated NR decoder/output launches: one process, the pool in one MTLBuffer, the same
// libraries, kernels and parameter bytes as the per-launch probe; per launch the coverage counters are reset with a blit fill and
// summarised afterwards, the output textures stay resident between the post block (155) and the copy (156). Comparisons against
// the recording are done afterwards on the CPU by the driver; this program only records hashes, coverage and timings.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>
namespace fs = std::filesystem;
static std::vector<uint8_t> Read(const fs::path& p) {
  std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p.string());
  const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); if (!s) throw std::runtime_error("read failed: " + p.string()); return v;
}
static std::string Sha(const void* d, size_t n) { std::array<unsigned char, 32> h{}; CC_SHA256(d, CC_LONG(n), h.data()); std::ostringstream o; o << std::hex << std::setfill('0'); for (auto b : h) o << std::setw(2) << unsigned(b); return o.str(); }
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static unsigned JUInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.unsignedIntValue : 0u; }
int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 3) throw std::runtime_error("usage: ptx_chain_probe CHAIN.json OUTPUT_DIR");
  fs::path out = argv[2]; if (fs::exists(out)) throw std::runtime_error("preserve previous run: " + out.string());
  NSError* err = nil; NSData* jd = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]]; NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&err]; if (!chain) throw std::runtime_error("chain json");
  auto pool = Read(JStr(chain, "pool_pre")); const fs::path tables = JStr(chain, "tables"); const unsigned poison = unsigned(std::stoul(JStr(chain, "poison"), nullptr, 16)) & 0xffu;
  const bool final_only = JStr(chain, "checks") == "final";   // production mode: no per-launch pool hash / coverage scan; one final pool hash; texture and immutability checks kept
  const unsigned tw = JUInt(chain, "tex_w"), th = JUInt(chain, "tex_h"); auto texin = Read(JStr(chain, "tex_in")); if (texin.size() != size_t(tw) * th * 8) throw std::runtime_error("texture size");
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> queue = [dev newCommandQueue];
  auto mk = [&](const std::vector<uint8_t>& v, size_t extra) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + extra options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + extra); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto opt = [&](const char* name) { fs::path f = tables / name; std::vector<uint8_t> v = fs::exists(f) ? Read(f) : std::vector<uint8_t>(16, 0); return mk(v, 0); };
  id<MTLBuffer> bpool = mk(pool, 4096), brsq = opt("rtx-rsqrt-half.bin"), brcp = opt("rtx-rcp-half.bin"), brom = opt("sm120-rcp-correction-rom.bin"), bex2u = opt("unit-corrections.bin"), bex2f = opt("fraction-corrections.bin");
  id<MTLBuffer> bcov = [dev newBufferWithLength:pool.size() + 4096 options:MTLResourceStorageModeShared]; id<MTLBuffer> btexcov = [dev newBufferWithLength:size_t(tw) * th * 4 + 16 options:MTLResourceStorageModeShared];
  std::vector<uint8_t> dbgz(64 * 32 * 4, 0); id<MTLBuffer> bdbg = mk(dbgz, 0);
  MTLTextureDescriptor* dIn = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; dIn.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; dIn.storageMode = MTLStorageModeShared;
  id<MTLTexture> tex[3] = {[dev newTextureWithDescriptor:dIn], [dev newTextureWithDescriptor:dIn], [dev newTextureWithDescriptor:dIn]};   // 0: input colour, 1: post-block output, 2: final surface
  [tex[0] replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:texin.data() bytesPerRow:tw * 8];
  std::vector<uint8_t> tpoison(size_t(tw) * th * 8, uint8_t(poison)); for (int i = 1; i < 3; ++i) [tex[i] replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:tpoison.data() bytesPerRow:tw * 8];
  std::map<std::string, id<MTLLibrary>> libs; std::map<std::string, id<MTLComputePipelineState>> pipes;
  fs::create_directories(out); std::ofstream js(out / "chain-run.json"); js << "{\n \"schema\": \"neuralstudio.nr.ptx-chain-probe.v1\",\n \"pid\": " << getpid() << ",\n \"poison\": " << poison << ",\n \"pool_pre_sha256\": \"" << Sha(pool.data(), pool.size()) << "\",\n \"texture_in_sha256\": \"" << Sha(texin.data(), texin.size()) << "\",\n \"launches\": [\n";
  const auto t_all = [NSDate date]; double gpu_total = 0; bool first = true;
  for (NSDictionary* L in chain[@"launches"]) {
    const unsigned n = JUInt(L, "launch"); const std::string lib = JStr(L, "lib"), kernel = JStr(L, "kernel"); auto params = Read(JStr(L, "params")), weights = Read(JStr(L, "weights"));
    NSArray* g = L[@"grid"]; NSArray* b = L[@"block"]; const std::string texmode = JStr(L, "textures");   // "", "post" (in 0 -> out 1), "copy" (in 1 -> out 2)
    if (!libs.count(lib)) { libs[lib] = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib.c_str()]] error:&err]; if (!libs[lib]) throw std::runtime_error(err.localizedDescription.UTF8String); }
    const std::string pk = lib + "|" + kernel;
    if (!pipes.count(pk)) { pipes[pk] = [dev newComputePipelineStateWithFunction:[libs[lib] newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]] error:&err]; if (!pipes[pk]) throw std::runtime_error(kernel + ": " + std::string(err.localizedDescription.UTF8String)); }
    id<MTLBuffer> bw = mk(weights, 4096), bprm = mk(params, 64); const std::string w_sha = Sha(weights.data(), weights.size()), p_sha = Sha(params.data(), params.size());
    const auto t0 = [NSDate date]; id<MTLCommandBuffer> cmd = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder]; [blit fillBuffer:bcov range:NSMakeRange(0, pool.size() + 4096) value:0]; if (!texmode.empty()) [blit fillBuffer:btexcov range:NSMakeRange(0, size_t(tw) * th * 4 + 16) value:0]; [blit endEncoding];
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pipes[pk]];
    [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:bw offset:0 atIndex:1]; [enc setBuffer:bprm offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6];
    [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10];
    if (texmode == "post") { [enc setTexture:tex[0] atIndex:0]; [enc setTexture:tex[1] atIndex:1]; } else if (texmode == "copy") { [enc setTexture:tex[1] atIndex:0]; [enc setTexture:tex[2] atIndex:1]; }
    [enc dispatchThreadgroups:MTLSizeMake([g[0] unsignedIntValue], [g[1] unsignedIntValue], [g[2] unsignedIntValue]) threadsPerThreadgroup:MTLSizeMake([b[0] unsignedIntValue], [b[1] unsignedIntValue], [b[2] unsignedIntValue])]; [enc endEncoding];
    [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed");
    const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0, wall_ms = -[t0 timeIntervalSinceNow] * 1000.0; gpu_total += gpu_ms;
    const uint32_t* cov = static_cast<const uint32_t*>(bcov.contents); size_t once = 0, multi = 0; if (!final_only) for (size_t i = 0; i < pool.size() / 4; ++i) { if (cov[i] == 1) ++once; else if (cov[i] > 1) ++multi; }
    std::string tex_line;
    if (!texmode.empty()) {
      const uint32_t* tc = static_cast<const uint32_t*>(btexcov.contents); size_t t1 = 0, tm = 0; for (size_t i = 0; i < size_t(tw) * th; ++i) { if (tc[i] == 1) ++t1; else if (tc[i] > 1) ++tm; }
      std::vector<uint8_t> o(size_t(tw) * th * 8); id<MTLTexture> to = tex[texmode == "post" ? 1 : 2]; [to getBytes:o.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
      std::ofstream(out / ("tex_out_" + std::to_string(n) + ".bin"), std::ios::binary).write(reinterpret_cast<const char*>(o.data()), std::streamsize(o.size()));
      std::vector<uint8_t> back(size_t(tw) * th * 8); [tex[texmode == "post" ? 0 : 1] getBytes:back.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0];
      std::ostringstream ts; ts << ", \"texture\": {\"pixels_once\": " << t1 << ", \"pixels_multi\": " << tm << ", \"pixels\": " << size_t(tw) * th << ", \"output_sha256\": \"" << Sha(o.data(), o.size()) << "\", \"input_after_sha256\": \"" << Sha(back.data(), back.size()) << "\"}"; tex_line = ts.str();
    }
    const std::string pool_sha = final_only ? std::string("skipped") : Sha(bpool.contents, pool.size());
    const bool w_ok = Sha(bw.contents, weights.size()) == w_sha, p_ok = Sha(bprm.contents, params.size()) == p_sha;
    js << (first ? "" : ",\n") << "  {\"launch\": " << n << ", \"kernel\": \"" << kernel << "\", \"gpu_ms\": " << gpu_ms << ", \"wall_ms\": " << wall_ms << ", \"pool_post_sha256\": \"" << pool_sha << "\", \"words_written_once\": " << (final_only ? std::string("null") : std::to_string(once)) << ", \"words_written_more_than_once\": " << (final_only ? std::string("null") : std::to_string(multi))
       << ", \"weights_immutable\": " << (w_ok ? "true" : "false") << ", \"params_immutable\": " << (p_ok ? "true" : "false") << ", \"weights_sha256\": \"" << w_sha << "\"" << tex_line << "}"; first = false;
    std::cout << "launch " << n << " gpu_ms " << gpu_ms << " wall_ms " << wall_ms << " once " << once << " multi " << multi << "\n";
  }
  const double wall_total = -[t_all timeIntervalSinceNow] * 1000.0;
  std::ofstream(out / "pool_post.bin", std::ios::binary).write(static_cast<const char*>(bpool.contents), std::streamsize(pool.size()));
  const std::string final_sha = Sha(bpool.contents, pool.size());
  js << "\n ],\n \"checks\": \"" << (final_only ? "final" : "per-launch") << "\",\n \"pool_final_sha256\": \"" << final_sha << "\",\n \"gpu_ms_total\": " << gpu_total << ",\n \"wall_ms_total\": " << wall_total << "\n}\n"; std::cout << "total gpu_ms " << gpu_total << " wall_ms " << wall_total << "\n"; return 0;
} catch (const std::exception& e) { std::cerr << "ptx_chain_probe: " << e.what() << "\n"; return 2; } } }
