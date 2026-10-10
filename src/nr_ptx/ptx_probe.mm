// Generic launch probe for PTX-transliterated NR kernels: loads the recorded pre-launch pool image, the weight blob and
// the recorded parameter bytes (with the emulator's fake bases), runs the Metal kernel over the recorded grid, and writes
// the post-launch pool image for comparison with the oracle. Inputs are hashed before and after.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
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
int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 10) throw std::runtime_error("usage: ptx_probe LIB KERNEL POOL_PRE.bin WEIGHTS.bin PARAMS.bin TABLES_DIR GX,GY,GZ BX,BY,BZ OUTPUT_DIR [TEXIN.rgba16f|- [W,H [TEX_POISON_HEX]]]");
  fs::path lib = argv[1]; std::string kernel = argv[2]; fs::path poolp = argv[3], wp = argv[4], pp = argv[5], tables = argv[6], out = argv[9];
  auto triple = [](const std::string& s) { std::array<unsigned, 3> r{}; sscanf(s.c_str(), "%u,%u,%u", &r[0], &r[1], &r[2]); return r; };
  const auto grid = triple(argv[7]), block = triple(argv[8]);
  const std::string texin = argc > 10 ? argv[10] : "-"; unsigned tw = 1920, th = 1080; if (argc > 11) sscanf(argv[11], "%u,%u", &tw, &th);
  const unsigned tex_poison = argc > 12 ? unsigned(std::stoul(argv[12], nullptr, 16)) & 0xffu : 0xa5u;   // the output texture is poisoned with this byte before the launch
  if (fs::exists(out)) throw std::runtime_error("preserve previous run: " + out.string());
  auto pool = Read(poolp), weights = Read(wp), params = Read(pp), rsq = Read(tables / "rtx-rsqrt-half.bin"), rcp = Read(tables / "rtx-rcp-half.bin");
  const std::string w_sha = Sha(weights.data(), weights.size()), p_sha = Sha(params.data(), params.size());
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> library = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib.c_str()]] error:&err]; if (!library) throw std::runtime_error(err.localizedDescription.UTF8String);
  id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:kernel.c_str()]] error:&err];
  if (!ps) throw std::runtime_error(kernel + ": " + std::string(err.localizedDescription.UTF8String));
  auto mk = [&](const std::vector<uint8_t>& v, size_t extra) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + extra options:MTLResourceStorageModeShared]; if (!b) throw std::runtime_error("alloc"); std::memset(b.contents, 0, v.size() + extra); std::memcpy(b.contents, v.data(), v.size()); return b; };
  id<MTLBuffer> bpool = mk(pool, 4096), bw = mk(weights, 4096), bprm = mk(params, 64), brsq = mk(rsq, 0), brcp = mk(rcp, 0);
  std::vector<uint8_t> dbgz(64 * 32 * 4, 0); id<MTLBuffer> bdbg = mk(dbgz, 0);
  auto opt = [&](const char* name) { fs::path f = tables / name; std::vector<uint8_t> v = fs::exists(f) ? Read(f) : std::vector<uint8_t>(16, 0); return mk(v, 0); };   // MUFU correction data (post block); dummies when absent
  id<MTLBuffer> brom = opt("sm120-rcp-correction-rom.bin"), bex2u = opt("unit-corrections.bin"), bex2f = opt("fraction-corrections.bin");
  id<MTLBuffer> bcov = [dev newBufferWithLength:pool.size() + 4096 options:MTLResourceStorageModeShared]; std::memset(bcov.contents, 0, pool.size() + 4096);
  id<MTLTexture> tin = nil, tout = nil; std::string tex_sha, tex_in_after_sha;
  if (texin != "-") {          // post block: sampled input color texture and the written output texture (RGBA16F, per the capture's CUarray formats)
    auto tex = Read(texin); if (tex.size() != size_t(tw) * th * 8) throw std::runtime_error("texture byte size mismatch"); tex_sha = Sha(tex.data(), tex.size());
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
    tin = [dev newTextureWithDescriptor:d]; [tin replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:tex.data() bytesPerRow:tw * 8];
    MTLTextureDescriptor* d2 = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; d2.usage = MTLTextureUsageShaderWrite | MTLTextureUsageShaderRead; d2.storageMode = MTLStorageModeShared;
    tout = [dev newTextureWithDescriptor:d2]; std::vector<uint8_t> poison(size_t(tw) * th * 8, uint8_t(tex_poison)); [tout replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:poison.data() bytesPerRow:tw * 8];
  }
  id<MTLBuffer> btexcov = [dev newBufferWithLength:size_t(tw) * th * 4 + 16 options:MTLResourceStorageModeShared]; std::memset(btexcov.contents, 0, size_t(tw) * th * 4 + 16);   // per-pixel write counters
  id<MTLCommandQueue> queue = [dev newCommandQueue]; id<MTLCommandBuffer> cmd = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
  [enc setComputePipelineState:ps]; [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:bw offset:0 atIndex:1]; [enc setBuffer:bprm offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6]; [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10];
  if (tin) { [enc setTexture:tin atIndex:0]; [enc setTexture:tout atIndex:1]; }
  [enc dispatchThreadgroups:MTLSizeMake(grid[0], grid[1], grid[2]) threadsPerThreadgroup:MTLSizeMake(block[0], block[1], block[2])]; [enc endEncoding];
  const auto t0 = [NSDate date]; [cmd commit]; [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed");
  const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0, wall_ms = -[t0 timeIntervalSinceNow] * 1000.0;
  fs::create_directories(out);
  std::ofstream(out / "pool_post.bin", std::ios::binary).write(static_cast<const char*>(bpool.contents), std::streamsize(pool.size()));
  std::ofstream(out / "dbg.bin", std::ios::binary).write(static_cast<const char*>(bdbg.contents), std::streamsize(dbgz.size()));
  std::ofstream(out / "cov.bin", std::ios::binary).write(static_cast<const char*>(bcov.contents), std::streamsize(pool.size()));
  if (tout) { std::vector<uint8_t> o(size_t(tw) * th * 8); [tout getBytes:o.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; std::ofstream(out / "tex_out.bin", std::ios::binary).write(reinterpret_cast<const char*>(o.data()), std::streamsize(o.size()));
    std::ofstream(out / "tex_coverage.bin", std::ios::binary).write(static_cast<const char*>(btexcov.contents), std::streamsize(size_t(tw) * th * 4));
    std::vector<uint8_t> back(size_t(tw) * th * 8); [tin getBytes:back.data() bytesPerRow:tw * 8 fromRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0]; tex_in_after_sha = Sha(back.data(), back.size()); }
  const bool w_ok = Sha(bw.contents, weights.size()) == w_sha, p_ok = Sha(bprm.contents, params.size()) == p_sha;
  std::ofstream(out / "run.json") << "{\n \"schema\": \"neuralstudio.nr.ptx-probe.v1\",\n \"kernel\": \"" << kernel << "\",\n \"pid\": " << getpid() << ",\n \"gpu_ms\": " << gpu_ms << ",\n \"wall_ms\": " << wall_ms
    << ",\n \"grid\": [" << grid[0] << "," << grid[1] << "," << grid[2] << "],\n \"block\": [" << block[0] << "," << block[1] << "," << block[2] << "],\n \"weights_sha256\": \"" << w_sha << "\",\n \"params_sha256\": \"" << p_sha << "\",\n \"texture_in_sha256\": \"" << tex_sha << "\",\n \"texture_in_after_sha256\": \"" << tex_in_after_sha << "\",\n \"texture_poison\": " << tex_poison << ",\n \"texture_size\": [" << tw << "," << th << "],\n \"weights_immutable\": " << (w_ok ? "true" : "false") << ",\n \"params_immutable\": " << (p_ok ? "true" : "false") << "\n}\n";
  std::cout << "gpu_ms " << gpu_ms << " wall_ms " << wall_ms << " weights_immutable " << w_ok << "\n"; return (w_ok && p_ok) ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "ptx_probe: " << e.what() << "\n"; return 2; } } }
