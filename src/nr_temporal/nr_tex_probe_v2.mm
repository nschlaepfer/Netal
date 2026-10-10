// nr_tex_probe_v2.mm — driver v2 for nr_tex_probe.metal (Codex's 00:26 evidence gap; the KERNEL and the candidate metallib are unchanged): explicit
// command-buffer status rejection, query buffer post-check (bytes unchanged), 256-byte redzones (0xC5) before and after the query, output and coverage
// buffers verified intact, output poison selectable, and a per-process JSON receipt (OS pid, poison, sha256 of library/queries/input image/samples, guard
// verdicts). Usage: nr_tex_probe_v2 LIB QUERIES.bin OUT_SAMPLES.bin INPUT.rgba16le RECEIPT.json   (env NR_TEX_PROBE_POISON=a5|5a, default a5)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <CommonCrypto/CommonDigest.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <unistd.h>
static std::string Sha(const void* p, size_t n) { unsigned char d[32]; CC_SHA256(p, (CC_LONG)n, d); char h[65]; for (int i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", d[i]); return std::string(h, 64); }
static std::vector<char> Read(const char* p) { std::ifstream f(p, std::ios::binary); return std::vector<char>((std::istreambuf_iterator<char>(f)), {}); }
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc != 6) { fprintf(stderr, "usage: nr_tex_probe_v2 LIB QUERIES.bin OUT.bin INPUT.rgba16le RECEIPT.json\n"); return 2; }
  const int W = 1920, H = 1080; const size_t RZ = 256; const unsigned char RZB = 0xC5;
  auto img = Read(argv[4]); if (img.size() != size_t(W) * H * 8) { fprintf(stderr, "input image must be 1920x1080 RGBA16F\n"); return 2; }
  auto qb = Read(argv[2]); const uint32_t n = uint32_t(qb.size() / 8); if (!n || qb.size() % 8) { fprintf(stderr, "queries malformed\n"); return 2; }
  auto lb = Read(argv[1]); const char* pz = getenv("NR_TEX_PROBE_POISON"); const unsigned poison = pz ? unsigned(strtoul(pz, nullptr, 16) & 0xffu) : 0xa5u;
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil; id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err]; if (!lib) { fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String); return 2; }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_tex_probe_rtx"] error:&err]; if (!pso) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 2; }
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:W height:H mipmapped:NO]; d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:d]; [tex replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:img.data() bytesPerRow:W * 8];
  // every buffer carries a redzone before and after its payload; the kernel is bound at offset RZ
  auto mk = [&](size_t payload, int fill) { id<MTLBuffer> b = [dev newBufferWithLength:payload + 2 * RZ options:MTLResourceStorageModeShared]; unsigned char* p = static_cast<unsigned char*>(b.contents); memset(p, RZB, RZ); memset(p + RZ, fill, payload); memset(p + RZ + payload, RZB, RZ); return b; };
  id<MTLBuffer> qbuf = mk(qb.size(), 0); memcpy(static_cast<char*>(qbuf.contents) + RZ, qb.data(), qb.size()); id<MTLBuffer> obuf = mk(size_t(n) * 16, int(poison)); id<MTLBuffer> cbuf = mk(size_t(n) * 4, 0); id<MTLBuffer> nbuf = [dev newBufferWithBytes:&n length:4 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso]; [enc setTexture:tex atIndex:0]; [enc setBuffer:qbuf offset:RZ atIndex:0]; [enc setBuffer:obuf offset:RZ atIndex:1]; [enc setBuffer:cbuf offset:RZ atIndex:2]; [enc setBuffer:nbuf offset:0 atIndex:3];
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  const bool cb_ok = cb.status == MTLCommandBufferStatusCompleted; std::string cb_err = cb.error ? cb.error.localizedDescription.UTF8String : "";
  auto rz_ok = [&](id<MTLBuffer> b, size_t payload) { const unsigned char* p = static_cast<const unsigned char*>(b.contents); for (size_t i = 0; i < RZ; ++i) if (p[i] != RZB || p[RZ + payload + i] != RZB) return false; return true; };
  const bool rz_q = rz_ok(qbuf, qb.size()), rz_o = rz_ok(obuf, size_t(n) * 16), rz_c = rz_ok(cbuf, size_t(n) * 4);
  const bool q_same = memcmp(static_cast<const char*>(qbuf.contents) + RZ, qb.data(), qb.size()) == 0;
  const uint32_t* cov = reinterpret_cast<const uint32_t*>(static_cast<const char*>(cbuf.contents) + RZ); size_t cov_bad = 0; for (uint32_t i = 0; i < n; ++i) if (cov[i] != 1) ++cov_bad;
  std::vector<char> after(img.size()); [tex getBytes:after.data() bytesPerRow:W * 8 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0]; const bool tex_same = after == img;
  const char* samples = static_cast<const char*>(obuf.contents) + RZ; size_t poisoned_words = 0; { const uint32_t pw = poison * 0x01010101u; const uint32_t* s = reinterpret_cast<const uint32_t*>(samples); for (size_t i = 0; i < size_t(n) * 4; ++i) if (s[i] == pw) ++poisoned_words; }
  const bool pass = cb_ok && rz_q && rz_o && rz_c && q_same && cov_bad == 0 && tex_same;
  if (pass) std::ofstream(argv[3], std::ios::binary).write(samples, std::streamsize(size_t(n) * 16));   // samples are written ONLY when every guard passed
  std::ofstream r(argv[5]); r << "{\"schema\": \"neuralstudio.nr.tex-probe-process.v2\", \"pid\": " << getpid() << ", \"poison\": \"" << std::string(1, "0123456789abcdef"[poison >> 4]) << "0123456789abcdef"[poison & 15] << "\", \"queries\": " << n
    << ", \"library\": \"" << argv[1] << "\", \"library_sha256\": \"" << Sha(lb.data(), lb.size()) << "\", \"queries_sha256\": \"" << Sha(qb.data(), qb.size()) << "\", \"input_sha256\": \"" << Sha(img.data(), img.size()) << "\", \"samples_sha256\": \"" << (pass ? Sha(samples, size_t(n) * 16) : std::string("not-written")) << "\""
    << ", \"command_buffer_completed\": " << (cb_ok ? "true" : "false") << ", \"command_buffer_error\": \"" << cb_err << "\", \"redzones_intact\": {\"queries\": " << (rz_q ? "true" : "false") << ", \"samples\": " << (rz_o ? "true" : "false") << ", \"coverage\": " << (rz_c ? "true" : "false") << "}"
    << ", \"queries_unchanged\": " << (q_same ? "true" : "false") << ", \"coverage_not_exactly_once\": " << cov_bad << ", \"input_texture_unchanged\": " << (tex_same ? "true" : "false") << ", \"sample_words_still_poison\": " << poisoned_words << ", \"pass\": " << (pass ? "true" : "false") << "}\n";
  printf("queries=%u poison=%02x cb=%s redzones=%s%s%s queries_unchanged=%s cov_bad=%zu tex_unchanged=%s %s\n", n, poison, cb_ok ? "completed" : "FAILED", rz_q ? "Q" : "q", rz_o ? "O" : "o", rz_c ? "C" : "c", q_same ? "yes" : "NO", cov_bad, tex_same ? "yes" : "NO", pass ? "pass" : "FAIL"); return pass ? 0 : 3;
} }
