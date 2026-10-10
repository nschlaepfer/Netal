// nr_tex_probe.mm — Metal driver for nr_tex_probe.metal: builds the micro-probe's step texture (R = x>=384, G = y>=1, B = (x+y)&1, A = 1),
// runs the queries, writes samples (u32 bits x4 per query) and a coverage check. Usage: nr_tex_probe LIB QUERIES.bin OUT_SAMPLES.bin
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc != 4 && argc != 5) { fprintf(stderr, "usage: nr_tex_probe LIB QUERIES.bin OUT.bin [INPUT.rgba16le]\n"); return 2; }
  const int W = 1920, H = 1080; std::vector<uint16_t> img(size_t(W) * H * 4);
  if (argc == 5) { std::ifstream f(argv[4], std::ios::binary); f.read(reinterpret_cast<char*>(img.data()), std::streamsize(img.size() * 2)); if (!f) { fprintf(stderr, "input image must be 1920x1080 RGBA16F\n"); return 2; } }
  else for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) { size_t i = (size_t(y) * W + x) * 4; img[i] = x >= 384 ? 0x3c00 : 0; img[i + 1] = y >= 1 ? 0x3c00 : 0; img[i + 2] = ((x + y) & 1) ? 0x3c00 : 0; img[i + 3] = 0x3c00; }
  std::ifstream qf(argv[2], std::ios::binary); std::vector<char> qb((std::istreambuf_iterator<char>(qf)), {}); const uint32_t n = uint32_t(qb.size() / 8); if (!n) { fprintf(stderr, "no queries\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil; id<MTLLibrary> lib = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:argv[1]]] error:&err]; if (!lib) { fprintf(stderr, "library: %s\n", err.localizedDescription.UTF8String); return 2; }
  id<MTLComputePipelineState> pso = [dev newComputePipelineStateWithFunction:[lib newFunctionWithName:@"nr_tex_probe_rtx"] error:&err]; if (!pso) { fprintf(stderr, "pipeline: %s\n", err.localizedDescription.UTF8String); return 2; }
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:W height:H mipmapped:NO]; d.usage = MTLTextureUsageShaderRead; d.storageMode = MTLStorageModeShared;
  id<MTLTexture> tex = [dev newTextureWithDescriptor:d]; [tex replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:img.data() bytesPerRow:W * 8];
  id<MTLBuffer> qbuf = [dev newBufferWithBytes:qb.data() length:qb.size() options:MTLResourceStorageModeShared]; id<MTLBuffer> obuf = [dev newBufferWithLength:size_t(n) * 16 options:MTLResourceStorageModeShared]; const char* pz = getenv("NR_TEX_PROBE_POISON"); const unsigned poison = pz ? unsigned(strtoul(pz, nullptr, 16) & 0xffu) : 0xa5u; memset(obuf.contents, int(poison), size_t(n) * 16);   // output poison (env NR_TEX_PROBE_POISON, hex; default a5)
  id<MTLBuffer> cbuf = [dev newBufferWithLength:size_t(n) * 4 options:MTLResourceStorageModeShared]; memset(cbuf.contents, 0, size_t(n) * 4); id<MTLBuffer> nbuf = [dev newBufferWithBytes:&n length:4 options:MTLResourceStorageModeShared];
  id<MTLCommandQueue> q = [dev newCommandQueue]; id<MTLCommandBuffer> cb = [q commandBuffer]; id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:pso]; [enc setTexture:tex atIndex:0]; [enc setBuffer:qbuf offset:0 atIndex:0]; [enc setBuffer:obuf offset:0 atIndex:1]; [enc setBuffer:cbuf offset:0 atIndex:2]; [enc setBuffer:nbuf offset:0 atIndex:3];
  [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding]; [cb commit]; [cb waitUntilCompleted];
  const uint32_t* cov = static_cast<const uint32_t*>(cbuf.contents); for (uint32_t i = 0; i < n; ++i) if (cov[i] != 1) { fprintf(stderr, "coverage not exactly one at %u\n", i); return 3; }
  std::vector<uint16_t> after(img.size()); [tex getBytes:after.data() bytesPerRow:W * 8 fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0]; if (after != img) { fprintf(stderr, "texture mutated\n"); return 3; }
  std::ofstream(argv[3], std::ios::binary).write(static_cast<const char*>(obuf.contents), std::streamsize(size_t(n) * 16)); printf("queries=%u poison=%02x pass\n", n, poison); return 0;
} }
