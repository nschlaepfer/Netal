// Offline gate for NeuralStudioDLSSFGSession: interpolator-shaped textures at RE9's live geometry
// (colour BGR10A2 1728x1117 with a 1728x972 content rect, D32S8 depth and RG16F motion at 1002x564),
// a reset then N frames through MTL4 command buffers; the output must be written, finite, and close
// to the mean of the two real frames for a slowly moving gradient.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "dlss_fg_session.h"
#include <chrono>
#include <cstdio>
#include <vector>
#include <cmath>
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 3) { fprintf(stderr, "usage: probe <bundle> <frames>\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  id<MTL4CommandQueue> q = [dev newMTL4CommandQueue];
  id<MTL4CommandAllocator> alloc = [dev newCommandAllocator];
  id<MTLSharedEvent> ev = [dev newSharedEvent];
  NSError* err = nil;
  auto t0 = std::chrono::steady_clock::now();
  NeuralStudioDLSSFGSession* s = [[NeuralStudioDLSSFGSession alloc] initWithDevice:dev runtimeBundleURL:[NSURL fileURLWithPath:@(argv[1])] error:&err];
  if (!s) { fprintf(stderr, "init failed: %s\n", err.localizedDescription.UTF8String); return 1; }
  printf("arm_ms=%.0f\n", std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
  auto tex = [&](MTLPixelFormat f, NSUInteger w, NSUInteger h, MTLTextureUsage u) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:w height:h mipmapped:NO];
    d.usage = u; d.storageMode = MTLStorageModeShared; return [dev newTextureWithDescriptor:d]; };
  auto envu = [](const char* k, NSUInteger d) { const char* v = getenv(k); return v ? (NSUInteger)atoi(v) : d; };
  const NSUInteger W = envu("PROBE_W", 1728), H = envu("PROBE_H", 1117), CW = envu("PROBE_CW", 1728), CH = envu("PROBE_CH", 972), RW = envu("PROBE_RW", 1002), RH = envu("PROBE_RH", 564);
  const char* dump = getenv("PROBE_DUMP");   // directory: write each output frame's content rect as raw BGR10A2
  const int frames = atoi(argv[2]);
  std::vector<id<MTLTexture>> colors;
  for (int f = 0; f < frames; ++f) {
    id<MTLTexture> c = tex(MTLPixelFormatBGR10A2Unorm, W, H, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    std::vector<uint32_t> px(W * H);
    for (NSUInteger y = 0; y < H; ++y) for (NSUInteger x = 0; x < W; ++x) {
      // gradient that slides 4 px/frame, plus a checker
      uint32_t r = (uint32_t)(((x + 4 * f) % CW) * 1023 / CW), g = (uint32_t)(y * 1023 / H), b = ((((x + 4 * f) / 64) + (y / 64)) & 1) ? 900 : 200;
      px[y * W + x] = (3u << 30) | (r << 20) | (g << 10) | b;   // BGR10A2: b in low bits? layout: B[0:9] G[10:19] R[20:29] A[30:31]
    }
    [c replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:px.data() bytesPerRow:W * 4];
    colors.push_back(c);
  }
  // motion: constant -4 px per frame in x, expressed as normalised (msx = RW): value = -4/RW... the game gives UV units.
  id<MTLTexture> motion = tex(MTLPixelFormatRG16Float, RW, RH, MTLTextureUsageShaderRead);
  std::vector<uint16_t> mv(RW * RH * 2);
  auto h = [](float v) { __fp16 x = (__fp16)v; uint16_t b; memcpy(&b, &x, 2); return b; };
  for (size_t i = 0; i < RW * RH; ++i) { mv[2 * i] = h(-4.0f * 1002.0f / 1728.0f / RW); mv[2 * i + 1] = h(0.0f); }
  [motion replaceRegion:MTLRegionMake2D(0, 0, RW, RH) mipmapLevel:0 withBytes:mv.data() bytesPerRow:RW * 4];
  id<MTLTexture> depth = tex(MTLPixelFormatDepth32Float, RW, RH, MTLTextureUsageShaderRead);
  std::vector<float> dz(RW * RH, 0.5f);
  [depth replaceRegion:MTLRegionMake2D(0, 0, RW, RH) mipmapLevel:0 withBytes:dz.data() bytesPerRow:RW * 4];
  id<MTLTexture> out = tex(MTLPixelFormatBGR10A2Unorm, W, H, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
  // The game's queue holds its textures resident; here the probe must do it itself.
  MTLResidencySetDescriptor* rd = [[MTLResidencySetDescriptor alloc] init];
  id<MTLResidencySet> rs = [dev newResidencySetWithDescriptor:rd error:&err];
  for (id<MTLTexture> c : colors) [rs addAllocation:c];
  [rs addAllocation:motion]; [rs addAllocation:depth]; [rs addAllocation:out];
  [rs commit];
  [q addResidencySet:rs];
  uint64_t sig = 0;
  MTL4CommitOptions* opts = [[MTL4CommitOptions alloc] init];
  [opts addFeedbackHandler:^(id<MTL4CommitFeedback> fb) { if (fb.error) fprintf(stderr, "gpu_error=%s\n", fb.error.localizedDescription.UTF8String); }];
  for (int f = 0; f < frames; ++f) {
    id<MTL4CommandBuffer> cb = [dev newCommandBuffer];
    [cb beginCommandBufferWithAllocator:alloc];
    auto e0 = std::chrono::steady_clock::now();
    NSError* e = nil;
    BOOL ok = [s encodeToMTL4CommandBuffer:cb colorTexture:colors[f] depthTexture:depth motionTexture:motion outputTexture:out
                              contentWidth:CW contentHeight:CH renderWidth:RW renderHeight:RH motionVectorScaleX:(float)RW motionVectorScaleY:(float)RH
                                     fence:nil reset:(f == 0) error:&e];
    double enc = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
    [cb endCommandBuffer];
    if (!ok) { fprintf(stderr, "frame %d not served: %s\n", f, e.localizedDescription.UTF8String); return 3; }
    id<MTL4CommandBuffer> l[1] = {cb};
    [q commit:l count:1 options:opts];
    [q signalEvent:ev value:++sig];
    if (![ev waitUntilSignaledValue:sig timeoutMS:20000]) { fprintf(stderr, "timeout\n"); return 4; }
    double total = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - e0).count();
    [alloc reset];
    // Check: sample the output centre row against the mean of frames f-1 and f (red channel gradient).
    std::vector<uint32_t> row(W);
    [out getBytes:row.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, CH / 2, W, 1) mipmapLevel:0];
    std::vector<uint32_t> rowA(W), rowB(W);
    [colors[f] getBytes:rowB.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, CH / 2, W, 1) mipmapLevel:0];
    [colors[f > 0 ? f - 1 : 0] getBytes:rowA.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, CH / 2, W, 1) mipmapLevel:0];
    double err_mean = 0, err_cur = 0; int n = 0;
    for (NSUInteger x = 100; x < CW - 100; x += 7) {
      double o = (row[x] >> 20) & 1023, a = (rowA[x] >> 20) & 1023, b = (rowB[x] >> 20) & 1023;
      err_mean += fabs(o - 0.5 * (a + b)); err_cur += fabs(o - b); ++n;
    }
    std::vector<uint32_t> below(W);
    [out getBytes:below.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, CH + 10, W, 1) mipmapLevel:0];
    [colors[f] getBytes:rowB.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, CH + 10, W, 1) mipmapLevel:0];
    if (dump) { std::vector<uint32_t> px(W * CH); [out getBytes:px.data() bytesPerRow:W * 4 fromRegion:MTLRegionMake2D(0, 0, W, CH) mipmapLevel:0];
      char path[512]; snprintf(path, sizeof path, "%s/out-f%d-%lux%lu.bin", dump, f, (unsigned long)W, (unsigned long)CH); FILE* fp = fopen(path, "wb"); fwrite(px.data(), 4, px.size(), fp); fclose(fp); }
    printf("frame=%d reset=%d encode_ms=%.2f total_ms=%.2f red_err_vs_mean=%.1f red_err_vs_current=%.1f outside_content_passthrough=%s dispatches=%lu fullres=%llu\n",
           f, f == 0, enc, total, err_mean / n, err_cur / n, below[W / 2] == rowB[W / 2] ? "yes" : "NO", (unsigned long)s.runtimeDispatchesLastFrame, (unsigned long long)s.fullResolutionFrames);
  }
  return 0;
} }
