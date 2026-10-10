// Replay a live FG dump (640x360 colour/motion/depth per frame) through the runtime offline:
//   fg-runtime-dump-replay <libs> <model> <rom> <sigmoid> <ex2-a> <ex2-b> <dump-dir> <first> <last> [metal4]
// Reset on <first>, then generate for first+1..last; outputs written as <dump>/replay-fN-out-640x360.bin.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "fg_runtime.h"
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
static std::vector<uint8_t> ReadFile(const std::string& path) {
  std::vector<uint8_t> out; FILE* f = fopen(path.c_str(), "rb"); if (!f) return out;
  fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET); out.resize(n); fread(out.data(), 1, n, f); fclose(f); return out;
}
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 10) { fprintf(stderr, "usage\n"); return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue];
  const std::string dump = argv[7]; const int first = atoi(argv[8]), last = atoi(argv[9]);
  auto tex = [&](MTLPixelFormat f, const std::vector<uint8_t>& bytes, size_t bpp, MTLTextureUsage u) {
    MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:640 height:360 mipmapped:NO];
    d.storageMode = MTLStorageModeShared; d.usage = u; id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    if (!bytes.empty()) [t replaceRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0 withBytes:bytes.data() bytesPerRow:640 * bpp];
    return t; };
  NSFGCreateInfo c{}; c.metal_device = (__bridge void*)dev; c.metal_command_queue = (__bridge void*)q;
  c.metallib_directory = argv[1]; c.model_directory = argv[2]; c.rcp_rom_path = argv[3]; c.sigmoid_lut_path = argv[4];
  c.ex2_unit_table_path = argv[5]; c.ex2_fraction_table_path = argv[6]; c.width = 640; c.height = 360;
  c.ex2_mode = NSFG_EX2_COMPACT; c.auxiliary_mode = NSFG_AUXILIARY_LIVE_EARLY_RESOURCES; c.poison_byte = 0xa5;
  NSFGSession* s = nullptr; if (NSFGSessionCreate(&c, &s) != NSFG_STATUS_OK) { fprintf(stderr, "create failed\n"); return 1; }
  std::vector<id<MTLTexture>> colors, motions, depths;
  for (int f = first; f <= last; ++f) {
    auto col = ReadFile(dump + "/f" + std::to_string(f) + "-color-640x360-f70.bin"), mo = ReadFile(dump + "/f" + std::to_string(f) + "-motion-640x360-f65.bin"),
         de = ReadFile(dump + "/f" + std::to_string(f) + "-depth-640x360-f55.bin");
    if (col.empty() || mo.empty() || de.empty()) { fprintf(stderr, "missing frame %d\n", f); return 1; }
    if (const char* m = getenv("FG_REPLAY_COLOR")) {   // shift:<px> moves the current colour right by px (wraps)
      if (!strncmp(m, "shift:", 6) && f > first) {
        int px = atoi(m + 6); std::vector<uint8_t> s(col.size());
        for (int y = 0; y < 360; ++y) for (int x = 0; x < 640; ++x) memcpy(&s[(y * 640 + x) * 4], &col[(y * 640 + ((x - px + 640 * 4) % 640)) * 4], 4);
        col = s;
      } else if (!strcmp(m, "gray") && f > first) { for (size_t i = 0; i < col.size(); i += 4) col[i] = col[i+1] = col[i+2] = 128; }
    }
    if (const char* m = getenv("FG_REPLAY_DEPTH")) {
      float* z = reinterpret_cast<float*>(de.data()); size_t n = de.size() / 4;
      if (!strncmp(m, "const:", 6)) { float v = atof(m + 6); for (size_t i = 0; i < n; ++i) z[i] = v; }
      else if (!strcmp(m, "invert")) { for (size_t i = 0; i < n; ++i) z[i] = 1.0f - z[i]; }
      else if (!strncmp(m, "scale:", 6)) { float k = atof(m + 6); for (size_t i = 0; i < n; ++i) z[i] *= k; }
    }
    if (const char* m = getenv("FG_REPLAY_MOTION")) {
      __fp16* v = reinterpret_cast<__fp16*>(mo.data()); size_t n = mo.size() / 2;
      if (!strcmp(m, "zero")) { for (size_t i = 0; i < n; ++i) v[i] = (__fp16)0.0f; }
      else if (!strcmp(m, "negate")) { for (size_t i = 0; i < n; ++i) v[i] = (__fp16)(-(float)v[i]); }
      else if (!strncmp(m, "scale:", 6)) { float k = atof(m + 6); for (size_t i = 0; i < n; ++i) v[i] = (__fp16)((float)v[i] * k); }
    }
    colors.push_back(tex(MTLPixelFormatRGBA8Unorm, col, 4, MTLTextureUsageShaderRead));
    motions.push_back(tex(MTLPixelFormatRG16Float, mo, 4, MTLTextureUsageShaderRead));
    depths.push_back(tex(MTLPixelFormatR32Float, de, 4, MTLTextureUsageShaderRead));
  }
  NSFGFrameInfo r{}; r.current_backbuffer_rgba8 = (__bridge void*)colors[0]; r.previous_backbuffer_rgba8 = (__bridge void*)colors[0];
  r.motion_rg16f = (__bridge void*)motions[0]; r.depth_r32f = (__bridge void*)depths[0]; r.frame_id = 0; r.reset = true;
  NSFGStatus st = NSFGSessionEvaluate(s, &r, nullptr); if (st != NSFG_STATUS_OK) { fprintf(stderr, "reset: %s %s\n", NSFGStatusName(st), NSFGSessionLastError(s)); return 1; }
  for (size_t i = 1; i < colors.size(); ++i) {
    id<MTLTexture> out = tex(MTLPixelFormatRGBA8Unorm, {}, 4, MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);
    NSFGFrameInfo in{}; in.current_backbuffer_rgba8 = (__bridge void*)colors[i]; in.previous_backbuffer_rgba8 = (__bridge void*)colors[i - 1];
    if (getenv("FG_REPLAY_PREV")) {   // identical frames: copy the current colour into the previous texture
      std::vector<uint8_t> b(640 * 360 * 4);
      [colors[i] getBytes:b.data() bytesPerRow:640 * 4 fromRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0];
      [colors[i - 1] replaceRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0 withBytes:b.data() bytesPerRow:640 * 4];
    }
    in.motion_rg16f = (__bridge void*)motions[i]; in.depth_r32f = (__bridge void*)depths[i]; in.generated_target_rgba8 = (__bridge void*)out; in.frame_id = i;
    NSFGFrameResult res{}; st = NSFGSessionEvaluate(s, &in, &res);
    if (st != NSFG_STATUS_OK) { fprintf(stderr, "frame %zu: %s %s\n", i, NSFGStatusName(st), NSFGSessionLastError(s)); return 1; }
    std::vector<uint8_t> bytes(640 * 360 * 4);
    [out getBytes:bytes.data() bytesPerRow:640 * 4 fromRegion:MTLRegionMake2D(0, 0, 640, 360) mipmapLevel:0];
    std::string tag = getenv("FG_REPLAY_TAG") ? getenv("FG_REPLAY_TAG") : "replay";
    std::string path = dump + "/" + tag + "-f" + std::to_string(first + (int)i) + "-out-640x360-f70.bin";
    FILE* f = fopen(path.c_str(), "wb"); fwrite(bytes.data(), 1, bytes.size(), f); fclose(f);
    printf("wrote %s\n", path.c_str());
  }
  NSFGSessionDestroy(s); return 0;
} }
