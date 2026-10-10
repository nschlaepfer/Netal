// What MetalFX's own temporal scaler costs, so Netal's cost can be compared against something
// measured rather than assumed.
//
// Netal replaces the game's MTLFXTemporalScaler encode with the DLSS network. Both paths save the
// same render time - the game draws the same reduced pixel count either way - so the only thing
// that differs is the price of the upscale itself. This times MetalFX at a geometry given on the
// command line, using the same clock the runtime uses for its own pass: the GPU start/end
// timestamps of a command buffer containing nothing but the encode.
//
//   metalfx-cost-bench 1536x961 2560x1600 [iterations]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

id<MTLTexture> MakeTexture(id<MTLDevice> device, NSUInteger width, NSUInteger height,
                           MTLPixelFormat format, MTLTextureUsage usage) {
  MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                                               width:width
                                                                              height:height
                                                                           mipmapped:NO];
  d.usage = usage;
  d.storageMode = MTLStorageModePrivate;
  return [device newTextureWithDescriptor:d];
}

bool ParseGeometry(const char* text, NSUInteger* width, NSUInteger* height) {
  unsigned long w = 0, h = 0;
  if (std::sscanf(text, "%lux%lu", &w, &h) != 2 || w == 0 || h == 0) return false;
  *width = w; *height = h;
  return true;
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc < 3) {
      std::fprintf(stderr, "usage: %s <renderWxH> <outputWxH> [iterations]\n", argv[0]);
      return 64;
    }
    NSUInteger input_width = 0, input_height = 0, output_width = 0, output_height = 0;
    if (!ParseGeometry(argv[1], &input_width, &input_height) ||
        !ParseGeometry(argv[2], &output_width, &output_height)) {
      std::fprintf(stderr, "geometry must look like 1536x961\n");
      return 64;
    }
    const int iterations = argc > 3 ? std::atoi(argv[3]) : 240;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    id<MTLCommandQueue> queue = [device newCommandQueue];

    // The formats Lies of P actually supplies, so the measurement is of the work the runtime
    // replaces rather than of a convenient synthetic case.
    MTLFXTemporalScalerDescriptor* descriptor = [MTLFXTemporalScalerDescriptor new];
    descriptor.inputWidth = input_width;
    descriptor.inputHeight = input_height;
    descriptor.outputWidth = output_width;
    descriptor.outputHeight = output_height;
    descriptor.colorTextureFormat = MTLPixelFormatRGBA16Float;
    descriptor.depthTextureFormat = MTLPixelFormatDepth32Float_Stencil8;
    descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
    descriptor.outputTextureFormat = MTLPixelFormatRG11B10Float;
    descriptor.autoExposureEnabled = YES;

    id<MTLFXTemporalScaler> scaler = [descriptor newTemporalScalerWithDevice:device];
    if (scaler == nil) {
      std::fprintf(stderr, "MetalFX refused this descriptor\n");
      return 1;
    }

    scaler.colorTexture = MakeTexture(device, input_width, input_height, MTLPixelFormatRGBA16Float,
                                      MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    scaler.depthTexture = MakeTexture(device, input_width, input_height,
                                      MTLPixelFormatDepth32Float_Stencil8,
                                      MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    scaler.motionTexture = MakeTexture(device, input_width, input_height, MTLPixelFormatRG16Float,
                                       MTLTextureUsageShaderRead | MTLTextureUsageRenderTarget);
    scaler.outputTexture = MakeTexture(device, output_width, output_height,
                                       MTLPixelFormatRG11B10Float,
                                       MTLTextureUsageShaderWrite | MTLTextureUsageRenderTarget);
    scaler.inputContentWidth = input_width;
    scaler.inputContentHeight = input_height;
    scaler.motionVectorScaleX = 1.0f;
    scaler.motionVectorScaleY = 1.0f;

    std::vector<double> samples;
    samples.reserve(iterations);
    for (int i = 0; i < iterations; ++i) {
      // A history-dependent filter: the first frame resets, the rest accumulate, which is the
      // steady state a running game is in.
      scaler.reset = (i == 0);
      scaler.jitterOffsetX = ((i % 8) - 4) * 0.0625f;
      scaler.jitterOffsetY = ((i % 5) - 2) * 0.0625f;
      id<MTLCommandBuffer> command_buffer = [queue commandBuffer];
      [scaler encodeToCommandBuffer:command_buffer];
      [command_buffer commit];
      [command_buffer waitUntilCompleted];
      const double ms = (command_buffer.GPUEndTime - command_buffer.GPUStartTime) * 1000.0;
      if (i >= 16 && ms > 0.0) samples.push_back(ms);  // discard warm-up
    }
    if (samples.empty()) {
      std::fprintf(stderr, "no timing samples\n");
      return 1;
    }
    std::sort(samples.begin(), samples.end());
    double sum = 0.0;
    for (double s : samples) sum += s;
    std::printf("render=%lux%lu output=%lux%lu samples=%zu mean_ms=%.3f median_ms=%.3f min_ms=%.3f\n",
                (unsigned long)input_width, (unsigned long)input_height,
                (unsigned long)output_width, (unsigned long)output_height, samples.size(),
                sum / samples.size(), samples[samples.size() / 2], samples.front());
    return 0;
  }
}
