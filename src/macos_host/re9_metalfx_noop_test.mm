#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>

#include <cstdint>
#include <cstring>
#include <iostream>

namespace {

id<MTLTexture> Texture(id<MTLDevice> device, MTLPixelFormat format,
                       NSUInteger width, NSUInteger height,
                       MTLStorageMode storage, MTLTextureUsage usage) {
  MTLTextureDescriptor* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = storage;
  descriptor.usage = usage;
  return [device newTextureWithDescriptor:descriptor];
}

int Run() {
  const char* fixture_path = std::getenv("NEURALSTUDIO_RE9_TEST_FIXTURE_ROOT");
  const bool exact = fixture_path != nullptr && *fixture_path != '\0';
  const NSUInteger input_width = exact ? 1280 : 640;
  const NSUInteger input_height = exact ? 720 : 360;
  const NSUInteger output_width = exact ? 1920 : 1280;
  const NSUInteger output_height = exact ? 1080 : 720;
  id<MTLDevice> device = MTLCreateSystemDefaultDevice();
  if (device == nil) return 2;
  MTLFXTemporalScalerDescriptor* descriptor =
      [[MTLFXTemporalScalerDescriptor alloc] init];
  descriptor.colorTextureFormat = MTLPixelFormatRGBA16Float;
  descriptor.depthTextureFormat = MTLPixelFormatR32Float;
  descriptor.motionTextureFormat = MTLPixelFormatRG16Float;
  descriptor.outputTextureFormat = MTLPixelFormatRGBA16Float;
  descriptor.inputWidth = input_width;
  descriptor.inputHeight = input_height;
  descriptor.outputWidth = output_width;
  descriptor.outputHeight = output_height;
  descriptor.autoExposureEnabled = !exact;
  descriptor.requiresSynchronousInitialization = YES;
  id<MTLFXTemporalScaler> scaler =
      [descriptor newTemporalScalerWithDevice:device];
  if (scaler == nil) return 3;
  id<MTLTexture> color = Texture(
      device, MTLPixelFormatRGBA16Float, input_width, input_height, MTLStorageModeShared,
      scaler.colorTextureUsage | MTLTextureUsageShaderRead);
  id<MTLTexture> depth = Texture(
      device, MTLPixelFormatR32Float, input_width, input_height, MTLStorageModeShared,
      scaler.depthTextureUsage | MTLTextureUsageShaderRead);
  id<MTLTexture> motion = Texture(
      device, MTLPixelFormatRG16Float, input_width, input_height, MTLStorageModeShared,
      scaler.motionTextureUsage | MTLTextureUsageShaderRead);
  id<MTLTexture> output = Texture(
      device, MTLPixelFormatRGBA16Float, output_width, output_height,
      exact ? MTLStorageModeShared : MTLStorageModePrivate,
      scaler.outputTextureUsage | MTLTextureUsageShaderRead);
  if (color == nil || depth == nil || motion == nil || output == nil) return 4;
  scaler.colorTexture = color;
  scaler.depthTexture = depth;
  scaler.motionTexture = motion;
  scaler.outputTexture = output;
  scaler.inputContentWidth = input_width;
  scaler.inputContentHeight = input_height;
  scaler.preExposure = 1.0f;
  scaler.motionVectorScaleX = 1.0f;
  scaler.motionVectorScaleY = 1.0f;
  scaler.jitterOffsetX = 0.0f;
  scaler.jitterOffsetY = 0.0f;
  scaler.reset = YES;
  scaler.depthReversed = NO;
  if (exact) {
    NSString* root = [NSString stringWithUTF8String:fixture_path];
    NSString* input = [root stringByAppendingPathComponent:@"hiluma-input-fixture"];
    NSData* color_data = [NSData dataWithContentsOfFile:
        [input stringByAppendingPathComponent:@"array-0007.bin"]];
    NSData* motion_data = [NSData dataWithContentsOfFile:
        [input stringByAppendingPathComponent:@"array-0010.bin"]];
    NSData* depth_data = [NSData dataWithContentsOfFile:
        [input stringByAppendingPathComponent:@"array-0009.bin"]];
    NSData* exposure_data = [NSData dataWithContentsOfFile:
        [input stringByAppendingPathComponent:@"array-0011.bin"]];
    id<MTLTexture> exposure = Texture(
        device, MTLPixelFormatR32Float, 1, 1, MTLStorageModeShared,
        MTLTextureUsageShaderRead);
    if (color_data.length != 1280 * 720 * 8 ||
        motion_data.length != 1280 * 720 * 4 ||
        depth_data.length != 1280 * 720 * 4 || exposure_data.length != 4 ||
        exposure == nil) return 6;
    [color replaceRegion:MTLRegionMake2D(0, 0, 1280, 720) mipmapLevel:0
                withBytes:color_data.bytes bytesPerRow:1280 * 8];
    [motion replaceRegion:MTLRegionMake2D(0, 0, 1280, 720) mipmapLevel:0
                 withBytes:motion_data.bytes bytesPerRow:1280 * 4];
    [depth replaceRegion:MTLRegionMake2D(0, 0, 1280, 720) mipmapLevel:0
                withBytes:depth_data.bytes bytesPerRow:1280 * 4];
    [exposure replaceRegion:MTLRegionMake2D(0, 0, 1, 1) mipmapLevel:0
                   withBytes:exposure_data.bytes bytesPerRow:4];
    scaler.exposureTexture = exposure;
  }
  id<MTLCommandQueue> queue = [device newCommandQueue];
  id<MTLCommandBuffer> command = [queue commandBuffer];
  [scaler encodeToCommandBuffer:command];
  [command commit];
  [command waitUntilCompleted];
  if (command.status != MTLCommandBufferStatusCompleted) return 5;
  if (exact) {
    NSString* root = [NSString stringWithUTF8String:fixture_path];
    NSData* expected = [NSData dataWithContentsOfFile:
        [[root stringByAppendingPathComponent:@"hiluma-output-fixture"]
            stringByAppendingPathComponent:@"array-0008-expected.bin"]];
    NSMutableData* actual = [NSMutableData dataWithLength:1920 * 1080 * 8];
    [output getBytes:actual.mutableBytes bytesPerRow:1920 * 8
          fromRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0];
    if (expected.length != actual.length || ![actual isEqualToData:expected]) return 7;
    std::cout << "exact_sr_fixture_output=true\n";
    id<MTLCommandBuffer> second = [queue commandBuffer];
    [scaler encodeToCommandBuffer:second];
    [second commit];
    [second waitUntilCompleted];
    if (second.status != MTLCommandBufferStatusCompleted) return 8;
    std::cout << "post_arm_frame_completed=true\n";
  }
  std::cout << "probe=re9_metalfx_noop_bridge\n";
  std::cout << "device=" << device.name.UTF8String << "\n";
  std::cout << "result=pass\n";
  return 0;
}

}  // namespace

int main() {
  @autoreleasepool {
    return Run();
  }
}
