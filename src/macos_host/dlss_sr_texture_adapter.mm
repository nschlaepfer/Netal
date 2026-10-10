#import "dlss_sr_texture_adapter.h"

#include <cmath>
#include <cstdlib>

#include <array>
#include <cstdint>
#include <cstring>

NSErrorDomain const NeuralStudioDLSSTextureAdapterErrorDomain =
    @"NeuralStudioDLSSTextureAdapterError";

namespace {

// Geometry comes from the session (resolution profile + render size); the
// macros below keep the method bodies readable.
#define kInputWidth (_session.inputWidth)
#define kInputHeight (_session.inputHeight)
#define kOutputWidth (_session.outputWidth)
#define kOutputHeight (_session.outputHeight)
#define kColorRowBytes (kInputWidth * 8)
#define kMotionRowBytes (kInputWidth * 4)
#define kDepthRowBytes (kInputWidth * 4)
#define kOutputRowBytes (kOutputWidth * 8)
constexpr NSUInteger kExposureRowBytes = 256;

void SetError(NSError** error, NSInteger code, NSString* description) {
  if (error == nullptr) return;
  *error = [NSError errorWithDomain:NeuralStudioDLSSTextureAdapterErrorDomain
                               code:code
                           userInfo:@{NSLocalizedDescriptionKey : description}];
}

// Everything about a texture except its size. Kept separate because the size rule differs between
// a resource that must be exactly one shape (the 1x1 exposure) and one that only has to be big
// enough (every image in the frame).
BOOL TextureShapeMatches(id<MTLTexture> texture, MTLPixelFormat format) {
  return texture != nil && texture.textureType == MTLTextureType2D && texture.depth == 1 &&
         texture.arrayLength == 1 && texture.sampleCount == 1 &&
         texture.pixelFormat == format && texture.storageMode != MTLStorageModeMemoryless;
}

BOOL TextureMatches(id<MTLTexture> texture, NSUInteger width, NSUInteger height,
                    MTLPixelFormat format) {
  return TextureShapeMatches(texture, format) && texture.width == width && texture.height == height;
}

/// True when a texture is large enough to carry a region, whatever its allocated size.
///
/// This is MetalFX's own rule, not a relaxation of it: MTLFXTemporalScaler requires the colour,
/// depth and motion textures to be *at least* inputContentWidth x inputContentHeight, and the
/// output texture at least outputWidth x outputHeight. The separate inputContentWidth/Height
/// properties exist precisely because engines allocate render targets larger than the region they
/// render into - Lies of P hands over a 2076x1296 colour texture for a 2074x1296 content region.
///
/// Demanding an exact match rejected every frame of such a game. RE9 never showed it because its
/// textures happen to be exactly content- or output-sized, which the >= rule still accepts.
///
/// A larger texture is safe because it is only ever read or written within the region: every
/// import kernel bounds its threads by content_size and every export kernel by output_size, both
/// taking the top-left rectangle - the same convention MetalFX uses.
BOOL TextureAtLeast(id<MTLTexture> texture, NSUInteger width, NSUInteger height,
                    MTLPixelFormat format) {
  return TextureShapeMatches(texture, format) && texture.width >= width && texture.height >= height;
}

}  // namespace

@implementation NeuralStudioDLSSTextureAdapter {
  __strong id<MTLDevice> _device;
  __strong NeuralStudioDLSSSRSession* _session;
  __strong id<MTLBuffer> _color;
  __strong id<MTLBuffer> _motion;
  __strong id<MTLBuffer> _depth;
  __strong id<MTLBuffer> _exposure;
  __strong id<MTLBuffer> _output;
  // Metal 4: residency for the staging buffers, per-frame ring of argument
  // tables (import color / import exposure / export), constants for the kernels'
  // former setBytes arguments.
  __strong id<MTLResidencySet> _metal4_residency;
  __strong id<MTLBuffer> _metal4_constants;
  NSMutableArray<id<MTL4ArgumentTable>>* _metal4_tables;
  NSUInteger _metal4_slot;
  __strong id<MTLComputePipelineState> _import_color_pipeline;
  // Import/export variants for games whose MetalFX textures are not RGB9E5, keyed by pixel format.
  // Empty for a runtime bundle predating netal_texture_import_ext.metallib, in which case such a
  // game is rejected by the contract check exactly as before.
  NSMutableDictionary<NSNumber*, id<MTLComputePipelineState>>* _import_by_format;
  NSMutableDictionary<NSNumber*, id<MTLComputePipelineState>>* _export_by_format;
  __strong id<MTLComputePipelineState> _import_depth_pipeline;
  __strong id<MTLComputePipelineState> _import_exposure_pipeline;
  __strong id<MTLComputePipelineState> _export_color_pipeline;
}

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                 session:(NeuralStudioDLSSSRSession*)session
                                   error:(NSError**)error {
  self = [super init];
  if (self == nil) return nil;
  if (device == nil || session == nil || session.isConsumed ||
      session.inputWidth == 0 || session.inputHeight == 0 ||
      session.outputWidth == 0 || session.outputHeight == 0) {
    SetError(error, 1, @"SR session contract mismatch");
    return nil;
  }
  _device = device;
  _session = session;
  _color = [device newBufferWithLength:kColorRowBytes * kInputHeight
                               options:MTLResourceStorageModeShared];
  _motion = [device newBufferWithLength:kMotionRowBytes * kInputHeight
                                options:MTLResourceStorageModeShared];
  _depth = [device newBufferWithLength:kDepthRowBytes * kInputHeight
                               options:MTLResourceStorageModeShared];
  _exposure = [device newBufferWithLength:kExposureRowBytes
                                  options:MTLResourceStorageModeShared];
  _output = [device newBufferWithLength:kOutputRowBytes * kOutputHeight
                                options:MTLResourceStorageModeShared];
  NSError* pipeline_error = nil;
  NSURL* converter_url = [[session.runtimeBundleURL
      URLByAppendingPathComponent:@"metallib"]
      URLByAppendingPathComponent:@"dlss_sr_live_texture_convert.metallib"];
  id<MTLLibrary> converter = [device newLibraryWithURL:converter_url
                                                 error:&pipeline_error];
  id<MTLFunction> import_color =
      [converter newFunctionWithName:@"neuralstudio_sr_import_rgb9e5"];
  id<MTLFunction> import_depth =
      [converter newFunctionWithName:@"neuralstudio_sr_import_depth32_stencil8"];
  id<MTLFunction> import_exposure =
      [converter newFunctionWithName:@"neuralstudio_sr_import_exposure_r16"];
  id<MTLFunction> export_color =
      [converter newFunctionWithName:@"neuralstudio_sr_export_rgb9e5"];
  if (import_color != nil) {
    _import_color_pipeline =
        [device newComputePipelineStateWithFunction:import_color error:&pipeline_error];
  }
  if (import_depth != nil) {
    _import_depth_pipeline =
        [device newComputePipelineStateWithFunction:import_depth error:&pipeline_error];
  }
  if (import_exposure != nil) {
    _import_exposure_pipeline =
        [device newComputePipelineStateWithFunction:import_exposure error:&pipeline_error];
  }
  if (export_color != nil) {
    _export_color_pipeline =
        [device newComputePipelineStateWithFunction:export_color error:&pipeline_error];
  }
  // Format extension, in its own library so the verified RE9 converter metallib is never touched.
  // Its absence is not an error: it only widens which games can be attached.
  _import_by_format = [NSMutableDictionary dictionary];
  _export_by_format = [NSMutableDictionary dictionary];
  NSURL* ext_url = [[session.runtimeBundleURL URLByAppendingPathComponent:@"metallib"]
      URLByAppendingPathComponent:@"netal_texture_import_ext.metallib"];
  if (id<MTLLibrary> ext = [device newLibraryWithURL:ext_url error:nil]) {
    const struct { MTLPixelFormat format; NSString* import_name; NSString* export_name; } variants[] = {
      {MTLPixelFormatRGBA16Float,  @"neuralstudio_sr_import_rgba16f",    @"neuralstudio_sr_export_rgba16f"},
      {MTLPixelFormatRG11B10Float, @"neuralstudio_sr_import_rg11b10f",   @"neuralstudio_sr_export_rg11b10f"},
      {MTLPixelFormatRGBA8Unorm,   @"neuralstudio_sr_import_rgba8unorm", @"neuralstudio_sr_export_rgba8unorm"},
      {MTLPixelFormatBGRA8Unorm,   @"neuralstudio_sr_import_rgba8unorm", @"neuralstudio_sr_export_rgba8unorm"},
      {MTLPixelFormatRGB10A2Unorm, @"neuralstudio_sr_import_rgb10a2",    @"neuralstudio_sr_export_rgb10a2"},
      {MTLPixelFormatBGR10A2Unorm, @"neuralstudio_sr_import_rgb10a2",    @"neuralstudio_sr_export_rgb10a2"},
    };
    for (const auto& variant : variants) {
      NSNumber* key = @(variant.format);
      if (id<MTLFunction> fn = [ext newFunctionWithName:variant.import_name]) {
        if (id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:nil]) {
          _import_by_format[key] = pso;
        }
      }
      if (id<MTLFunction> fn = [ext newFunctionWithName:variant.export_name]) {
        if (id<MTLComputePipelineState> pso = [device newComputePipelineStateWithFunction:fn error:nil]) {
          _export_by_format[key] = pso;
        }
      }
    }
  }
  if (_color == nil || _motion == nil || _depth == nil || _exposure == nil ||
      _output == nil || _import_color_pipeline == nil ||
      _import_depth_pipeline == nil || _import_exposure_pipeline == nil ||
      _export_color_pipeline == nil) {
    SetError(error, 2,
             pipeline_error == nil ? @"texture staging or converter setup failed"
                                   : pipeline_error.localizedDescription);
    return nil;
  }
  return self;
}


static constexpr NSUInteger kMetal4Slots = 6;

- (BOOL)prepareMetal4 {
  if (_metal4_residency != nil) return YES;
  if (![_device supportsFamily:MTLGPUFamilyMetal4]) return NO;
  _metal4_constants = [_device newBufferWithLength:64 options:MTLResourceStorageModeShared];
  std::uint32_t* constants = static_cast<std::uint32_t*>(_metal4_constants.contents);
  constants[0] = static_cast<std::uint32_t>(kInputWidth);
  constants[1] = static_cast<std::uint32_t>(kInputHeight);
  constants[8] = static_cast<std::uint32_t>(kOutputWidth);   // byte offset 32
  constants[9] = static_cast<std::uint32_t>(kOutputHeight);
  MTLResidencySetDescriptor* residency_descriptor = [[MTLResidencySetDescriptor alloc] init];
  residency_descriptor.initialCapacity = 8;
  NSError* residency_error = nil;
  _metal4_residency = [_device newResidencySetWithDescriptor:residency_descriptor error:&residency_error];
  if (_metal4_residency == nil) return NO;
  for (id<MTLBuffer> buffer in @[_color, _motion, _depth, _exposure, _output, _metal4_constants]) {
    [_metal4_residency addAllocation:buffer];
  }
  [_metal4_residency commit];
  MTL4ArgumentTableDescriptor* table_descriptor = [[MTL4ArgumentTableDescriptor alloc] init];
  table_descriptor.maxBufferBindCount = 2;
  table_descriptor.maxTextureBindCount = 1;
  _metal4_tables = [NSMutableArray array];
  for (NSUInteger index = 0; index < kMetal4Slots * 3; ++index) {
    NSError* table_error = nil;
    id<MTL4ArgumentTable> table = [_device newArgumentTableWithDescriptor:table_descriptor error:&table_error];
    if (table == nil) { _metal4_residency = nil; return NO; }
    [_metal4_tables addObject:table];
  }
  return YES;
}

/// A game's colour texture is acceptable when it is the RGB9E5 the original converter unpacks, or
/// - only when the extension library loaded - RGBA16Float, which the colour buffer already holds
/// natively (kColorRowBytes is width * 8, one half4 per pixel). Games differ here: RE9 supplies
/// RGB9E5, Lies of P supplies RGBA16Float.
/// Describes exactly which resource fails the contract, so a rejection is actionable instead of
/// requiring a bisect. Returns nil when everything matches.
- (NSString*)contractMismatchFor:(id<MTLTexture>)color
                          motion:(id<MTLTexture>)motion
                           depth:(id<MTLTexture>)depth
                        exposure:(id<MTLTexture>)exposure
                          output:(id<MTLTexture>)output {
  // Every attribute TextureMatches tests, not just the obvious three: a texture can have the right
  // size and format and still be rejected for being an array, multisampled, memoryless, or not a
  // plain 2D texture - and a message that omits those reads as "want exactly what you have".
  auto describe = [](id<MTLTexture> t) -> NSString* {
    if (t == nil) return @"none";
    return [NSString stringWithFormat:@"%lux%lu fmt=%lu type=%lu depth=%lu array=%lu samples=%lu storage=%lu",
                                      (unsigned long)t.width, (unsigned long)t.height,
                                      (unsigned long)t.pixelFormat, (unsigned long)t.textureType,
                                      (unsigned long)t.depth, (unsigned long)t.arrayLength,
                                      (unsigned long)t.sampleCount, (unsigned long)t.storageMode];
  };
  if (![self acceptsColorTexture:color]) {
    return [NSString stringWithFormat:@"color %@ (want at least %lux%lu, fmt 93 RGB9E5 or one of %@)",
            describe(color), (unsigned long)kInputWidth, (unsigned long)kInputHeight,
              [self supportedFormatList:_import_by_format]];
  }
  if (!TextureAtLeast(motion, kInputWidth, kInputHeight,
                     MTLPixelFormatRG16Float)) {
    return [NSString stringWithFormat:@"motion %@ (want at least %lux%lu fmt 65 RG16Float)",
            describe(motion), (unsigned long)kInputWidth, (unsigned long)kInputHeight];
  }
  if (!TextureAtLeast(depth, kInputWidth, kInputHeight,
                     MTLPixelFormatDepth32Float_Stencil8)) {
    return [NSString stringWithFormat:@"depth %@ (want at least %lux%lu fmt 260)",
            describe(depth), (unsigned long)kInputWidth, (unsigned long)kInputHeight];
  }
  if (![self acceptsOutputTexture:output]) {
    return [NSString stringWithFormat:
            @"output %@ (want at least %lux%lu, fmt 93 RGB9E5 or one of %@, type=2 depth=1 array=1 samples=1)",
            describe(output), (unsigned long)kOutputWidth, (unsigned long)kOutputHeight,
            [self supportedFormatList:_export_by_format]];
  }
  if (exposure != nil && !TextureMatches(exposure, 1, 1, MTLPixelFormatR16Float)) {
    return [NSString stringWithFormat:@"exposure %@ (want 1x1 fmt 25, or none)", describe(exposure)];
  }
  return nil;
}

/// A colour texture is acceptable at the right size when it is the RGB9E5 the original converter
/// unpacks, or a format the extension library covers. Games differ here: RE9 supplies RGB9E5,
/// Lies of P supplies RGBA16Float.
/// Comma-separated pixel-format numbers a table can handle, for diagnostics.
- (NSString*)supportedFormatList:(NSDictionary<NSNumber*, id<MTLComputePipelineState>>*)table {
  NSMutableArray<NSString*>* names = [NSMutableArray array];
  for (NSNumber* key in [table.allKeys sortedArrayUsingSelector:@selector(compare:)]) {
    [names addObject:key.stringValue];
  }
  return names.count ? [names componentsJoinedByString:@","] : @"none";
}

- (BOOL)acceptsColorTexture:(id<MTLTexture>)color {
  if (TextureAtLeast(color, kInputWidth, kInputHeight,
                    MTLPixelFormatRGB9E5Float)) {
    return YES;
  }
  if (color == nil || _import_by_format[@(color.pixelFormat)] == nil) return NO;
  return TextureAtLeast(color, kInputWidth, kInputHeight,
                       color.pixelFormat);
}

/// An output texture is acceptable when it is RGB9E5, or a format the extension can pack into.
- (BOOL)acceptsOutputTexture:(id<MTLTexture>)output {
  if (TextureAtLeast(output, kOutputWidth, kOutputHeight, MTLPixelFormatRGB9E5Float)) return YES;
  if (output == nil || _export_by_format[@(output.pixelFormat)] == nil) return NO;
  return TextureAtLeast(output, kOutputWidth, kOutputHeight, output.pixelFormat);
}

/// The import pipeline matching whatever the game actually handed us.
- (id<MTLComputePipelineState>)importPipelineForColorTexture:(id<MTLTexture>)color {
  if (color != nil && color.pixelFormat != MTLPixelFormatRGB9E5Float) {
    if (id<MTLComputePipelineState> pso = _import_by_format[@(color.pixelFormat)]) return pso;
  }
  return _import_color_pipeline;
}

/// The export pipeline matching the game's output texture.
- (id<MTLComputePipelineState>)exportPipelineForOutputTexture:(id<MTLTexture>)output {
  if (output != nil && output.pixelFormat != MTLPixelFormatRGB9E5Float) {
    if (id<MTLComputePipelineState> pso = _export_by_format[@(output.pixelFormat)]) return pso;
  }
  return _export_color_pipeline;
}

- (BOOL)encodeTemporalFrameToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command
                                colorTexture:(id<MTLTexture>)color
                               motionTexture:(id<MTLTexture>)motion
                                depthTexture:(id<MTLTexture>)depth
                             exposureTexture:(id<MTLTexture>)exposure
                               outputTexture:(id<MTLTexture>)output
                           inputContentWidth:(NSUInteger)content_width
                          inputContentHeight:(NSUInteger)content_height
                               jitterOffsetX:(float)jitter_x
                               jitterOffsetY:(float)jitter_y
                          motionVectorScaleX:(float)motion_scale_x
                          motionVectorScaleY:(float)motion_scale_y
                                 preExposure:(float)pre_exposure
                                       reset:(BOOL)reset
                               depthReversed:(BOOL)depth_reversed
                                       error:(NSError**)error {
  if (command == nil) {
    SetError(error, 3, @"Metal 4 command buffer missing");
    return NO;
  }
  const bool live_textures =
      [self acceptsColorTexture:color] &&
      TextureAtLeast(motion, kInputWidth, kInputHeight, MTLPixelFormatRG16Float) &&
      TextureAtLeast(depth, kInputWidth, kInputHeight,
                    MTLPixelFormatDepth32Float_Stencil8) &&
      [self acceptsOutputTexture:output];
  if (content_width != kInputWidth || content_height != kInputHeight || !live_textures ||
      (exposure != nil && !TextureMatches(exposure, 1, 1, MTLPixelFormatR16Float))) {
    NSString* why = [self contractMismatchFor:color motion:motion depth:depth
                                     exposure:exposure output:output] ?: @"scalars";
    SetError(error, 5, [NSString stringWithFormat:@"%lux%lu to %lux%lu contract mismatch (metal4): %@",
                        (unsigned long)kInputWidth, (unsigned long)kInputHeight, (unsigned long)kOutputWidth,
                        (unsigned long)kOutputHeight, why]);
    return NO;
  }
  // The motion scale is not hardcoded downstream: the session derives the input words as
  // motion_scale / render_ratio, so any finite non-zero scale is arithmetically fine. This gate is
  // deliberately narrower than the maths - it restricts the runtime to the one game whose output
  // was verified against RTX receipts.
  //
  // A native macOS game reports its own convention (Lies of P uses 1,1 - motion already in pixels
  // - where RE9 reports the render size). Accepting those requires an explicit opt-in, because
  // there is no oracle for a native title: nothing here can prove the result correct, only
  // plausible. NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE=1 means "exploring an unverified game",
  // mirroring NEURALSTUDIO_SR_ALLOW_ANY_JITTER.
  static const bool any_motion_scale = [] {
    const char* value = std::getenv("NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE");
    return value != nullptr && *value == '1';
  }();
  const bool motion_scale_exact = motion_scale_x == static_cast<float>(kInputWidth) &&
                                  motion_scale_y == static_cast<float>(kInputHeight);
  const bool motion_scale_usable = std::isfinite(motion_scale_x) && std::isfinite(motion_scale_y) &&
                                   motion_scale_x != 0.0f && motion_scale_y != 0.0f;
  const bool live_scalars = depth_reversed &&
                            (motion_scale_exact || (any_motion_scale && motion_scale_usable));
  const bool jitter_ok = std::isfinite(jitter_x) && std::isfinite(jitter_y) &&
                         std::fabs(jitter_x) <= 0.5f && std::fabs(jitter_y) <= 0.5f;
  if (pre_exposure != 1.0f || !live_scalars || !jitter_ok) {
    SetError(error, 6, @"temporal-frame scalar contract mismatch");
    return NO;
  }
  if (![self prepareMetal4]) {
    SetError(error, 33, @"Metal 4 unavailable on this device");
    return NO;
  }
  if (exposure == nil) {
    std::memset(_exposure.contents, 0, _exposure.length);
    *static_cast<float*>(_exposure.contents) = 1.0f;
  }
  const NSUInteger slot = _metal4_slot++ % kMetal4Slots;
  id<MTL4ArgumentTable> table_color = _metal4_tables[slot * 3];
  id<MTL4ArgumentTable> table_exposure = _metal4_tables[slot * 3 + 1];
  id<MTL4ArgumentTable> table_export = _metal4_tables[slot * 3 + 2];
  [command useResidencySet:_metal4_residency];

  // Import: RGB9E5 color -> RGBA16F buffer, exposure texel, motion and depth copies.
  id<MTL4ComputeCommandEncoder> import_encoder = [command computeCommandEncoder];
  if (import_encoder == nil) {
    SetError(error, 33, @"Metal 4 compute encoder unavailable");
    return NO;
  }
  [import_encoder barrierAfterQueueStages:MTLStageAll
                             beforeStages:(MTLStageDispatch | MTLStageBlit)
                        visibilityOptions:MTL4VisibilityOptionDevice];
  id<MTLFence> fence = self.pendingFence;
  if (fence != nil) [import_encoder waitForFence:fence beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)];
  [table_color setAddress:_color.gpuAddress atIndex:0];
  [table_color setAddress:_metal4_constants.gpuAddress atIndex:1];
  [table_color setTexture:color.gpuResourceID atIndex:0];
  [import_encoder setComputePipelineState:[self importPipelineForColorTexture:color]];
  [import_encoder setArgumentTable:table_color];
  [import_encoder dispatchThreads:MTLSizeMake(kInputWidth, kInputHeight, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  if (exposure != nil) {
    [table_exposure setAddress:_exposure.gpuAddress atIndex:0];
    [table_exposure setTexture:exposure.gpuResourceID atIndex:0];
    [import_encoder setComputePipelineState:_import_exposure_pipeline];
    [import_encoder setArgumentTable:table_exposure];
    [import_encoder dispatchThreads:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
  }
  [import_encoder copyFromTexture:motion
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_motion
                destinationOffset:0
           destinationBytesPerRow:kMotionRowBytes
         destinationBytesPerImage:kMotionRowBytes * kInputHeight];
  [import_encoder copyFromTexture:depth
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_depth
                destinationOffset:0
           destinationBytesPerRow:kDepthRowBytes
         destinationBytesPerImage:kDepthRowBytes * kInputHeight
                          options:MTLBlitOptionDepthFromDepthStencil];
  [import_encoder endEncoding];

  if (![_session encodeFrameToMTL4CommandBuffer:command
                                    colorBuffer:_color
                                    colorOffset:0
                                   motionBuffer:_motion
                                   motionOffset:0
                                    depthBuffer:_depth
                                    depthOffset:0
                                 exposureBuffer:_exposure
                                 exposureOffset:0
                                   outputBuffer:_output
                                   outputOffset:0
                                  jitterOffsetX:jitter_x
                                  jitterOffsetY:jitter_y
                             motionVectorScaleX:motion_scale_x
                             motionVectorScaleY:motion_scale_y
                                          reset:reset
                                  depthReversed:depth_reversed
                                          error:error]) {
    return NO;
  }

  // Export: RGBA16F output buffer -> the game's RGB9E5 output texture.
  id<MTL4ComputeCommandEncoder> export_encoder = [command computeCommandEncoder];
  if (export_encoder == nil) {
    SetError(error, 33, @"Metal 4 compute encoder unavailable");
    return NO;
  }
  [export_encoder barrierAfterQueueStages:MTLStageAll
                             beforeStages:MTLStageDispatch
                        visibilityOptions:MTL4VisibilityOptionDevice];
  [table_export setAddress:_output.gpuAddress atIndex:0];
  [table_export setAddress:_metal4_constants.gpuAddress + 32 atIndex:1];
  [table_export setTexture:output.gpuResourceID atIndex:0];
  [export_encoder setComputePipelineState:_export_color_pipeline];
  [export_encoder setArgumentTable:table_export];
  [export_encoder dispatchThreads:MTLSizeMake(kOutputWidth, kOutputHeight, 1)
            threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
  if (fence != nil) [export_encoder updateFence:fence afterEncoderStages:MTLStageDispatch];
  [export_encoder endEncoding];
  return YES;
}

- (id<MTLBuffer>)diagnosticModelOutputBuffer { return _output; }
- (id<MTLBuffer>)diagnosticColorBuffer { return _color; }
- (id<MTLBuffer>)diagnosticMotionBuffer { return _motion; }
- (id<MTLBuffer>)diagnosticDepthBuffer { return _depth; }
- (id<MTLBuffer>)diagnosticExposureBuffer { return _exposure; }

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)command
                            colorTexture:(id<MTLTexture>)color
                           motionTexture:(id<MTLTexture>)motion
                            depthTexture:(id<MTLTexture>)depth
                         exposureTexture:(id<MTLTexture>)exposure
                           outputTexture:(id<MTLTexture>)output
                       inputContentWidth:(NSUInteger)content_width
                      inputContentHeight:(NSUInteger)content_height
                           jitterOffsetX:(float)jitter_x
                           jitterOffsetY:(float)jitter_y
                      motionVectorScaleX:(float)motion_scale_x
                      motionVectorScaleY:(float)motion_scale_y
                             preExposure:(float)pre_exposure
                                   reset:(BOOL)reset
                           depthReversed:(BOOL)depth_reversed
                                   error:(NSError**)error {
  if (command == nil || command.device != _device) {
    SetError(error, 3, @"command buffer device mismatch");
    return NO;
  }
  if (_session.isConsumed) {
    SetError(error, 4, @"one-frame adapter already consumed");
    return NO;
  }
  const bool baseline_textures =
      TextureMatches(color, kInputWidth, kInputHeight, MTLPixelFormatRGBA16Float) &&
      TextureMatches(motion, kInputWidth, kInputHeight, MTLPixelFormatRG16Float) &&
      TextureMatches(depth, kInputWidth, kInputHeight, MTLPixelFormatR32Float) &&
      TextureMatches(output, kOutputWidth, kOutputHeight, MTLPixelFormatRGBA16Float);
  const bool live_textures =
      [self acceptsColorTexture:color] &&
      TextureAtLeast(motion, kInputWidth, kInputHeight, MTLPixelFormatRG16Float) &&
      TextureAtLeast(depth, kInputWidth, kInputHeight,
                    MTLPixelFormatDepth32Float_Stencil8) &&
      [self acceptsOutputTexture:output];
  if (content_width != kInputWidth || content_height != kInputHeight ||
      (!baseline_textures && !live_textures) ||
      (baseline_textures && exposure != nil &&
       !TextureMatches(exposure, 1, 1, MTLPixelFormatR32Float)) ||
      (live_textures && exposure != nil &&
       !TextureMatches(exposure, 1, 1, MTLPixelFormatR16Float))) {
    NSString* why = [self contractMismatchFor:color motion:motion depth:depth
                                     exposure:exposure output:output] ?: @"scalars";
    SetError(error, 5, [NSString stringWithFormat:@"%lux%lu to %lux%lu contract mismatch: %@",
                        (unsigned long)kInputWidth, (unsigned long)kInputHeight,
                        (unsigned long)kOutputWidth, (unsigned long)kOutputHeight, why]);
    return NO;
  }
  const bool baseline_scalars = !depth_reversed && jitter_x == 0.0f &&
                                jitter_y == 0.0f && motion_scale_x == 1.0f &&
                                motion_scale_y == 1.0f;
  const bool live_scalars =
      depth_reversed && motion_scale_x == static_cast<float>(kInputWidth) &&
      motion_scale_y == static_cast<float>(kInputHeight) &&
      NeuralStudioDLSSSRIsProvenLiveJitter(jitter_x, jitter_y);
  if (!reset || pre_exposure != 1.0f ||
      (!(baseline_textures && baseline_scalars) &&
       !(live_textures && live_scalars))) {
    SetError(error, 6, @"reset-frame scalar contract mismatch");
    return NO;
  }

  std::memset(_color.contents, 0x5a, _color.length);
  std::memset(_motion.contents, 0x5a, _motion.length);
  std::memset(_depth.contents, 0x5a, _depth.length);
  std::memset(_exposure.contents, 0, _exposure.length);
  std::memset(_output.contents, 0x5a, _output.length);
  *static_cast<float*>(_exposure.contents) = 1.0f;

  if (baseline_textures) {
    id<MTLBlitCommandEncoder> import_encoder = [command blitCommandEncoder];
    [import_encoder copyFromTexture:color
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_color
                destinationOffset:0
           destinationBytesPerRow:kColorRowBytes
         destinationBytesPerImage:kColorRowBytes * kInputHeight];
    [import_encoder copyFromTexture:motion
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_motion
                destinationOffset:0
           destinationBytesPerRow:kMotionRowBytes
         destinationBytesPerImage:kMotionRowBytes * kInputHeight];
    [import_encoder copyFromTexture:depth
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_depth
                destinationOffset:0
           destinationBytesPerRow:kDepthRowBytes
         destinationBytesPerImage:kDepthRowBytes * kInputHeight];
    if (exposure != nil) {
      [import_encoder copyFromTexture:exposure
                        sourceSlice:0
                        sourceLevel:0
                       sourceOrigin:MTLOriginMake(0, 0, 0)
                         sourceSize:MTLSizeMake(1, 1, 1)
                           toBuffer:_exposure
                  destinationOffset:0
             destinationBytesPerRow:kExposureRowBytes
           destinationBytesPerImage:kExposureRowBytes];
    }
    [import_encoder endEncoding];
  } else {
    const std::array<std::uint32_t, 2> content_size = {
        static_cast<unsigned int>(kInputWidth),
        static_cast<unsigned int>(kInputHeight)};
    id<MTLComputeCommandEncoder> convert = [command computeCommandEncoder];
    [convert setComputePipelineState:[self importPipelineForColorTexture:color]];
    [convert setTexture:color atIndex:0];
    [convert setBuffer:_color offset:0 atIndex:0];
    [convert setBytes:&content_size length:sizeof(content_size) atIndex:1];
    [convert dispatchThreads:MTLSizeMake(kInputWidth, kInputHeight, 1)
       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [convert endEncoding];
    if (exposure != nil) {
      convert = [command computeCommandEncoder];
      [convert setComputePipelineState:_import_exposure_pipeline];
      [convert setTexture:exposure atIndex:0];
      [convert setBuffer:_exposure offset:0 atIndex:0];
      [convert dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [convert endEncoding];
    }
    id<MTLBlitCommandEncoder> import_blit = [command blitCommandEncoder];
    [import_blit copyFromTexture:motion
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                        toBuffer:_motion
               destinationOffset:0
          destinationBytesPerRow:kMotionRowBytes
        destinationBytesPerImage:kMotionRowBytes * kInputHeight];
    [import_blit copyFromTexture:depth
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                        toBuffer:_depth
               destinationOffset:0
          destinationBytesPerRow:kDepthRowBytes
        destinationBytesPerImage:kDepthRowBytes * kInputHeight
                         options:MTLBlitOptionDepthFromDepthStencil];
    [import_blit endEncoding];
  }

  if (![_session encodeResetFrameToCommandBuffer:command
                                      colorBuffer:_color
                                      colorOffset:0
                                     motionBuffer:_motion
                                     motionOffset:0
                                      depthBuffer:_depth
                                      depthOffset:0
                                   exposureBuffer:_exposure
                                   exposureOffset:0
                                     outputBuffer:_output
                                     outputOffset:0
                                    jitterOffsetX:jitter_x
                                    jitterOffsetY:jitter_y
                               motionVectorScaleX:motion_scale_x
                               motionVectorScaleY:motion_scale_y
                                    depthReversed:depth_reversed
                                            error:error]) {
    return NO;
  }

  if (baseline_textures) {
    id<MTLBlitCommandEncoder> export_encoder = [command blitCommandEncoder];
    [export_encoder copyFromBuffer:_output
                    sourceOffset:0
               sourceBytesPerRow:kOutputRowBytes
             sourceBytesPerImage:kOutputRowBytes * kOutputHeight
                      sourceSize:MTLSizeMake(kOutputWidth, kOutputHeight, 1)
                       toTexture:output
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    [export_encoder endEncoding];
  } else {
    const std::array<std::uint32_t, 2> output_size = {
        static_cast<unsigned int>(kOutputWidth),
        static_cast<unsigned int>(kOutputHeight)};
    id<MTLComputeCommandEncoder> export_encoder = [command computeCommandEncoder];
    [export_encoder setComputePipelineState:_export_color_pipeline];
    [export_encoder setBuffer:_output offset:0 atIndex:0];
    [export_encoder setTexture:output atIndex:0];
    [export_encoder setBytes:&output_size length:sizeof(output_size) atIndex:1];
    [export_encoder dispatchThreads:MTLSizeMake(kOutputWidth, kOutputHeight, 1)
             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [export_encoder endEncoding];
  }
  return YES;
}


- (BOOL)encodeTemporalFrameToCommandBuffer:(id<MTLCommandBuffer>)command
                            colorTexture:(id<MTLTexture>)color
                           motionTexture:(id<MTLTexture>)motion
                            depthTexture:(id<MTLTexture>)depth
                         exposureTexture:(id<MTLTexture>)exposure
                           outputTexture:(id<MTLTexture>)output
                       inputContentWidth:(NSUInteger)content_width
                      inputContentHeight:(NSUInteger)content_height
                           jitterOffsetX:(float)jitter_x
                           jitterOffsetY:(float)jitter_y
                      motionVectorScaleX:(float)motion_scale_x
                      motionVectorScaleY:(float)motion_scale_y
                             preExposure:(float)pre_exposure
                                   reset:(BOOL)reset
                           depthReversed:(BOOL)depth_reversed
                                   error:(NSError**)error {
  if (command == nil || command.device != _device) {
    SetError(error, 3, @"command buffer device mismatch");
    return NO;
  }
  const bool baseline_textures =
      TextureMatches(color, kInputWidth, kInputHeight, MTLPixelFormatRGBA16Float) &&
      TextureMatches(motion, kInputWidth, kInputHeight, MTLPixelFormatRG16Float) &&
      TextureMatches(depth, kInputWidth, kInputHeight, MTLPixelFormatR32Float) &&
      TextureMatches(output, kOutputWidth, kOutputHeight, MTLPixelFormatRGBA16Float);
  const bool live_textures =
      [self acceptsColorTexture:color] &&
      TextureAtLeast(motion, kInputWidth, kInputHeight, MTLPixelFormatRG16Float) &&
      TextureAtLeast(depth, kInputWidth, kInputHeight,
                    MTLPixelFormatDepth32Float_Stencil8) &&
      [self acceptsOutputTexture:output];
  if (content_width != kInputWidth || content_height != kInputHeight ||
      (!baseline_textures && !live_textures) ||
      (baseline_textures && exposure != nil &&
       !TextureMatches(exposure, 1, 1, MTLPixelFormatR32Float)) ||
      (live_textures && exposure != nil &&
       !TextureMatches(exposure, 1, 1, MTLPixelFormatR16Float))) {
    NSString* why = [self contractMismatchFor:color motion:motion depth:depth
                                     exposure:exposure output:output] ?: @"scalars";
    SetError(error, 5, [NSString stringWithFormat:@"%lux%lu to %lux%lu contract mismatch: %@",
                        (unsigned long)kInputWidth, (unsigned long)kInputHeight,
                        (unsigned long)kOutputWidth, (unsigned long)kOutputHeight, why]);
    return NO;
  }
  // Sustained contract: RE9 live scalars (reversed depth, 1280x720 motion
  // scale), pre-exposure 1 (not modelled otherwise), finite jitter within the
  // pixel. Jitter is consumed bit-for-bit by the RTX-verified kernels.
  // The motion scale is not hardcoded downstream: the session derives the input words as
  // motion_scale / render_ratio, so any finite non-zero scale is arithmetically fine. This gate is
  // deliberately narrower than the maths - it restricts the runtime to the one game whose output
  // was verified against RTX receipts.
  //
  // A native macOS game reports its own convention (Lies of P uses 1,1 - motion already in pixels
  // - where RE9 reports the render size). Accepting those requires an explicit opt-in, because
  // there is no oracle for a native title: nothing here can prove the result correct, only
  // plausible. NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE=1 means "exploring an unverified game",
  // mirroring NEURALSTUDIO_SR_ALLOW_ANY_JITTER.
  static const bool any_motion_scale = [] {
    const char* value = std::getenv("NEURALSTUDIO_SR_ALLOW_ANY_MOTION_SCALE");
    return value != nullptr && *value == '1';
  }();
  const bool motion_scale_exact = motion_scale_x == static_cast<float>(kInputWidth) &&
                                  motion_scale_y == static_cast<float>(kInputHeight);
  const bool motion_scale_usable = std::isfinite(motion_scale_x) && std::isfinite(motion_scale_y) &&
                                   motion_scale_x != 0.0f && motion_scale_y != 0.0f;
  const bool live_scalars = depth_reversed &&
                            (motion_scale_exact || (any_motion_scale && motion_scale_usable));
  const bool jitter_ok = std::isfinite(jitter_x) && std::isfinite(jitter_y) &&
                         std::fabs(jitter_x) <= 0.5f && std::fabs(jitter_y) <= 0.5f;
  if (pre_exposure != 1.0f || !live_scalars || !jitter_ok) {
    SetError(error, 6, @"temporal-frame scalar contract mismatch");
    return NO;
  }

  // Staging buffers are fully rewritten by the import step every frame; only
  // the exposure default needs a value when the game passes no texture.
  if (exposure == nil) {
    std::memset(_exposure.contents, 0, _exposure.length);
    *static_cast<float*>(_exposure.contents) = 1.0f;
  }

  id<MTLFence> fence = self.pendingFence;
  if (baseline_textures) {
    id<MTLBlitCommandEncoder> import_encoder = [command blitCommandEncoder];
    if (fence != nil) [import_encoder waitForFence:fence];
    [import_encoder copyFromTexture:color
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_color
                destinationOffset:0
           destinationBytesPerRow:kColorRowBytes
         destinationBytesPerImage:kColorRowBytes * kInputHeight];
    [import_encoder copyFromTexture:motion
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_motion
                destinationOffset:0
           destinationBytesPerRow:kMotionRowBytes
         destinationBytesPerImage:kMotionRowBytes * kInputHeight];
    [import_encoder copyFromTexture:depth
                      sourceSlice:0
                      sourceLevel:0
                     sourceOrigin:MTLOriginMake(0, 0, 0)
                       sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                         toBuffer:_depth
                destinationOffset:0
           destinationBytesPerRow:kDepthRowBytes
         destinationBytesPerImage:kDepthRowBytes * kInputHeight];
    if (exposure != nil) {
      [import_encoder copyFromTexture:exposure
                        sourceSlice:0
                        sourceLevel:0
                       sourceOrigin:MTLOriginMake(0, 0, 0)
                         sourceSize:MTLSizeMake(1, 1, 1)
                           toBuffer:_exposure
                  destinationOffset:0
             destinationBytesPerRow:kExposureRowBytes
           destinationBytesPerImage:kExposureRowBytes];
    }
    [import_encoder endEncoding];
  } else {
    const std::array<std::uint32_t, 2> content_size = {
        static_cast<unsigned int>(kInputWidth),
        static_cast<unsigned int>(kInputHeight)};
    id<MTLComputeCommandEncoder> convert = [command computeCommandEncoder];
    if (fence != nil) [convert waitForFence:fence];
    [convert setComputePipelineState:[self importPipelineForColorTexture:color]];
    [convert setTexture:color atIndex:0];
    [convert setBuffer:_color offset:0 atIndex:0];
    [convert setBytes:&content_size length:sizeof(content_size) atIndex:1];
    [convert dispatchThreads:MTLSizeMake(kInputWidth, kInputHeight, 1)
       threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [convert endEncoding];
    if (exposure != nil) {
      convert = [command computeCommandEncoder];
      [convert setComputePipelineState:_import_exposure_pipeline];
      [convert setTexture:exposure atIndex:0];
      [convert setBuffer:_exposure offset:0 atIndex:0];
      [convert dispatchThreads:MTLSizeMake(1, 1, 1)
         threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
      [convert endEncoding];
    }
    id<MTLBlitCommandEncoder> import_blit = [command blitCommandEncoder];
    [import_blit copyFromTexture:motion
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                        toBuffer:_motion
               destinationOffset:0
          destinationBytesPerRow:kMotionRowBytes
        destinationBytesPerImage:kMotionRowBytes * kInputHeight];
    [import_blit copyFromTexture:depth
                     sourceSlice:0
                     sourceLevel:0
                    sourceOrigin:MTLOriginMake(0, 0, 0)
                      sourceSize:MTLSizeMake(kInputWidth, kInputHeight, 1)
                        toBuffer:_depth
               destinationOffset:0
          destinationBytesPerRow:kDepthRowBytes
        destinationBytesPerImage:kDepthRowBytes * kInputHeight
                         options:MTLBlitOptionDepthFromDepthStencil];
    [import_blit endEncoding];
  }

  if (![_session encodeFrameToCommandBuffer:command
                                 colorBuffer:_color
                                 colorOffset:0
                                motionBuffer:_motion
                                motionOffset:0
                                 depthBuffer:_depth
                                 depthOffset:0
                              exposureBuffer:_exposure
                              exposureOffset:0
                                outputBuffer:_output
                                outputOffset:0
                               jitterOffsetX:jitter_x
                               jitterOffsetY:jitter_y
                          motionVectorScaleX:motion_scale_x
                          motionVectorScaleY:motion_scale_y
                                       reset:reset
                               depthReversed:depth_reversed
                                       error:error]) {
    return NO;
  }

  if (baseline_textures) {
    id<MTLBlitCommandEncoder> export_encoder = [command blitCommandEncoder];
    [export_encoder copyFromBuffer:_output
                    sourceOffset:0
               sourceBytesPerRow:kOutputRowBytes
             sourceBytesPerImage:kOutputRowBytes * kOutputHeight
                      sourceSize:MTLSizeMake(kOutputWidth, kOutputHeight, 1)
                       toTexture:output
                destinationSlice:0
                destinationLevel:0
               destinationOrigin:MTLOriginMake(0, 0, 0)];
    if (fence != nil) [export_encoder updateFence:fence];
    [export_encoder endEncoding];
  } else {
    const std::array<std::uint32_t, 2> output_size = {
        static_cast<unsigned int>(kOutputWidth),
        static_cast<unsigned int>(kOutputHeight)};
    id<MTLComputeCommandEncoder> export_encoder = [command computeCommandEncoder];
    [export_encoder setComputePipelineState:_export_color_pipeline];
    [export_encoder setBuffer:_output offset:0 atIndex:0];
    [export_encoder setTexture:output atIndex:0];
    [export_encoder setBytes:&output_size length:sizeof(output_size) atIndex:1];
    [export_encoder dispatchThreads:MTLSizeMake(kOutputWidth, kOutputHeight, 1)
             threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    if (fence != nil) [export_encoder updateFence:fence];
    [export_encoder endEncoding];
  }
  return YES;
}

@end
