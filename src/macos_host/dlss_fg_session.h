#pragma once
// Netal frame generation session for the game bridge.
//
// Owns one Netal FG runtime session (the exact DLSS-G port, pinned 640x360 profile) in Metal 4
// mode, plus the texture adapter that turns a MetalFX frame interpolator's inputs into the
// runtime's contract and writes the generated frame back into the interpolator's output. The
// runtime works at 640x360 internally; the adapter resamples the game's frames down to it and the
// generated frame back up, so at game resolution this is a resolution-decoupled interpolation,
// not the RTX-exact one. Everything is encoded into the game's MTL4CommandBuffer in the
// interpolator's place; nothing here waits on the GPU.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

@interface NeuralStudioDLSSFGSession : NSObject

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                       runtimeBundleURL:(NSURL*)bundle
                                  error:(NSError**)error;

/// Encode one interpolated frame between `prevColor`-in-lineage and `color` into `output`.
/// `reset` restarts the runtime's temporal state: that frame produces no interpolation and the
/// output receives `color` unchanged (the real frame is presented twice). Returns NO with an error
/// when the contract is not served; the caller must then run the original interpolator.
- (BOOL)encodeToMTL4CommandBuffer:(id<MTL4CommandBuffer>)command
                     colorTexture:(id<MTLTexture>)color
                     depthTexture:(id<MTLTexture>)depth
                    motionTexture:(id<MTLTexture>)motion
                    outputTexture:(id<MTLTexture>)output
                     contentWidth:(NSUInteger)contentWidth
                    contentHeight:(NSUInteger)contentHeight
                      renderWidth:(NSUInteger)renderWidth
                     renderHeight:(NSUInteger)renderHeight
               motionVectorScaleX:(float)motionScaleX
               motionVectorScaleY:(float)motionScaleY
                            fence:(nullable id<MTLFence>)fence
                            reset:(BOOL)reset
                            error:(NSError**)error;

/// Dump-only: the interpolator's UI texture for the frame about to be encoded.
- (void)setDumpUITexture:(nullable id<MTLTexture>)ui;

/// The Metal 3 seam (native MetalFX titles): the same frame into an MTLCommandBuffer the caller
/// will commit.
- (BOOL)encodeToCommandBuffer:(id<MTLCommandBuffer>)command
                 colorTexture:(id<MTLTexture>)color
                 depthTexture:(id<MTLTexture>)depth
                motionTexture:(id<MTLTexture>)motion
                outputTexture:(id<MTLTexture>)output
                 contentWidth:(NSUInteger)contentWidth
                contentHeight:(NSUInteger)contentHeight
                  renderWidth:(NSUInteger)renderWidth
                 renderHeight:(NSUInteger)renderHeight
           motionVectorScaleX:(float)motionScaleX
           motionVectorScaleY:(float)motionScaleY
                        fence:(nullable id<MTLFence>)fence
                        reset:(BOOL)reset
                        error:(NSError**)error;

/// Force the next frame to be a reset (after a fallback, or when the game asks).
- (void)invalidate;

@property (readonly) uint64_t generatedFrames;
@property (readonly) uint64_t resetFrames;
@property (readonly) NSUInteger runtimeDispatchesLastFrame;
/// Frames composited at the game's resolution (launches 55/84 re-evaluated at content size) versus
/// frames that fell back to upsampling the 640x360 result; NEURALSTUDIO_FG_FULLRES=0 disables it.
@property (readonly) uint64_t fullResolutionFrames;
@property (readonly) uint64_t fullResolutionSkipped;
@property (readonly) BOOL fullResolutionEnabled;
/// YES when both configured stages use reference arithmetic. This records selection,
/// not a claim of byte-exact game output. Mode changes require a fresh process.
@property (readonly) BOOL exactLowering;

@end

NS_ASSUME_NONNULL_END
