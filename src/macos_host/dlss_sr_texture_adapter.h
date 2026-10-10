#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#import "dlss_sr_session.h"

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const NeuralStudioDLSSTextureAdapterErrorDomain;

@interface NeuralStudioDLSSTextureAdapter : NSObject

// MetalFX scaler fence: waited on by the first encoder of the next temporal
// encode (before reading the game's textures) and updated by its last encoder
// (after writing the output), as the replaced scaler would.
@property(nonatomic, strong, nullable) id<MTLFence> pendingFence;
@property(nonatomic, readonly) id<MTLBuffer> diagnosticModelOutputBuffer;
@property(nonatomic, readonly) id<MTLBuffer> diagnosticColorBuffer;     // RGBA16F, input resolution
@property(nonatomic, readonly) id<MTLBuffer> diagnosticMotionBuffer;    // RG16F
@property(nonatomic, readonly) id<MTLBuffer> diagnosticDepthBuffer;     // R32F
@property(nonatomic, readonly) id<MTLBuffer> diagnosticExposureBuffer;  // R32F 1x1

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                                 session:(NeuralStudioDLSSSRSession*)session
                                   error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                            colorTexture:(id<MTLTexture>)colorTexture
                           motionTexture:(id<MTLTexture>)motionTexture
                            depthTexture:(id<MTLTexture>)depthTexture
                         exposureTexture:(nullable id<MTLTexture>)exposureTexture
                           outputTexture:(id<MTLTexture>)outputTexture
                       inputContentWidth:(NSUInteger)inputContentWidth
                      inputContentHeight:(NSUInteger)inputContentHeight
                           jitterOffsetX:(float)jitterOffsetX
                           jitterOffsetY:(float)jitterOffsetY
                      motionVectorScaleX:(float)motionVectorScaleX
                      motionVectorScaleY:(float)motionVectorScaleY
                             preExposure:(float)preExposure
                                   reset:(BOOL)reset
                           depthReversed:(BOOL)depthReversed
                                   error:(NSError**)error;

// Sustained temporal path: same texture contract, any frame (reset or not).
// The session keeps the temporal state; a fallback frame (MetalFX) must be
// followed by reset=YES because the session history missed that frame.
- (BOOL)encodeTemporalFrameToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                               colorTexture:(id<MTLTexture>)colorTexture
                              motionTexture:(id<MTLTexture>)motionTexture
                               depthTexture:(id<MTLTexture>)depthTexture
                            exposureTexture:(nullable id<MTLTexture>)exposureTexture
                              outputTexture:(id<MTLTexture>)outputTexture
                          inputContentWidth:(NSUInteger)inputContentWidth
                         inputContentHeight:(NSUInteger)inputContentHeight
                              jitterOffsetX:(float)jitterOffsetX
                              jitterOffsetY:(float)jitterOffsetY
                         motionVectorScaleX:(float)motionVectorScaleX
                         motionVectorScaleY:(float)motionVectorScaleY
                                preExposure:(float)preExposure
                                      reset:(BOOL)reset
                              depthReversed:(BOOL)depthReversed
                                      error:(NSError**)error;

// Metal 4 (GPTK 4): same temporal frame into an MTL4 command buffer (live game
// textures only); the adapter supplies residency for its staging buffers.
- (BOOL)encodeTemporalFrameToMTL4CommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
                               colorTexture:(id<MTLTexture>)colorTexture
                              motionTexture:(id<MTLTexture>)motionTexture
                               depthTexture:(id<MTLTexture>)depthTexture
                            exposureTexture:(nullable id<MTLTexture>)exposureTexture
                              outputTexture:(id<MTLTexture>)outputTexture
                          inputContentWidth:(NSUInteger)inputContentWidth
                         inputContentHeight:(NSUInteger)inputContentHeight
                              jitterOffsetX:(float)jitterOffsetX
                              jitterOffsetY:(float)jitterOffsetY
                         motionVectorScaleX:(float)motionVectorScaleX
                         motionVectorScaleY:(float)motionVectorScaleY
                                preExposure:(float)preExposure
                                      reset:(BOOL)reset
                              depthReversed:(BOOL)depthReversed
                                      error:(NSError**)error;

@end

NS_ASSUME_NONNULL_END
