#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const NeuralStudioDLSSSRSessionErrorDomain;
FOUNDATION_EXPORT BOOL NeuralStudioDLSSSRIsProvenLiveJitter(float jitterOffsetX,
                                                            float jitterOffsetY);

@interface NeuralStudioDLSSSRSession : NSObject

@property(nonatomic, readonly) NSUInteger inputWidth;
@property(nonatomic, readonly) NSUInteger inputHeight;
@property(nonatomic, readonly) NSUInteger outputWidth;
@property(nonatomic, readonly) NSUInteger outputHeight;
@property(nonatomic, readonly, copy) NSString* runtimeBundleRoot;
@property(nonatomic, readonly, copy) NSURL* runtimeBundleURL;
@property(nonatomic, readonly, getter=isConsumed) BOOL consumed;
@property(nonatomic, readonly) NSUInteger encodedFrameCount;
/// Batched-pipeline coverage for the selected profile. A layer without one silently runs the
/// transpiled per-window kernel - correct, roughly twice the GPU time - so a half-generated profile
/// reads as a healthy session. Exposed so the bridge can log it and the app can show it, rather than
/// it only reaching stderr where nothing reads it.
@property(nonatomic, readonly) NSUInteger pipelineLayersTotal;
@property(nonatomic, readonly) NSUInteger pipelineLayersPresent;
@property(nonatomic, readonly, copy) NSString* pipelineLayersMissing;

- (nullable instancetype)initWithDevice:(id<MTLDevice>)device
                       runtimeBundleURL:(NSURL*)runtimeBundleURL
                                  error:(NSError**)error NS_DESIGNATED_INITIALIZER;

- (instancetype)init NS_UNAVAILABLE;

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                             colorBuffer:(id<MTLBuffer>)colorBuffer
                             colorOffset:(NSUInteger)colorOffset
                            motionBuffer:(id<MTLBuffer>)motionBuffer
                            motionOffset:(NSUInteger)motionOffset
                             depthBuffer:(id<MTLBuffer>)depthBuffer
                             depthOffset:(NSUInteger)depthOffset
                          exposureBuffer:(id<MTLBuffer>)exposureBuffer
                          exposureOffset:(NSUInteger)exposureOffset
                            outputBuffer:(id<MTLBuffer>)outputBuffer
                            outputOffset:(NSUInteger)outputOffset
                                   error:(NSError**)error;

- (BOOL)encodeResetFrameToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                             colorBuffer:(id<MTLBuffer>)colorBuffer
                             colorOffset:(NSUInteger)colorOffset
                            motionBuffer:(id<MTLBuffer>)motionBuffer
                            motionOffset:(NSUInteger)motionOffset
                             depthBuffer:(id<MTLBuffer>)depthBuffer
                             depthOffset:(NSUInteger)depthOffset
                          exposureBuffer:(id<MTLBuffer>)exposureBuffer
                          exposureOffset:(NSUInteger)exposureOffset
                            outputBuffer:(id<MTLBuffer>)outputBuffer
                            outputOffset:(NSUInteger)outputOffset
                           jitterOffsetX:(float)jitterOffsetX
                           jitterOffsetY:(float)jitterOffsetY
                      motionVectorScaleX:(float)motionVectorScaleX
                      motionVectorScaleY:(float)motionVectorScaleY
                           depthReversed:(BOOL)depthReversed
                                   error:(NSError**)error;

- (BOOL)encodeFrameToCommandBuffer:(id<MTLCommandBuffer>)commandBuffer
                        colorBuffer:(id<MTLBuffer>)colorBuffer
                        colorOffset:(NSUInteger)colorOffset
                       motionBuffer:(id<MTLBuffer>)motionBuffer
                       motionOffset:(NSUInteger)motionOffset
                        depthBuffer:(id<MTLBuffer>)depthBuffer
                        depthOffset:(NSUInteger)depthOffset
                     exposureBuffer:(id<MTLBuffer>)exposureBuffer
                     exposureOffset:(NSUInteger)exposureOffset
                       outputBuffer:(id<MTLBuffer>)outputBuffer
                       outputOffset:(NSUInteger)outputOffset
                      jitterOffsetX:(float)jitterOffsetX
                      jitterOffsetY:(float)jitterOffsetY
                 motionVectorScaleX:(float)motionVectorScaleX
                 motionVectorScaleY:(float)motionVectorScaleY
                              reset:(BOOL)reset
                      depthReversed:(BOOL)depthReversed
                              error:(NSError**)error;

// Metal 4 (GPTK 4 / D3DMetal 4): same frame into an MTL4 command buffer; the
// caller supplies residency for the frame buffers it passes.
- (BOOL)encodeFrameToMTL4CommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
                        colorBuffer:(id<MTLBuffer>)colorBuffer
                        colorOffset:(NSUInteger)colorOffset
                       motionBuffer:(id<MTLBuffer>)motionBuffer
                       motionOffset:(NSUInteger)motionOffset
                        depthBuffer:(id<MTLBuffer>)depthBuffer
                        depthOffset:(NSUInteger)depthOffset
                     exposureBuffer:(id<MTLBuffer>)exposureBuffer
                     exposureOffset:(NSUInteger)exposureOffset
                       outputBuffer:(id<MTLBuffer>)outputBuffer
                       outputOffset:(NSUInteger)outputOffset
                      jitterOffsetX:(float)jitterOffsetX
                      jitterOffsetY:(float)jitterOffsetY
                 motionVectorScaleX:(float)motionVectorScaleX
                 motionVectorScaleY:(float)motionVectorScaleY
                              reset:(BOOL)reset
                      depthReversed:(BOOL)depthReversed
                              error:(NSError**)error;

/// Frame-ring backpressure counters, for the runtime log: how many frames had to wait for a ring
/// slot, how many gave up (and fell back), and the deepest in-flight count seen. A non-zero wait
/// count means the game was submitting SR frames faster than they complete.
- (void)neuralStudioRingStatsWaits:(uint64_t* _Nullable)waits
                          timeouts:(uint64_t* _Nullable)timeouts
                          maxInFlight:(int* _Nullable)maxInFlight;

@end

NS_ASSUME_NONNULL_END
