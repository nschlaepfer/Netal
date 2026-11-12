#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

FOUNDATION_EXPORT NSErrorDomain const NeuralStudioDLSSSRSessionErrorDomain;
FOUNDATION_EXPORT BOOL NeuralStudioDLSSSRIsProvenLiveJitter(float jitterOffsetX,
                                                            float jitterOffsetY);

@protocol NeuralStudioDLSSSRNeuralCoreBackend;
@class NeuralStudioDLSSSRSession;

/// Opaque ownership of one managed recording. Encode success is provisional, never GPU proof.
/// Retained (including its session/resources) until exact verified feedback or explicit pre-submit discard.
typedef NS_ENUM(NSUInteger, NeuralStudioDLSSSRManagedFrameStatus) {
  NeuralStudioDLSSSRManagedFrameReserved,
  NeuralStudioDLSSSRManagedFrameEncoding,
  NeuralStudioDLSSSRManagedFrameCoreEncoded,
  NeuralStudioDLSSSRManagedFrameEncoded,
  NeuralStudioDLSSSRManagedFrameFailedUnsubmitted,
  NeuralStudioDLSSSRManagedFrameSubmitIssued,
  NeuralStudioDLSSSRManagedFrameSubmitUnknown,
  NeuralStudioDLSSSRManagedFrameQuarantined,
  NeuralStudioDLSSSRManagedFrameRetired,
};
@interface NeuralStudioDLSSSRFrameLease : NSObject
@property(nonatomic, readonly) uint64_t sequence;
@property(nonatomic, readonly) uint64_t scheduledFrameIndex;
/// Safe ownership retirement; raw Metal feedback (including nil error) alone is insufficient.
@property(atomic, readonly, getter=isRetired) BOOL retired;
@property(atomic, readonly) NeuralStudioDLSSSRManagedFrameStatus status;
- (instancetype)init NS_UNAVAILABLE;
@end

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
// Actual libraries selected during initialization for reversed-depth boundaries.
@property(nonatomic, readonly, copy) NSString* depthInputLibraryName;
@property(nonatomic, readonly, copy) NSString* depthOutputLibraryName;

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

/// Opt-in only, before any frame; legacy callers stay untracked. Mixing APIs is rejected.
@property(nonatomic, readonly) BOOL managedMTL4FramesEnabled;
- (BOOL)enableManagedMTL4FramesWithError:(NSError**)error;
/// Lease pins the supplied allocator; caller must supply the allocator backing this recording
/// and must not reset it until every associated lease reports isRetired. This API
/// owns submission. Every CB object entering managed helpers is one-use across the process;
/// retired/discarded/tainted objects cannot be recycled, even as host CBs. Use a fresh CB object.
/// Direct raw commit/re-recording violates ownership. A private GPU marker checks
/// recording identity on nil-error feedback. Wrong/missing markers and errors quarantine resources.
- (nullable NeuralStudioDLSSSRFrameLease*)newManagedFrameLeaseForCommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
                                                                      allocator:(id<MTL4CommandAllocator>)allocator
                                                                      queue:(id<MTL4CommandQueue>)queue
                                                                      reset:(BOOL)reset
                                                                      error:(NSError**)error;
- (BOOL)encodeManagedFrameWithLease:(NeuralStudioDLSSSRFrameLease*)lease
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
/// Explicit discontinuity after external fallback/gap; releases nothing. Next chain needs reset/drain.
- (void)invalidateManagedTemporalChain;
/// Permanently abandon the whole unsubmitted recording before discarding. Submitted/unknown work
/// cannot be discarded or retried. Abandoning any dependency cuts the temporal chain.
- (BOOL)discardUnsubmittedManagedFrameLease:(NeuralStudioDLSSSRFrameLease*)lease error:(NSError**)error;
/// Ends no CBs: caller must end all recordings first. Fresh options are internal and cannot be reused.
/// Options/handler setup failure remains unsubmitted/discardable. Persists SUBMIT_ISSUED before
/// calling Metal; authentic late completion may retire SubmitUnknown ownership without reviving
/// cut content. Supports feedback before return and never retries ambiguous invocation.
- (BOOL)commitManagedCommandBuffers:(NSArray<id<MTL4CommandBuffer>>*)commandBuffers
                             queue:(id<MTL4CommandQueue>)queue
                            leases:(NSArray<NeuralStudioDLSSSRFrameLease*>*)leases
                          feedback:(nullable MTL4CommitFeedbackHandler)feedback
                             error:(NSError**)error;
/// This public source has no trusted full-core numeric/temporal qualification authority yet.
/// Candidate registration returns unavailable; native stays selected. No model is supplied.
- (BOOL)configureOptionalNeuralCoreBackend:(id<NeuralStudioDLSSSRNeuralCoreBackend>)backend error:(NSError**)error;
@property(nonatomic, readonly, copy) NSString* neuralCoreBackendStatus;

/// Frame-ring backpressure counters, for the runtime log: how many frames had to wait for a ring
/// slot, how many gave up (and fell back), and the deepest in-flight count seen. A non-zero wait
/// count means the game was submitting SR frames faster than they complete.
- (void)neuralStudioRingStatsWaits:(uint64_t* _Nullable)waits
                          timeouts:(uint64_t* _Nullable)timeouts
                          maxInFlight:(int* _Nullable)maxInFlight;

@end

NS_ASSUME_NONNULL_END
