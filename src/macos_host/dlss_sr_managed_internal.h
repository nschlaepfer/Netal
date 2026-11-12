#import "dlss_sr_session.h"

// Internal adapter plumbing; no completion/success injection is exposed.
@interface NeuralStudioDLSSSRSession (ManagedAdapterInternal)
- (BOOL)neuralStudioValidateManagedEncodingLease:(NeuralStudioDLSSSRFrameLease*)lease error:(NSError**)error;
- (BOOL)neuralStudioBeginManagedLease:(NeuralStudioDLSSSRFrameLease*)lease error:(NSError**)error;
- (NSUInteger)neuralStudioManagedSlotForLease:(NeuralStudioDLSSSRFrameLease*)lease;
- (BOOL)neuralStudioRetainManagedResources:(NSArray*)resources lease:(NeuralStudioDLSSSRFrameLease*)lease error:(NSError**)error;
- (void)neuralStudioFailManagedLease:(NeuralStudioDLSSSRFrameLease*)lease;
- (BOOL)neuralStudioEncodeManagedFrameWithLease:(NeuralStudioDLSSSRFrameLease*)lease
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
- (BOOL)neuralStudioFinishManagedFrameLease:(NeuralStudioDLSSSRFrameLease*)lease error:(NSError**)error;
@end
