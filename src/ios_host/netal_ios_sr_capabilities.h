#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN

/// Compatibility for the experimental mobile SR path, not a model, accuracy or latency verdict.
/// GPU names are a compatibility policy, not a security attestation. Unknown labels fail closed.
/// A Pro-chip Air, Duo or iPad is admitted by chip identity rather than the product's marketing name.
@interface NetalIOSSRCapabilities : NSObject

@property(nonatomic, readonly, getter=isEligible) BOOL eligible;
@property(nonatomic, readonly, copy) NSString* chipName;
@property(nonatomic, readonly, copy) NSString* reason;
/// Candidate per-GPU-core Neural Accelerator support, distinct from the Neural Engine.
/// This does not prove that a workload was dispatched to those accelerators.
@property(nonatomic, readonly) BOOL supportsNeuralAccelerators;
/// Successful small E4M3 and E5M2 tensor allocations on iOS 27+, not FP8 arithmetic or SR support.
@property(nonatomic, readonly) BOOL supportsFP8TensorTypes;

+ (instancetype)capabilitiesForDevice:(nullable id<MTLDevice>)device;
- (instancetype)init NS_UNAVAILABLE;
+ (instancetype)new NS_UNAVAILABLE;

@end

NS_ASSUME_NONNULL_END
