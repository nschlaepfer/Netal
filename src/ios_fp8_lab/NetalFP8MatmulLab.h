#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN
// Model-free, synchronous operator experiment. Call on a background queue while
// the app is active. Only physical A20 Pro/iOS 27 is admitted. A successful run
// validates the reported synthetic inputs; it does not validate DLSS, image
// quality, full-frame latency, or the hardware instruction selected by a driver.
// sampleCount: 3...15; repetitionsPerSample: 1...128. Invalid values fail closed.
// Missing driver timestamps produce an error, never zero milliseconds.
FOUNDATION_EXPORT NSDictionary<NSString*, id>* _Nullable NetalRunFP8MatmulLab(
    id<MTLDevice> device, NSURL* libraryURL, NSUInteger sampleCount,
    NSUInteger repetitionsPerSample, NSError* _Nullable* _Nullable error);
// The predicate is queried only between completed GPU commands; returning YES
// stops before another submission. The caller may check cancellation/activity.
FOUNDATION_EXPORT NSDictionary<NSString*, id>* _Nullable NetalRunFP8MatmulLabWithStop(
    id<MTLDevice> device, NSURL* libraryURL, NSUInteger sampleCount,
    NSUInteger repetitionsPerSample, BOOL (^ _Nullable shouldStop)(void),
    NSError* _Nullable* _Nullable error);
NS_ASSUME_NONNULL_END
