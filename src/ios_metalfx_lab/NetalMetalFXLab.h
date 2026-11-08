#pragma once
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

NS_ASSUME_NONNULL_BEGIN
// Synchronous, model-free comparison. Call on a background queue while active.
// Configuration: metalfx_mode = spatial|temporal; render_size/profile = WxH;
// warmup_frames = 8...32; measured_frames = 1...240; capture = JSON boolean.
// Defaults: temporal, 640x360 -> 1280x720, 8 warm-up + 60 measured frames.
// Captures require a separate MetalCaptureEnabled build, remain private, and
// never produce performance samples. No Netal model/session is allocated.
// shouldStop is checked between completed commands, never by freeing in-flight work.
FOUNDATION_EXPORT NSDictionary<NSString*, id>* NetalRunMetalFXLabWithStop(
    id<MTLDevice> device, NSDictionary<NSString*, id>* configuration,
    BOOL (^ _Nullable shouldStop)(void), NSError* _Nullable* _Nullable error);
NS_ASSUME_NONNULL_END
