#pragma once
// Metal 4 shim for the FG runtime.
//
// The runtime encodes with Metal 3 objects (command queue -> command buffers -> compute/blit
// encoders) and, in non-audit frames, never waits on them. Under GPTK 4 the game hands us an
// MTL4CommandBuffer instead, so this shim stands in for the queue: every command buffer the
// runtime asks for is a lightweight object whose encoders forward into one
// MTL4ComputeCommandEncoder on the target command buffer, with an argument table per dispatch, a
// per-frame constants ring for setBytes:, a barrier after every command, and a residency set that
// learns every resource the runtime binds. The runtime's source is untouched; the objects answer
// the selectors it already sends.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

// A shim queue. Typed id so the runtime can hold it as id<MTLCommandQueue>.
id NSFGMetal4ShimCreateQueue(id<MTLDevice> device);
// Route the runtime's next commands into the caller's command buffer (Metal 4, already begun, or
// Metal 3, not yet committed).
void NSFGMetal4ShimBeginFrame(id shim_queue, id target_buffer);   // id<MTL4CommandBuffer> or id<MTLCommandBuffer>
// End the frame's encoder and commit any residency-set additions. Safe to call once per frame.
void NSFGMetal4ShimEndFrame(id shim_queue);
// Diagnostics.
NSUInteger NSFGMetal4ShimDispatchCount(id shim_queue);
NSUInteger NSFGMetal4ShimResidentResourceCount(id shim_queue);
