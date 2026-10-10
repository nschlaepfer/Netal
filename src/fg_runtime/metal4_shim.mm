#import "metal4_shim.h"

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <unordered_set>
#include <vector>

namespace {
constexpr NSUInteger kSlots = 4;             // frames that may be in flight on the game's queue
constexpr NSUInteger kBindings = 31;         // argument-table capacity (Metal caps buffer binds at 31)
constexpr NSUInteger kConstantsBytes = 1u << 20;
constexpr NSUInteger kConstantsAlign = 256;

struct Binding {
  id<MTLBuffer> buffer = nil;
  NSUInteger offset = 0;
  std::vector<uint8_t> bytes;   // setBytes: payload; wins over `buffer` when non-empty
};
}  // namespace

@interface NSFGM4State : NSObject {
 @public
  id<MTLDevice> device;
  id<MTL4CommandBuffer> target;
  id<MTLCommandBuffer> target3;          // Metal 3 frame: encoders are the real thing, ordered by hazard tracking
  id<MTL4ComputeCommandEncoder> encoder;
  id<MTLResidencySet> residency;
  std::unordered_set<void*> seen;
  bool residency_dirty;
  NSUInteger slot;
  uint64_t frame_index;
  std::vector<id<MTL4ArgumentTable>> tables[kSlots];
  NSUInteger table_cursor;
  id<MTLBuffer> constants[kSlots];
  NSUInteger constants_cursor;
  NSUInteger dispatches;
}
- (id<MTL4ComputeCommandEncoder>)encoderOrThrow;
- (void)noteResource:(id<MTLResource>)resource;
- (id<MTL4ArgumentTable>)nextTable;
- (MTLGPUAddress)pushConstants:(const void*)bytes length:(NSUInteger)length;
- (void)barrierAfter:(MTLStages)stages;
@end

@implementation NSFGM4State
- (id<MTL4ComputeCommandEncoder>)encoderOrThrow {
  if (target == nil) throw std::runtime_error("FG Metal 4 shim: no target command buffer for this frame");
  if (encoder == nil) {
    encoder = [target computeCommandEncoder];
    if (encoder == nil) throw std::runtime_error("FG Metal 4 shim: computeCommandEncoder failed");
    // Everything the game encoded before us (its render) must be visible to our first dispatch.
    [encoder barrierAfterQueueStages:MTLStageAll beforeStages:(MTLStageDispatch | MTLStageBlit)
                   visibilityOptions:MTL4VisibilityOptionDevice];
  }
  return encoder;
}
- (void)noteResource:(id<MTLResource>)resource {
  if (resource == nil) return;
  void* key = (__bridge void*)resource;
  if (seen.insert(key).second) {
    [residency addAllocation:resource];
    residency_dirty = true;
  }
}
- (id<MTL4ArgumentTable>)nextTable {
  auto& ring = tables[slot];
  if (table_cursor == ring.size()) {
    MTL4ArgumentTableDescriptor* d = [[MTL4ArgumentTableDescriptor alloc] init];
    d.maxBufferBindCount = kBindings;
    d.maxTextureBindCount = kBindings;
    d.initializeBindings = YES;
    NSError* error = nil;
    id<MTL4ArgumentTable> table = [device newArgumentTableWithDescriptor:d error:&error];
    if (table == nil) throw std::runtime_error("FG Metal 4 shim: argument table allocation failed");
    ring.push_back(table);
  }
  return ring[table_cursor++];
}
- (MTLGPUAddress)pushConstants:(const void*)bytes length:(NSUInteger)length {
  id<MTLBuffer> ring = constants[slot];
  const NSUInteger aligned = (length + kConstantsAlign - 1) & ~(kConstantsAlign - 1);
  if (constants_cursor + aligned > ring.length)
    throw std::runtime_error("FG Metal 4 shim: constants ring exhausted");
  std::memcpy(static_cast<uint8_t*>(ring.contents) + constants_cursor, bytes, length);
  const MTLGPUAddress address = ring.gpuAddress + constants_cursor;
  constants_cursor += aligned;
  return address;
}
// Barrier policy experiments (NSFG_M4_BARRIER): "device" (default) = per-command barrier with
// device-scope visibility; "none" = per-command barrier without the visibility flush; "off" = no
// barriers at all (bounds the cost; not correct in general).
- (void)barrierAfter:(MTLStages)stages {
  static const int policy = [] {
    const char* v = getenv("NSFG_M4_BARRIER");
    if (v == nullptr) return 0;
    if (strcmp(v, "none") == 0) return 1;
    if (strcmp(v, "off") == 0) return 2;
    return 0;
  }();
  if (policy == 2) return;
  [encoder barrierAfterEncoderStages:stages beforeEncoderStages:(MTLStageDispatch | MTLStageBlit)
                   visibilityOptions:(policy == 1 ? MTL4VisibilityOptionNone : MTL4VisibilityOptionDevice)];
}
@end

// Stands in for both MTLComputeCommandEncoder and MTLBlitCommandEncoder.
@interface NSFGM4Encoder : NSObject {
  NSFGM4State* _state;
  id<MTLComputePipelineState> _pso;
  Binding _buffers[kBindings];
  id<MTLTexture> _textures[kBindings];
}
- (instancetype)initWithState:(NSFGM4State*)state;
@end

@implementation NSFGM4Encoder
- (instancetype)initWithState:(NSFGM4State*)state {
  self = [super init];
  if (self) _state = state;
  return self;
}
- (void)setComputePipelineState:(id<MTLComputePipelineState>)state { _pso = state; }
- (void)setBuffer:(id<MTLBuffer>)buffer offset:(NSUInteger)offset atIndex:(NSUInteger)index {
  if (index >= kBindings) throw std::runtime_error("FG Metal 4 shim: buffer index out of range");
  _buffers[index].buffer = buffer;
  _buffers[index].offset = offset;
  _buffers[index].bytes.clear();
  [_state noteResource:buffer];
}
- (void)setBytes:(const void*)bytes length:(NSUInteger)length atIndex:(NSUInteger)index {
  if (index >= kBindings) throw std::runtime_error("FG Metal 4 shim: bytes index out of range");
  _buffers[index].buffer = nil;
  _buffers[index].bytes.assign(static_cast<const uint8_t*>(bytes), static_cast<const uint8_t*>(bytes) + length);
}
- (void)setTexture:(id<MTLTexture>)texture atIndex:(NSUInteger)index {
  if (index >= kBindings) throw std::runtime_error("FG Metal 4 shim: texture index out of range");
  _textures[index] = texture;
  [_state noteResource:texture];
}
- (void)useResource:(id<MTLResource>)resource usage:(MTLResourceUsage)usage {
  (void)usage;
  [_state noteResource:resource];
}
- (void)bind {
  id<MTL4ComputeCommandEncoder> enc = [_state encoderOrThrow];
  if (_pso == nil) throw std::runtime_error("FG Metal 4 shim: dispatch without a pipeline state");
  id<MTL4ArgumentTable> table = [_state nextTable];
  for (NSUInteger i = 0; i < kBindings; ++i) {
    if (!_buffers[i].bytes.empty()) {
      [table setAddress:[_state pushConstants:_buffers[i].bytes.data() length:_buffers[i].bytes.size()] atIndex:i];
    } else if (_buffers[i].buffer != nil) {
      [table setAddress:_buffers[i].buffer.gpuAddress + _buffers[i].offset atIndex:i];
    }
    if (_textures[i] != nil) [table setTexture:_textures[i].gpuResourceID atIndex:i];
  }
  [enc setComputePipelineState:_pso];
  [enc setArgumentTable:table];
  ++_state->dispatches;
}
- (void)dispatchThreads:(MTLSize)threads threadsPerThreadgroup:(MTLSize)group {
  [self bind];
  [_state->encoder dispatchThreads:threads threadsPerThreadgroup:group];
  [_state barrierAfter:MTLStageDispatch];
}
- (void)dispatchThreadgroups:(MTLSize)groups threadsPerThreadgroup:(MTLSize)group {
  [self bind];
  [_state->encoder dispatchThreadgroups:groups threadsPerThreadgroup:group];
  [_state barrierAfter:MTLStageDispatch];
}
- (void)dispatchThreadgroupsWithIndirectBuffer:(id<MTLBuffer>)indirect indirectBufferOffset:(NSUInteger)offset
                         threadsPerThreadgroup:(MTLSize)group {
  [_state noteResource:indirect];
  [self bind];
  [_state->encoder dispatchThreadgroupsWithIndirectBuffer:indirect.gpuAddress + offset threadsPerThreadgroup:group];
  [_state barrierAfter:MTLStageDispatch];
}
// Blit selectors.
- (void)fillBuffer:(id<MTLBuffer>)buffer range:(NSRange)range value:(uint8_t)value {
  [_state noteResource:buffer];
  [[_state encoderOrThrow] fillBuffer:buffer range:range value:value];
  [_state barrierAfter:MTLStageBlit];
}
- (void)copyFromBuffer:(id<MTLBuffer>)source sourceOffset:(NSUInteger)source_offset toBuffer:(id<MTLBuffer>)destination
     destinationOffset:(NSUInteger)destination_offset size:(NSUInteger)size {
  [_state noteResource:source];
  [_state noteResource:destination];
  [[_state encoderOrThrow] copyFromBuffer:source sourceOffset:source_offset toBuffer:destination
                        destinationOffset:destination_offset size:size];
  [_state barrierAfter:MTLStageBlit];
}
- (void)copyFromTexture:(id<MTLTexture>)source sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl
           sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size toTexture:(id<MTLTexture>)destination
       destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorigin {
  [_state noteResource:source];
  [_state noteResource:destination];
  [[_state encoderOrThrow] copyFromTexture:source sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:size
                                 toTexture:destination destinationSlice:ds destinationLevel:dl destinationOrigin:dorigin];
  [_state barrierAfter:MTLStageBlit];
}
- (void)copyFromTexture:(id<MTLTexture>)source sourceSlice:(NSUInteger)ss sourceLevel:(NSUInteger)sl
           sourceOrigin:(MTLOrigin)so sourceSize:(MTLSize)size toBuffer:(id<MTLBuffer>)destination
      destinationOffset:(NSUInteger)offset destinationBytesPerRow:(NSUInteger)bpr
destinationBytesPerImage:(NSUInteger)bpi {
  [_state noteResource:source];
  [_state noteResource:destination];
  [[_state encoderOrThrow] copyFromTexture:source sourceSlice:ss sourceLevel:sl sourceOrigin:so sourceSize:size
                                  toBuffer:destination destinationOffset:offset destinationBytesPerRow:bpr
                  destinationBytesPerImage:bpi];
  [_state barrierAfter:MTLStageBlit];
}
- (void)copyFromBuffer:(id<MTLBuffer>)source sourceOffset:(NSUInteger)offset sourceBytesPerRow:(NSUInteger)bpr
     sourceBytesPerImage:(NSUInteger)bpi sourceSize:(MTLSize)size toTexture:(id<MTLTexture>)destination
        destinationSlice:(NSUInteger)ds destinationLevel:(NSUInteger)dl destinationOrigin:(MTLOrigin)dorigin {
  [_state noteResource:source];
  [_state noteResource:destination];
  [[_state encoderOrThrow] copyFromBuffer:source sourceOffset:offset sourceBytesPerRow:bpr sourceBytesPerImage:bpi
                               sourceSize:size toTexture:destination destinationSlice:ds destinationLevel:dl
                        destinationOrigin:dorigin];
  [_state barrierAfter:MTLStageBlit];
}
- (void)copyFromTexture:(id<MTLTexture>)source toTexture:(id<MTLTexture>)destination {
  [_state noteResource:source];
  [_state noteResource:destination];
  [[_state encoderOrThrow] copyFromTexture:source toTexture:destination];
  [_state barrierAfter:MTLStageBlit];
}
- (void)endEncoding {}
@end

@interface NSFGM4CommandBuffer : NSObject {
  NSFGM4State* _state;
}
- (instancetype)initWithState:(NSFGM4State*)state;
@end

@implementation NSFGM4CommandBuffer
- (instancetype)initWithState:(NSFGM4State*)state {
  self = [super init];
  if (self) _state = state;
  return self;
}
- (id)computeCommandEncoder {
  if (_state->target3 != nil) return [_state->target3 computeCommandEncoder];
  return [[NSFGM4Encoder alloc] initWithState:_state];
}
- (id)blitCommandEncoder {
  if (_state->target3 != nil) return [_state->target3 blitCommandEncoder];
  return [[NSFGM4Encoder alloc] initWithState:_state];
}
- (void)commit {}
- (void)waitUntilCompleted {}
- (void)addCompletedHandler:(void (^)(id))handler { (void)handler; }
- (MTLCommandBufferStatus)status { return MTLCommandBufferStatusCompleted; }
- (NSError*)error { return nil; }
- (CFTimeInterval)GPUStartTime { return 0; }
- (CFTimeInterval)GPUEndTime { return 0; }
@end

@interface NSFGM4Queue : NSObject {
 @public
  NSFGM4State* _state;
}
@end

@implementation NSFGM4Queue
- (id)commandBuffer { return [[NSFGM4CommandBuffer alloc] initWithState:_state]; }
- (id<MTLDevice>)device { return _state->device; }
@end

id NSFGMetal4ShimCreateQueue(id<MTLDevice> device) {
  NSFGM4State* state = [[NSFGM4State alloc] init];
  state->device = device;
  MTLResidencySetDescriptor* rd = [[MTLResidencySetDescriptor alloc] init];
  rd.initialCapacity = 512;
  NSError* error = nil;
  state->residency = [device newResidencySetWithDescriptor:rd error:&error];
  if (state->residency == nil) return nil;
  for (NSUInteger s = 0; s < kSlots; ++s) {
    state->constants[s] = [device newBufferWithLength:kConstantsBytes options:MTLResourceStorageModeShared];
    if (state->constants[s] == nil) return nil;
    [state noteResource:state->constants[s]];   // the setBytes: payloads live here; the GPU reads them
  }
  NSFGM4Queue* queue = [[NSFGM4Queue alloc] init];
  queue->_state = state;
  return queue;
}

void NSFGMetal4ShimBeginFrame(id shim_queue, id target_buffer) {
  NSFGM4State* s = ((NSFGM4Queue*)shim_queue)->_state;
  // Either kind of command buffer: a Metal 3 buffer gets real encoders and needs no argument tables
  // or residency; a Metal 4 buffer goes through the forwarding encoders below.
  if ([target_buffer conformsToProtocol:@protocol(MTLCommandBuffer)]) {
    s->target3 = (id<MTLCommandBuffer>)target_buffer;
    s->target = nil;
  } else {
    s->target3 = nil;
    s->target = (id<MTL4CommandBuffer>)target_buffer;
  }
  s->encoder = nil;
  s->slot = static_cast<NSUInteger>(s->frame_index % kSlots);
  ++s->frame_index;
  s->table_cursor = 0;
  s->constants_cursor = 0;
  s->dispatches = 0;
}

void NSFGMetal4ShimEndFrame(id shim_queue) {
  NSFGM4State* s = ((NSFGM4Queue*)shim_queue)->_state;
  if (s->encoder != nil) [s->encoder endEncoding];
  s->encoder = nil;
  id<MTL4CommandBuffer> target = s->target;
  s->target = nil;
  s->target3 = nil;
  if (s->residency_dirty) {
    [s->residency commit];
    s->residency_dirty = false;
  }
  // Attached after the commit so this command buffer sees every resource the frame bound; the
  // caller has not ended the command buffer yet.
  if (target != nil) [target useResidencySet:s->residency];
  if (getenv("NSFG_M4_DEBUG") != nullptr)
    fprintf(stderr, "m4_frame=%llu dispatches=%lu tables=%lu constants=%lu resident=%lu\n",
            (unsigned long long)s->frame_index, (unsigned long)s->dispatches, (unsigned long)s->table_cursor,
            (unsigned long)s->constants_cursor, (unsigned long)s->seen.size());
}

NSUInteger NSFGMetal4ShimDispatchCount(id shim_queue) { return ((NSFGM4Queue*)shim_queue)->_state->dispatches; }
NSUInteger NSFGMetal4ShimResidentResourceCount(id shim_queue) {
  return static_cast<NSUInteger>(((NSFGM4Queue*)shim_queue)->_state->seen.size());
}
