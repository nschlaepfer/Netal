#pragma once

#import <Metal/Metal.h>

#include <cstdint>

void NSFGSetTailOutputTarget(id<MTLTexture> target);

// Runtime resources a resolution-decoupled compositor needs after a generated frame: the network's
// 8-channel per-pixel output (two flows + blend logit), launch 53's two candidate buffers, launch
// 54's blend texture and launch 84's sigmoid LUT. Recorded by the tail each frame; identities are
// stable across frames (arena-cached) but re-recorded anyway.
typedef struct NSFGTailExports {
  __unsafe_unretained id<MTLBuffer> motion_half8;      // 640x368x8 halves
  NSUInteger motion_offset;
  __unsafe_unretained id<MTLBuffer> candidate_a;       // 320x180 ulong
  __unsafe_unretained id<MTLBuffer> candidate_b;
  NSUInteger candidate_offset;
  __unsafe_unretained id<MTLTexture> blend_rg16f;      // 320x180
  __unsafe_unretained id<MTLBuffer> sigmoid_lut;       // 65536 x uint (float bits) indexed by half bits
  NSUInteger sigmoid_offset;
} NSFGTailExports;
void NSFGRecordTailExports(id<MTLBuffer> motion, NSUInteger motion_offset, id<MTLBuffer> candidate_a,
                           id<MTLBuffer> candidate_b, NSUInteger candidate_offset, id<MTLTexture> blend,
                           id<MTLBuffer> sigmoid, NSUInteger sigmoid_offset);
NSFGTailExports NSFGTailLastExports(void);

// NSFG_PREFIX_MODE=hw selects the hardware lowering of the encoder/decoder convolutions
// (`<name>_hw.metallib`, an fp32 FMA chain per stage instead of the exact HMMA emulation).
const char* NSFGPrefixLibName(const char* name);
void NSFGCopyTailOutput(id<MTLCommandQueue> queue, id<MTLTexture> source);
void NSFGSetTailContext(id<MTLDevice> device, id<MTLCommandQueue> queue,
                        bool audit_enabled, bool live_auxiliaries);
id<MTLDevice> NSFGTailDevice(void);
id<MTLCommandQueue> NSFGTailCommandQueue(void);
bool NSFGTailAuditEnabled(void);
bool NSFGTailUsesLiveAuxiliaries(void);
void NSFGRecordTailResourceAllocations(std::uint64_t count);
std::uint64_t NSFGTailLastResourceAllocations(void);
void NSFGTailReleasePersistentResources(void);
void NSFGRecordTailCheckpoint(std::size_t index, const char* sha256);
std::size_t NSFGTailCheckpointCount(void);
const char* NSFGTailCheckpoint(std::size_t index);

typedef struct NSFGTailLiveResources {
  __unsafe_unretained id<MTLTexture> auxiliary_mask;
  __unsafe_unretained id<MTLTexture> flow;
  __unsafe_unretained id<MTLTexture> working_vector;
  __unsafe_unretained id<MTLTexture> current_depth;
  __unsafe_unretained id<MTLTexture> previous_depth;
  __unsafe_unretained id<MTLTexture> previous_history;
  __unsafe_unretained id<MTLTexture> next_history;
  __unsafe_unretained id<MTLTexture> current_projection;
} NSFGTailLiveResources;

void NSFGSetTailLiveResources(const NSFGTailLiveResources* resources);
const NSFGTailLiveResources* NSFGTailGetLiveResources(void);
