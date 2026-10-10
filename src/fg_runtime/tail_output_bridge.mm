#import "tail_output_bridge.h"
#import "span_recorder.h"

#include <algorithm>
#include <stdexcept>
#include <array>
#include <cstring>
#include <cstdlib>
#include <string>

namespace {
thread_local __unsafe_unretained id<MTLTexture> g_tail_output_target = nil;
thread_local __unsafe_unretained id<MTLDevice> g_tail_device = nil;
thread_local __unsafe_unretained id<MTLCommandQueue> g_tail_queue = nil;
thread_local std::uint64_t g_tail_resource_allocations = 0;
thread_local bool g_tail_audit_enabled = false;
thread_local bool g_tail_live_auxiliaries = false;
thread_local std::array<std::array<char, 65>, 68> g_tail_checkpoints{};
thread_local std::size_t g_tail_checkpoint_count = 0;
thread_local NSFGTailLiveResources g_tail_live_resources{};
thread_local bool g_tail_live_resources_bound = false;
thread_local NSFGTailExports g_tail_exports{};
}

void NSFGRecordTailExports(id<MTLBuffer> motion, NSUInteger motion_offset, id<MTLBuffer> candidate_a,
                           id<MTLBuffer> candidate_b, NSUInteger candidate_offset, id<MTLTexture> blend,
                           id<MTLBuffer> sigmoid, NSUInteger sigmoid_offset) {
  g_tail_exports = {motion, motion_offset, candidate_a, candidate_b, candidate_offset, blend, sigmoid, sigmoid_offset};
}

NSFGTailExports NSFGTailLastExports(void) { return g_tail_exports; }

void NSFGSetTailOutputTarget(id<MTLTexture> target) {
  g_tail_output_target = target;
}

void NSFGCopyTailOutput(id<MTLCommandQueue> queue, id<MTLTexture> source) {
  if (g_tail_output_target == nil)
    throw std::runtime_error("FG runtime output target is not bound");
  id<MTLCommandBuffer> command = [queue commandBuffer];
  id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
  [encoder copyFromTexture:source
               sourceSlice:0
               sourceLevel:0
              sourceOrigin:MTLOriginMake(0, 0, 0)
                sourceSize:MTLSizeMake(640, 360, 1)
                 toTexture:g_tail_output_target
          destinationSlice:0
          destinationLevel:0
         destinationOrigin:MTLOriginMake(0, 0, 0)];
  [encoder endEncoding];
  NSFGWaitAndRecord(command, NSFG_SPAN_OUTPUT_COPY);
}

void NSFGSetTailContext(id<MTLDevice> device, id<MTLCommandQueue> queue,
                        bool audit_enabled, bool live_auxiliaries) {
  g_tail_device = device;
  g_tail_queue = queue;
  g_tail_resource_allocations = 0;
  g_tail_audit_enabled = audit_enabled;
  g_tail_live_auxiliaries = live_auxiliaries;
  g_tail_checkpoint_count = 0;
}

id<MTLDevice> NSFGTailDevice(void) { return g_tail_device; }
id<MTLCommandQueue> NSFGTailCommandQueue(void) { return g_tail_queue; }
bool NSFGTailAuditEnabled(void) { return g_tail_audit_enabled; }
bool NSFGTailUsesLiveAuxiliaries(void) { return g_tail_live_auxiliaries; }

void NSFGRecordTailResourceAllocations(std::uint64_t count) {
  g_tail_resource_allocations = count;
}

std::uint64_t NSFGTailLastResourceAllocations(void) {
  return g_tail_resource_allocations;
}

void NSFGRecordTailCheckpoint(std::size_t index, const char* sha256) {
  if (index >= g_tail_checkpoints.size() || sha256 == nullptr ||
      std::strlen(sha256) != 64)
    throw std::runtime_error("FG runtime tail checkpoint shape drift");
  std::memcpy(g_tail_checkpoints[index].data(), sha256, 65);
  g_tail_checkpoint_count = std::max(g_tail_checkpoint_count, index + 1);
}

std::size_t NSFGTailCheckpointCount(void) { return g_tail_checkpoint_count; }

const char* NSFGTailCheckpoint(std::size_t index) {
  return index < g_tail_checkpoint_count
             ? g_tail_checkpoints[index].data()
             : nullptr;
}

void NSFGSetTailLiveResources(const NSFGTailLiveResources* resources) {
  if (resources == nullptr) {
    g_tail_live_resources = {};
    g_tail_live_resources_bound = false;
    return;
  }
  g_tail_live_resources = *resources;
  g_tail_live_resources_bound = true;
}

const NSFGTailLiveResources* NSFGTailGetLiveResources(void) {
  return g_tail_live_resources_bound ? &g_tail_live_resources : nullptr;
}

const char* NSFGPrefixLibName(const char* name) {
  static const bool hw = std::getenv("NSFG_PREFIX_MODE") != nullptr && std::strcmp(std::getenv("NSFG_PREFIX_MODE"), "hw") == 0;
  if (!hw) return name;
  static thread_local std::string buffer;
  std::string stem(name);
  if (stem.size() > 9 && stem.compare(stem.size() - 9, 9, ".metallib") == 0) stem.resize(stem.size() - 9);
  buffer = stem + "_hw.metallib";
  return buffer.c_str();
}
