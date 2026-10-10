#pragma once
// The experimental runtime's per-context state. Extracted from nr_experimental_runtime.mm so the bounded
// reset worker can live in its OWN translation unit: that keeps the machine-checked property that the
// encode-only path's file never commits, never installs a completion handler and never creates a queue.
#import <Metal/Metal.h>
#include "../nr_experimental_api.h"
#include "nr_experimental_assets.h"
#include "nr_experimental_block0.h"
#include "nr_experimental_encoder.h"
#include "nr_body_runtime.h"
#include "nr_temporal/nr_temporal.h"
#include <mutex>
#include <string>

namespace netal_nr_experimental {

struct Context {
  std::mutex lock;
  id<MTLDevice> device = nil;
  nr_exp::Assets assets;
  nr_exp::Block0Resources block0;
  nr_exp::EncoderResources encoder;
  NrBodyContext body;                  // pipelines, libraries and resident buffers, all built during prepare()
  NSDictionary* descriptor = nil;      // the body bundle's chain.json
  std::string bundle;
  PrepareOptions options;
  bool prepared = false;
  bool history_valid = false;              // candidate-owned NR history; never the game's
  Arithmetic arithmetic = Arithmetic::OriginalKernels;   // history built under one accumulation rule is never valid under the other
  NrTemporalState temporal;                // OUR history texture, created in prepare(); never the game's
  id<MTLTexture> output = nil;             // OUR output surface, created in prepare() so the bridge can see it before commit
  uint32_t frames_since_reset = 0;         // reset-LOCAL step handed to block 0, not the caller's global frame id
  uint64_t continuity_generation = 1;      // bumped by every invalidation; a ticket from an older generation may not advance history
  uint64_t last_stream_id = 0, last_frame_id = 0;
  bool have_last_frame = false;            // stream/frame continuity is ENFORCED for temporal frames, not merely recorded
  uint64_t admitted_stream = 0, admitted_frame = 0;
  bool frame_in_flight = false;            // at most one admitted frame per context
  std::string invalidate_reason = "never admitted";
  double prepare_cpu_ms = 0;
};

}  // namespace netal_nr_experimental
