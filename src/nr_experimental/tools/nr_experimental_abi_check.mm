// Compiled ABI check for the reset-worker public types (Opus-owned).
//
// This exists because I claimed in v4 that a new ResetProgress field was "trailing, nothing moved". It was
// not: it was inserted before body_command_buffers, output_publishable and error, shifting all three, and a
// consumer compiled against the v3 header then read output_publishable from the wrong offset and saw a
// completed job as not publishable. Source-level intuition about layout is exactly what failed, so the
// layout is now asserted by the COMPILER against the frozen v3 baseline, and the assertions are static:
// this file cannot build if the layout drifts.
#import <Metal/Metal.h>
#include "../nr_experimental_reset_worker.h"
#include <cstddef>
#include <cstdio>
using namespace netal_nr_experimental;

// Baseline captured from ARCHIVE-reset-worker-v3/source/nr_experimental_reset_worker.h, the header Codex's
// consumer is compiled against. Any change here must be a deliberate, announced ABI break.
// Baseline MEASURED by compiling ARCHIVE-reset-worker-v3/source/nr_experimental_reset_worker.h - the header
// Codex's consumer is built against - not guessed. (My first attempt at this file guessed the ResetOptions
// offsets and the compiler rejected them, which is the whole point of doing this in the compiler.)
static_assert(sizeof(ResetProgress)                         == 72, "ResetProgress size changed");
static_assert(offsetof(ResetProgress, state)                == 0,  "ResetProgress::state moved");
static_assert(offsetof(ResetProgress, chunks_total)         == 4,  "ResetProgress::chunks_total moved");
static_assert(offsetof(ResetProgress, chunks_done)          == 8,  "ResetProgress::chunks_done moved");
static_assert(offsetof(ResetProgress, last_chunk_gpu_ms)    == 16, "ResetProgress::last_chunk_gpu_ms moved");
static_assert(offsetof(ResetProgress, worst_chunk_gpu_ms)   == 24, "ResetProgress::worst_chunk_gpu_ms moved");
static_assert(offsetof(ResetProgress, total_gpu_ms)         == 32, "ResetProgress::total_gpu_ms moved");
static_assert(offsetof(ResetProgress, body_command_buffers) == 40, "ResetProgress::body_command_buffers moved");
static_assert(offsetof(ResetProgress, output_publishable)   == 44, "ResetProgress::output_publishable moved - THIS is what v4 broke");
static_assert(offsetof(ResetProgress, error)                == 48, "ResetProgress::error moved");

static_assert(sizeof(ResetOptions)                              == 56, "ResetOptions size changed");
static_assert(offsetof(ResetOptions, arithmetic)                == 24, "ResetOptions::arithmetic moved");
static_assert(offsetof(ResetOptions, max_units_per_chunk)       == 28, "ResetOptions::max_units_per_chunk moved");
static_assert(offsetof(ResetOptions, max_dispatches_per_chunk)  == 32, "ResetOptions::max_dispatches_per_chunk moved");
static_assert(offsetof(ResetOptions, max_encoders_per_chunk)    == 36, "ResetOptions::max_encoders_per_chunk moved");
static_assert(offsetof(ResetOptions, max_chunk_gpu_ms)          == 40, "ResetOptions::max_chunk_gpu_ms moved");
static_assert(offsetof(ResetOptions, body_emission_encoder_bound) == 48, "ResetOptions::body_emission_encoder_bound moved");

int main() {
  printf("{\"schema\": \"neuralstudio.nr.reset-worker-abi.v1\", \"baseline\": \"ARCHIVE-reset-worker-v3 header\",\n");
  printf(" \"ResetProgress\": {\"sizeof\": %zu, \"state\": %zu, \"chunks_total\": %zu, \"chunks_done\": %zu,\n",
         sizeof(ResetProgress), offsetof(ResetProgress, state), offsetof(ResetProgress, chunks_total), offsetof(ResetProgress, chunks_done));
  printf("   \"last_chunk_gpu_ms\": %zu, \"worst_chunk_gpu_ms\": %zu, \"total_gpu_ms\": %zu,\n",
         offsetof(ResetProgress, last_chunk_gpu_ms), offsetof(ResetProgress, worst_chunk_gpu_ms), offsetof(ResetProgress, total_gpu_ms));
  printf("   \"body_command_buffers\": %zu, \"output_publishable\": %zu, \"error\": %zu},\n",
         offsetof(ResetProgress, body_command_buffers), offsetof(ResetProgress, output_publishable), offsetof(ResetProgress, error));
  printf(" \"ResetOptions\": {\"sizeof\": %zu, \"max_units_per_chunk\": %zu, \"max_dispatches_per_chunk\": %zu,\n",
         sizeof(ResetOptions), offsetof(ResetOptions, max_units_per_chunk), offsetof(ResetOptions, max_dispatches_per_chunk));
  printf("   \"max_encoders_per_chunk\": %zu, \"max_chunk_gpu_ms\": %zu, \"body_emission_encoder_bound\": %zu},\n",
         offsetof(ResetOptions, max_encoders_per_chunk), offsetof(ResetOptions, max_chunk_gpu_ms), offsetof(ResetOptions, body_emission_encoder_bound));
  printf(" \"static_assertions\": \"compiled\", \"result\": \"pass\"}\n");
  return 0;
}
