#pragma once
#include "nr_encoder_runtime.h"
#include "nr_temporal.h"
// Experimental only. Caller serializes the shared temporal state with the body.
// History must be the prior LIVE native result of this same sequence.
struct NrEncoderTemporalFrame {
  NrEncoderFrame frame;
  const NrTemporalState* temporal = nullptr;
  id<MTLTexture> motion = nil;
  unsigned sampler_variant = 0; // 0 hardware, 1 Q8 truncation, 2 Q8 nearest; unaccepted candidates
};
// Read-only history/motion; no descriptor or activation filesystem ingress.
// Current experimental admission is full-size shared textures and zero motion.
int nr_encoder_run_temporal_frame(NrEncoderContext&,const NrEncoderTemporalFrame&,
                                 uint8_t,NrEncoderResult&,const char* diagnostics=nullptr);
