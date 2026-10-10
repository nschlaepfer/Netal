#pragma once
#include "nr_encoder_runtime.h"

// Optional, checked-mode measurement only. No shader/dispatch/validation change.
// The synchronous encoder API and NrEncoderResult layout remain unchanged.
// Enabling/disabling clears any earlier snapshot. A new admitted frame clears
// its context's snapshot before execution; only success publishes a replacement.
int nr_encoder_enable_telemetry(NrEncoderContext&, bool enabled, std::string& error);

// Copy this context's most recent successful telemetry-enabled frame as JSON.
// Rejected if unprepared, concurrent, disabled, or no successful snapshot exists.
// Includes frame/reset/input identity, a context-local generation, command GPU
// timestamps (including both input blits), and overlapping CPU phase timings.
// Missing GPU timestamps are explicitly invalid, never a measured zero cost.
// JSON serialization occurs in this accessor, outside the encoder call; callers
// should invoke it outside timing regions. Snapshot construction/collection still
// adds host overhead during the checked call. This is not resident/quiet latency.
int nr_encoder_copy_telemetry(const NrEncoderContext&, std::string& json, std::string& error);
