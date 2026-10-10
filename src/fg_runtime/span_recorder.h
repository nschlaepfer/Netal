#pragma once

#import <Metal/Metal.h>

#include "fg_runtime.h"

#include <vector>

void NSFGSpanBegin(bool enabled);
// Deferred mode: intermediate command buffers are committed without a CPU wait (the queue keeps GPU
// order); the output-copy buffer and NSFGSpanFlush() wait for everything. Used when neither the audit
// nor the span profile needs per-command completion; the CPU never touches GPU-visible memory during
// a non-audit frame, so pipelining inside the frame is safe.
void NSFGSpanSetDeferWaits(bool defer);
void NSFGSpanFlush(void);
void NSFGWaitAndRecord(id<MTLCommandBuffer> command, NSFGSpanDomain domain);
const std::vector<NSFGCommandSpan>& NSFGRecordedSpans(void);

