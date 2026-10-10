#include "rtx_temporal_handoff.h"

namespace {
thread_local std::uint64_t g_allocation29_shared_handle = 0;
}

void FgConnectedSetRtxAllocation29Handle(std::uint64_t shared_handle) {
  g_allocation29_shared_handle = shared_handle;
}
std::uint64_t FgConnectedRtxAllocation29Handle(void) {
  return g_allocation29_shared_handle;
}
