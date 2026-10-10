#include "rtx_live_aux_bridge.h"

namespace {
thread_local NSFGRtxLiveAuxiliaryHandles g_handles{};
thread_local bool g_bound = false;
}

void NSFGSetRtxLiveAuxiliaryHandles(
    const NSFGRtxLiveAuxiliaryHandles* handles) {
  if (handles == nullptr) {
    g_handles = {};
    g_bound = false;
    return;
  }
  g_handles = *handles;
  g_bound = true;
}

const NSFGRtxLiveAuxiliaryHandles* NSFGGetRtxLiveAuxiliaryHandles() {
  return g_bound ? &g_handles : nullptr;
}
