#pragma once

#include <cstdint>

struct NSFGRtxLiveAuxiliaryHandles {
  std::uint64_t auxiliary_mask;
  std::uint64_t flow;
  std::uint64_t working_vector;
  std::uint64_t current_depth;
  std::uint64_t previous_depth;
  std::uint64_t previous_history;
  std::uint64_t next_history;
  std::uint64_t current_projection;
};

void NSFGSetRtxLiveAuxiliaryHandles(
    const NSFGRtxLiveAuxiliaryHandles* handles);
const NSFGRtxLiveAuxiliaryHandles* NSFGGetRtxLiveAuxiliaryHandles();

