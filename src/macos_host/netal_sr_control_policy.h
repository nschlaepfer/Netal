#pragma once

#include <string_view>

namespace netal {
// The empty suffix selects translated reference arithmetic. A contradictory batching request
// must not keep the optimized kernels active underneath a Reference label.
struct SRControlPlan {
  bool pipeline;
  bool clear_hardware_input;
};

inline SRControlPlan PlanSRControl(std::string_view suffix, bool requested_pipeline,
                                   std::string_view current_input_library) {
  const bool reference = suffix.empty();
  return {reference ? false : requested_pipeline,
          reference && current_input_library == "hiluma_input_depthinv"};
}
// FG polling must not mutate SR state; pending initialization must keep its environment stable.
inline bool MayApplySRControl(bool sr_hook_owns_state, bool initialization_pending) {
  return sr_hook_owns_state && !initialization_pending;
}
// Called under the control mutex. Separate cadences prevent FG calls from repeatedly
// consuming every shared polling slot and starving SR configuration updates.
struct ControlPollCadence {
  unsigned long long sr_checks = 0;
  unsigned long long fg_checks = 0;
  bool Due(bool sr_owner) {
    auto& checks = sr_owner ? sr_checks : fg_checks;
    return (checks++ % 30) == 0;
  }
};
}  // namespace netal
