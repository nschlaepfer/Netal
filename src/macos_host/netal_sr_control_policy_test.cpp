#include "netal_sr_control_policy.h"
#include <cassert>
#include <iostream>

int main() {
  const auto reference = netal::PlanSRControl("", true, "hiluma_input_depthinv");
  assert(!reference.pipeline && reference.clear_hardware_input);
  const auto fast = netal::PlanSRControl("_hw7", true, "hiluma_input_depthinv");
  assert(fast.pipeline && !fast.clear_hardware_input);
  const auto diagnostic = netal::PlanSRControl("_hw7", false, "hiluma_input_depthinv");
  assert(!diagnostic.pipeline && !diagnostic.clear_hardware_input);
  const auto custom = netal::PlanSRControl("", false, "explicit_developer_kernel");
  assert(!custom.pipeline && !custom.clear_hardware_input);
  const auto clean = netal::PlanSRControl("", false, "");
  assert(!clean.pipeline && !clean.clear_hardware_input);
  assert(netal::MayApplySRControl(true, false));
  assert(!netal::MayApplySRControl(true, true));
  assert(!netal::MayApplySRControl(false, false));
  assert(!netal::MayApplySRControl(false, true));
  netal::ControlPollCadence cadence;
  unsigned sr_polls = 0, fg_polls = 0;
  for (int frame = 0; frame < 120; ++frame) {
    fg_polls += cadence.Due(false);  // FG arrives first: the shared-counter starvation case.
    sr_polls += cadence.Due(true);
  }
  assert(sr_polls == 4 && fg_polls == 4);
  std::cout << "10 SR control policy cases passed\n";
}
