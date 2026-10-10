#include "../src/macos_host/netal_fg_lowering.h"
#include <cassert>
#include <iostream>
int main() {
  auto automatic = NetalResolveFGLowering(false, nullptr, nullptr);
  assert(automatic.valid && automatic.prefixHardware && automatic.blockHardware);
  auto reference = NetalResolveFGLowering(true, nullptr, nullptr);
  assert(reference.valid && reference.reference());
  assert(!NetalResolveFGLowering(true, "hw", nullptr).valid);
  assert(!NetalResolveFGLowering(true, nullptr, "hw").valid);
  assert(!NetalResolveFGLowering(true, "hw", "hw").valid);
  assert(NetalResolveFGLowering(false, "reference", "reference").reference());
  auto mixed = NetalResolveFGLowering(false, "reference", "hw");
  assert(mixed.valid && !mixed.reference() && !mixed.prefixHardware && mixed.blockHardware);
  // The existing kernel helper selects hardware only for the exact string "hw".
  assert(NetalResolveFGLowering(false, "", "other").reference());
  std::cout << "eight FG lowering-selection checks passed\n";
}
