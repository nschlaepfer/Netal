#pragma once
#include <cstring>

// Configure once before the process's first FG session: kernel helpers cache
// their selected modes. A mode change requires a fresh game process.
struct NetalFGLowering {
  bool valid;
  bool prefixHardware;
  bool blockHardware;
  bool reference() const { return !prefixHardware && !blockHardware; }
};

inline NetalFGLowering NetalResolveFGLowering(bool requestReference,
                                             const char* prefix,
                                             const char* block) {
  const bool prefixHardware = prefix ? std::strcmp(prefix, "hw") == 0 : !requestReference;
  const bool blockHardware = block ? std::strcmp(block, "hw") == 0 : !requestReference;
  return {!(requestReference && (prefixHardware || blockHardware)), prefixHardware, blockHardware};
}
