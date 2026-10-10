#import "macos_subrect_live_handoff.h"

namespace {
id<MTLBuffer> g_subrect_live_prefix_input = nil;
}

extern "C" void FgSubrectSetMacLivePrefixInput(id<MTLBuffer> input) {
  g_subrect_live_prefix_input = input;
}

extern "C" id<MTLBuffer> FgSubrectMacLivePrefixInput(void) {
  return g_subrect_live_prefix_input;
}
