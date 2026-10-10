#import "macos_live_handoff.h"

namespace {
id<MTLBuffer> g_live_prefix_input = nil;
}

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input) {
  g_live_prefix_input = input;
}

extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void) {
  return g_live_prefix_input;
}
