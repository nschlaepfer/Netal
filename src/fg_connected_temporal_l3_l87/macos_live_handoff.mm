#import "macos_live_handoff.h"

namespace {
id<MTLBuffer> g_live_prefix_input = nil;
id<MTLTexture> g_allocation29 = nil;
}

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input) {
  g_live_prefix_input = input;
}
extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void) {
  return g_live_prefix_input;
}
extern "C" void FgConnectedSetMacAllocation29(id<MTLTexture> texture) {
  g_allocation29 = texture;
}
extern "C" id<MTLTexture> FgConnectedMacAllocation29(void) {
  return g_allocation29;
}
