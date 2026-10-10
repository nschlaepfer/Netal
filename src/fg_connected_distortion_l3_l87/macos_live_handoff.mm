#import "macos_live_handoff.h"

namespace {
id<MTLBuffer> g_live_prefix_input = nil;
id<MTLBuffer> g_ex2_unit_table = nil;
id<MTLBuffer> g_ex2_fraction_table = nil;
}

extern "C" void FgConnectedSetMacLivePrefixInput(id<MTLBuffer> input) {
  g_live_prefix_input = input;
}

extern "C" id<MTLBuffer> FgConnectedMacLivePrefixInput(void) {
  return g_live_prefix_input;
}
extern "C" void FgConnectedSetMacEx2Tables(id<MTLBuffer> unit_table,
                                             id<MTLBuffer> fraction_table) {
  g_ex2_unit_table = unit_table;
  g_ex2_fraction_table = fraction_table;
}
extern "C" id<MTLBuffer> FgConnectedMacEx2UnitTable(void) {
  return g_ex2_unit_table;
}
extern "C" id<MTLBuffer> FgConnectedMacEx2FractionTable(void) {
  return g_ex2_fraction_table;
}
