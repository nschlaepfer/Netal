// CPU-only API ownership counterexample. Links the archived, unmodified v15
// runtime, but substitutes an NSObject for a completed command buffer. This is
// NOT GPU execution, model evidence, parity evidence, or a performance test.
#import <Foundation/Foundation.h>
#include "nr_body_runtime.h"
#include <iostream>
#include <type_traits>

@interface NRCPUCompletedCommand : NSObject
@end
@implementation NRCPUCompletedCommand
- (void)waitUntilCompleted {}
- (MTLCommandBufferStatus)status { return MTLCommandBufferStatusCompleted; }
- (CFTimeInterval)GPUStartTime { return 0; }
- (CFTimeInterval)GPUEndTime { return 0; }
@end

int main() { @autoreleasepool {
  static_assert(std::is_copy_constructible_v<NrBodyPending>);
  static_assert(std::is_copy_assignable_v<NrBodyPending>);
  NrBodyContext context{};
  context.in_flight = true;
  unsigned calls = 0;
  NrBodyPending original;
  original.ctx = &context;
  original.state = NrBodyPending::Committed;
  original.cmd = (id<MTLCommandBuffer>)[NRCPUCompletedCommand new];
  original.complete = [&calls](NrBodyResult& result, double) {
    ++calls;
    result.code = 0;
    return 0;
  };
  NrBodyPending alias = original;  // Public, implicit copy; no field tampering.
  NrBodyResult first, second;
  int rc1 = nr_body_complete(original, first);
  // Model the reservation of the NEXT frame after original completed. The stale
  // alias must not be able to clear that reservation or repeat completion.
  context.in_flight = true;
  int rc2 = nr_body_complete(alias, second);
  bool reproduced = rc1 == 0 && rc2 == 0 && calls == 2 && !context.in_flight;
  std::cout << "{\"scope\":\"cpu-mock-api-only\",\"gpu_execution\":false,"
            << "\"first_rc\":" << rc1 << ",\"alias_rc\":" << rc2
            << ",\"completion_calls\":" << calls
            << ",\"next_frame_reservation_cleared\":" << (!context.in_flight ? "true" : "false")
            << ",\"counterexample_reproduced\":" << (reproduced ? "true" : "false") << "}\n";
  return reproduced ? 0 : 1;
} }
