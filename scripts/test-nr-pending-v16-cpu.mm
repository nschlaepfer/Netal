// Independent CPU-only tests of archived v16 API mechanics. No device, queue,
// model execution or timing evidence. The fake command only models status/wait.
#import <Foundation/Foundation.h>
#include "nr_body_runtime.h"
#include <iostream>
#include <type_traits>
#include <utility>
static unsigned waits=0, commits=0;
@interface NRCPUCommandV16 : NSObject
@end
@implementation NRCPUCommandV16
- (void)waitUntilCompleted { ++waits; }
- (void)commit { ++commits; }
- (MTLCommandBufferStatus)status { return MTLCommandBufferStatusCompleted; }
- (CFTimeInterval)GPUStartTime { return 0; }
- (CFTimeInterval)GPUEndTime { return 0; }
@end
static_assert(!std::is_copy_constructible_v<NrBodyPending>);
static_assert(!std::is_copy_assignable_v<NrBodyPending>);
static_assert(std::is_move_constructible_v<NrBodyPending>);

static void prepare(NrBodyPending& p, NrBodyContext& c, unsigned& calls) {
  p.ctx=&c; p.frame_id=++c.frame_serial;
  c.in_flight=true; c.in_flight_id=p.frame_id;
  p.state=NrBodyPending::Encoded;
  p.cmd=(id<MTLCommandBuffer>)[NRCPUCommandV16 new];
  p.complete=[&calls](NrBodyResult& r,double) { ++calls; r.code=0; return 0; };
}
int main() { @autoreleasepool {
  unsigned passed=0,failed=0;
  auto test=[&](const char* name,bool ok) {
    std::cout<<(ok?"PASS ":"FAIL ")<<name<<"\n";
    ok?++passed:++failed;
  };
  {
    NrBodyContext c{}; unsigned calls=0; NrBodyPending a; prepare(a,c,calls);
    auto b=std::move(a); NrBodyResult r;
    test("moved-from-refused",nr_body_commit(a)==2 && nr_body_complete(a,r)==2);
    nr_body_pending_reset(b);
    test("encoded-reset-noop",b.state==NrBodyPending::Encoded && c.in_flight);
    test("moved-owner-completes",nr_body_commit(b)==0 && nr_body_complete(b,r)==0 && calls==1 && !c.in_flight);
    c.in_flight=true; c.in_flight_id=++c.frame_serial;
    test("moved-alias-cannot-clear-new-frame",nr_body_complete(a,r)==2 && c.in_flight && calls==1);
    test("completed-owner-cannot-repeat",nr_body_complete(b,r)==2 && c.in_flight && calls==1);
    c.in_flight=false; c.in_flight_id=0;
  }
  {
    NrBodyContext c{}; unsigned calls=0; unsigned w=waits;
    { NrBodyPending p; prepare(p,c,calls); }
    test("encoded-abandon-no-wait-no-damage",!c.in_flight && c.damaged.empty() && waits==w && calls==0);
  }
  {
    NrBodyContext c{}; unsigned calls=0; unsigned w=waits;
    { NrBodyPending p; prepare(p,c,calls); nr_body_commit(p); }
    test("committed-abandon-waits-and-damages",!c.in_flight && !c.damaged.empty() && waits==w+1 && calls==0);
  }
  {
    NrBodyContext c{},d{}; unsigned calls=0; NrBodyPending a,b;
    prepare(a,c,calls); prepare(b,d,calls); b=std::move(a);
    test("move-assignment-releases-old-encoded-owner",!d.in_flight && d.damaged.empty() && c.in_flight);
    NrBodyResult r; test("move-assigned-owner-completes",nr_body_commit(b)==0 && nr_body_complete(b,r)==0 && calls==1);
  }
  // Header trait evidence only: copied contexts duplicate the residency map's
  // Objective-C resource handles, not the resources themselves. A GPU aliasing
  // demonstration is outside this CPU-only fixture.
  std::cout<<"context_copy_constructible "<<std::is_copy_constructible_v<NrBodyContext><<"\n";
  std::cout<<"scope cpu-mock-api-only; gpu_execution false; passed "<<passed<<" failed "<<failed<<"\n";
  return failed?1:0;
} }
