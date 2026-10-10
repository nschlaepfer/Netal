// CPU-only ownership/lifetime tests linked to the archived v17 runtime.
// A command mock is used; there is no device, queue, model inference or timing.
#import <Foundation/Foundation.h>
#include "nr_body_runtime.h"
#include "nr_temporal/nr_temporal.h"
#include <iostream>
#include <memory>
#include <type_traits>
static unsigned waits=0;
@interface NRCPUCommandV17 : NSObject
@end
@implementation NRCPUCommandV17
- (void)waitUntilCompleted { ++waits; }
- (void)commit {}
- (MTLCommandBufferStatus)status { return MTLCommandBufferStatusCompleted; }
- (CFTimeInterval)GPUStartTime { return 0; }
- (CFTimeInterval)GPUEndTime { return 0; }
@end
static_assert(!std::is_copy_constructible_v<NrBodyPending> && !std::is_copy_assignable_v<NrBodyPending>);
static_assert(!std::is_copy_constructible_v<NrBodyContext> && !std::is_copy_assignable_v<NrBodyContext>);
static_assert(!std::is_move_constructible_v<NrBodyContext> && !std::is_move_assignable_v<NrBodyContext>);
static void prepare(NrBodyPending& p,NrBodyContext& c,unsigned& calls,NrTemporalState* t=nullptr) {
  p.ctx=&c;p.ctx_alive=c.alive;p.temporal=t;p.frame_id=++c.frame_serial;
  c.in_flight=true;c.in_flight_id=p.frame_id;p.state=NrBodyPending::Encoded;
  p.cmd=(id<MTLCommandBuffer>)[NRCPUCommandV17 new];
  p.complete=[&calls](NrBodyResult& r,double) { ++calls;r.code=0;return 0; };
}
int main() { @autoreleasepool {
  unsigned pass=0,fail=0;
  auto check=[&](const char* label,bool ok) { std::cout<<(ok?"PASS ":"FAIL ")<<label<<"\n";ok?++pass:++fail; };
  {
    NrBodyContext c{};unsigned calls=0;NrBodyPending a;prepare(a,c,calls);
    NrBodyPending b=std::move(a);NrBodyResult r;
    check("moved-source-refused",nr_body_commit(a)==2 && nr_body_complete(a,r)==2 && c.in_flight);
    nr_body_pending_reset(b);
    check("reset-inflight-noop",b.state==NrBodyPending::Encoded && c.in_flight);
    check("unique-owner-completes-once",nr_body_commit(b)==0 && nr_body_complete(b,r)==0 && calls==1 && !c.in_flight);
    NrBodyPending next;prepare(next,c,calls);
    check("old-handles-cannot-clear-next-frame",nr_body_complete(a,r)==2 && nr_body_complete(b,r)==2 && c.in_flight && calls==1);
  }
  {
    unsigned calls=0;NrBodyPending p;auto c=std::make_unique<NrBodyContext>();prepare(p,*c,calls);c.reset();
    unsigned before=waits;
    check("destroyed-encoded-context-refused",nr_body_commit(p)==2 && p.state==NrBodyPending::Failed && waits==before && calls==0);
  }
  {
    unsigned calls=0;NrTemporalState temporal;temporal.frames_since_reset=7;
    NrBodyPending p;auto c=std::make_unique<NrBodyContext>();prepare(p,*c,calls,&temporal);nr_body_commit(p);c.reset();
    unsigned before=waits;NrBodyResult r;
    check("destroyed-committed-context-refused-and-waited",nr_body_complete(p,r)==2 && waits==before+1 && calls==0);
    check("destroyed-context-damages-history-without-advance",temporal.history_sha256.rfind("damaged:",0)==0 && temporal.frames_since_reset==7);
  }
  {
    unsigned calls=0;NrTemporalState t;unsigned before=waits;
    { NrBodyPending p;auto c=std::make_unique<NrBodyContext>();prepare(p,*c,calls,&t);nr_body_commit(p);c.reset(); }
    check("abandon-after-context-destruction-safe",waits==before+1 && calls==0 && t.history_sha256.rfind("damaged:",0)==0);
  }
  {
    NrBodyContext c{};unsigned calls=0;NrBodyPending owner;prepare(owner,c,calls);
    { NrBodyPending stale;stale.ctx=&c;stale.ctx_alive=c.alive;stale.frame_id=owner.frame_id+1;stale.state=NrBodyPending::Committed; }
    check("wrong-id-destructor-does-not-damage-owner",c.damaged.empty() && c.in_flight && c.in_flight_id==owner.frame_id);
  }
  std::cout<<"scope cpu-mock-api-only; gpu_execution false; passed "<<pass<<" failed "<<fail<<"\n";
  return fail?1:0;
} }
