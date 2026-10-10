// CPU-only ownership regression for the two preserved delayed-adapter crashes.
// Links the real bridge but never creates a Metal device or starts NR.
#define NETAL_NR_DELAYED_ONCE 1
#include "netal_nr_experimental_bridge.mm"
#include <cassert>
#include <chrono>
#include <thread>
#include <type_traits>

static void BeginGateFromTemporaryOwner(std::shared_ptr<DelayedNRJob> job) {
  // End the stack block's C++ capture lifetime before checking weak ownership.
  // A block literal in the test loop can retain its copy until that scope ends.
  dispatch_sync(job->host_queue,^{DelayedBeginCompute(job);});
}
int main() { @autoreleasepool {
  using Job = std::shared_ptr<DelayedNRJob>;
  static_assert(std::is_same_v<decltype(&DelayedPrepareInput), void(*)(Job)>);
  static_assert(std::is_same_v<decltype(&DelayedExport), void(*)(Job)>);
  static_assert(std::is_same_v<decltype(&DelayedPoll), void(*)(Job)>);
  static_assert(std::is_same_v<decltype(&DelayedBeginCompute), void(*)(Job)>);
  static_assert(std::is_same_v<decltype(&DelayedWrite), void(*)(Job,const char*,bool)>);
  for(unsigned i=0;i<32;++i) {
    auto queue=dispatch_queue_create("netal.nr.callback-lifetime-test",DISPATCH_QUEUE_SERIAL);
    auto finished=dispatch_semaphore_create(0);
    std::weak_ptr<DelayedNRJob> weak;
    std::atomic<bool> called{false};
    auto* observed=&called;
    dispatch_suspend(queue);
    {
      auto job=std::make_shared<DelayedNRJob>();weak=job;
      // Mirror the actual nested C++ lambda -> escaping ObjC block handoff:
      // the local copy must outlive both the lambda and this caller's owner.
      DelayedSafe(job,[&]{
        const auto keep=job;
        dispatch_async(queue,^{
          observed->store(keep->state.begin_capture(1,7,1920,1200));
          dispatch_semaphore_signal(finished);
        });
      });
      job.reset();
      assert(!weak.expired());
    }
    dispatch_resume(queue);
    assert(dispatch_semaphore_wait(finished,dispatch_time(DISPATCH_TIME_NOW,5*NSEC_PER_SEC))==0);
    dispatch_sync(queue,^{}); // Wait for the previous block's owner to be released.
    assert(called.load() && weak.expired());
  }
  // Exercise the REAL asynchronous gate poll, not just a mirrored capture. An
  // unreleased gate must not enter NR, and cancellation must release its owner.
  for(unsigned i=0;i<16;++i) {
    auto job=std::make_shared<DelayedNRJob>();std::weak_ptr<DelayedNRJob> weak=job;
    job->compute_path=std::string(NSTemporaryDirectory().UTF8String)+NSUUID.UUID.UUIDString.UTF8String+"/unreleased";
    assert(job->state.begin_capture(1,7,1920,1200));assert(job->state.capture_completed(true,true));
    BeginGateFromTemporaryOwner(job);
    assert(job->state.snapshot().phase==netal_nr_adapter::DelayedOnceState::Phase::WaitingForCompute);
    assert(!job->reset_job && !job->state.snapshot().model_computed);
    job->state.fail();job.reset();
    const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
    while(!weak.expired() && std::chrono::steady_clock::now()<deadline)
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
    assert(weak.expired());
  }
  for(unsigned kind=0;kind<2;++kind) {
    auto job=std::make_shared<DelayedNRJob>();
    assert(job->state.begin_capture(1,7,1920,1200));assert(job->state.capture_completed(true,true));
    job->compute_gate_polls=kind?6000:1;
    job->compute_gate_deadline=std::chrono::steady_clock::now()+(kind?std::chrono::seconds(10):std::chrono::seconds(-1));
    BeginGateFromTemporaryOwner(job);
    assert(job->state.snapshot().phase==netal_nr_adapter::DelayedOnceState::Phase::Failed);
    assert(!job->reset_job && !job->state.snapshot().model_computed && !job->state.snapshot().publication_may_be_encoded);
  }
  std::puts("PASS 32 CPU asynchronous-owner,16 actual gate-poll cancellation/lifetime and2 bounded-wait expiry cases; no GPU or model execution");
} }
