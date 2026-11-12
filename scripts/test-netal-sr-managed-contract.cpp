// Synthetic CPU controls only: no Metal, model, application or physical device execution.
#include "../src/macos_host/dlss_sr_frame_ledger.h"
#include "../src/macos_host/dlss_sr_neural_backend.h"
#include <cassert>
#include <iostream>
using namespace netal::sr;

FrameToken reserve(FrameLedger& l, std::uintptr_t cb, bool reset = false) {
  FrameToken t; assert(l.reserve(cb, 10, reset, t) == FrameResult::Accepted); return t;
}
FrameToken encode(FrameLedger& l, std::uintptr_t cb, bool reset = false) {
  auto t = reserve(l, cb, reset); assert(l.beginEncoding(t) == FrameResult::Accepted); assert(l.encoded(t) == FrameResult::Accepted); assert(l.finalize(t) == FrameResult::Accepted); return t;
}
FrameLedger::Submission issue(FrameLedger& l, std::vector<FrameToken> t) {
  FrameLedger::Submission b; assert(l.issue(t, 10, b) == FrameResult::Accepted); return b;
}
void provisional_and_exact_ownership() {
  FrameLedger l(5);
  auto a=encode(l,1,true), b=encode(l,2), c=encode(l,3);
  assert(a.frame==0 && b.frame==1 && c.frame==2 && l.scheduledFrames()==3);
  auto first=issue(l,{a}), second=issue(l,{b,c});
  assert(l.feedback(second,true,{true,true})==FrameResult::Accepted);
  assert(l.scheduledFrames()==3 && l.completedFrames()==2 && l.retiredPrefix()==0);
  auto d=encode(l,4); assert(d.frame==3 && d.slot==b.slot && d.generation!=b.generation);
  // Old callback cannot release a newly reused slot or advance its parity.
  assert(l.feedback(second,true,{true,true})==FrameResult::WrongAssociation);
  assert(l.ownership(d)==FrameOwnership::Encoded && l.scheduledFrames()==4);
  assert(l.feedback(first,true,{true})==FrameResult::Accepted);
  assert(l.retiredPrefix()==3 && !l.drained());
}
void admission_and_cuts() {
  FrameLedger l(6); FrameToken scratch;
  assert(l.reserve(1,10,false,scratch)==FrameResult::NeedsReset);
  auto a=reserve(l,1,true);
  assert(l.reserve(2,10,false,scratch)==FrameResult::Backpressure);
  assert(l.reserve(1,10,true,scratch)==FrameResult::WrongAssociation);
  assert(l.beginEncoding(a)==FrameResult::Accepted);
  assert(l.encoded(a)==FrameResult::Accepted);
  assert(l.finalize(a)==FrameResult::Accepted);
  assert(l.reserve(2,11,false,scratch)==FrameResult::WrongAssociation);
  auto b=encode(l,2); auto c=reserve(l,3); assert(l.beginEncoding(c)==FrameResult::Accepted); assert(l.encoded(c)==FrameResult::Accepted);
  FrameLedger::Submission rejected;
  assert(l.issue({b},10,rejected)==FrameResult::WrongOrder);
  assert(l.issue({b,a},10,rejected)==FrameResult::WrongOrder);
  auto batch=issue(l,{a,b});
  assert(l.reserve(4,10,true,scratch)==FrameResult::NeedsDrain);
  assert(l.discard(a)==FrameResult::WrongState);
  // Unsubmitted failed successor cuts the chain but does not release submitted dependencies.
  assert(l.failEncoding(c)==FrameResult::Accepted && l.needsReset());
  assert(l.issue({c},10,rejected)==FrameResult::NeedsReset);
  assert(l.discard(c)==FrameResult::Accepted);
  assert(l.feedback(batch,true,{true,true})==FrameResult::Accepted);
  assert(l.drained() && l.needsReset());
  auto reset=encode(l,4,true); assert(reset.epoch!=a.epoch && reset.frame==0);
  assert(l.discard(a)==FrameResult::WrongAssociation && l.ownership(reset)==FrameOwnership::Encoded);
}
void failed_and_unknown_work_is_not_drain() {
  for (int failure=0;failure<3;++failure) {
    FrameLedger l(7); auto a=encode(l,1,true), b=encode(l,2), c=encode(l,3);
    auto batch=issue(l,{a,b});
    if (failure==0) assert(l.feedback(batch,false,{true,true})==FrameResult::Accepted); // Metal error
    if (failure==1) assert(l.feedback(batch,true,{true,false})==FrameResult::Accepted); // CB reset/reused: missing marker
    if (failure==2) l.ambiguous(batch); // thrown/unknown commit, no callback
    assert(l.needsReset() && !l.drained() && l.completedFrames()==0);
    const auto expected = failure==2 ? FrameOwnership::SubmitUnknown : FrameOwnership::Quarantined;
    assert(l.ownership(a)==expected && l.ownership(b)==expected);
    assert(l.discard(a)==FrameResult::WrongState);
    FrameLedger::Submission retry;
    assert(l.issue({a,b},10,retry)==FrameResult::NeedsReset);
    assert(l.discard(c)==FrameResult::Accepted);
    FrameToken t; assert(l.reserve(4,10,true,t)==FrameResult::NeedsDrain);
    if (failure==2) {
      assert(l.feedback(batch,true,{true,true})==FrameResult::Accepted);
      assert(l.drained() && l.needsReset() && l.completedFrames()==2); // ownership only, never content revival
    } else {
      assert(l.feedback(batch,true,{true,true})==FrameResult::WrongAssociation);
      assert(!l.drained() && l.completedFrames()==0);
    }
  }
}
void early_feedback_and_slots() {
  FrameLedger l(8); auto a=encode(l,1,true); auto batch=issue(l,{a});
  // Simulated callback inside commit before its return/exception handling.
  assert(l.feedback(batch,true,{true})==FrameResult::Accepted);
  l.ambiguous(batch); assert(l.drained() && !l.needsReset() && l.completedFrames()==1);
  FrameLedger::Submission reused;
  assert(l.issue({a},10,reused)==FrameResult::WrongAssociation);
  std::vector<FrameToken> full;
  for (int i=0;i<6;++i) full.push_back(encode(l,20+i));
  FrameToken no_mutation;
  assert(l.reserve(99,10,false,no_mutation)==FrameResult::Backpressure);
  assert(l.scheduledFrames()==7); // reservation rejected before any encoder/table mutation
  auto all=issue(l,full);
  auto foreign=all; ++foreign.identity;
  assert(l.feedback(foreign,true,{true,true,true,true,true,true})==FrameResult::WrongAssociation);
  assert(l.feedback(all,true,{true,true,true,true,true,true})==FrameResult::Accepted);
  assert(l.drained() && l.completedFrames()==7 && l.retiredPrefix()==7);
}
void bounded_missing_callback() {
  FrameLedger l(9); auto a=encode(l,1,true); auto missing=issue(l,{a});
  for (int i=0;i<2000;++i) {
    auto t=encode(l,2); auto batch=issue(l,{t});
    assert(l.feedback(batch,true,{true})==FrameResult::Accepted);
    assert(l.retirementRangeCount()<=FrameLedger::capacity && !l.drained());
  }
  assert(l.scheduledFrames()==2001 && l.retiredPrefix()==0 && l.retirementRangeCount()==1);
  assert(l.feedback(missing,true,{true})==FrameResult::Accepted);
  assert(l.retiredPrefix()==2001 && l.drained() && l.retirementRangeCount()==0);
}
void qualification_is_not_capability() {
  FeatureLayout layout{"feature-plane-v1",64,0,2,{2,2,2,2},{16,8,4,2}};
  assert(layout.bounded());
  NeuralCoreContract c{"same-CB-ML","caller-model","profile",layout,layout,0xff,64,128,true,true};
  assert(InspectNeuralCoreContract(nullptr,"profile",layout,layout,256)==NeuralCoreReadiness::NativeDefault);
  assert(InspectNeuralCoreContract(&c,"profile",layout,layout,256)==NeuralCoreReadiness::Unqualified);
  auto bad=c; bad.phaseMask=1; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.profileIdentity="other"; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.input.strides[0]++; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.input.elementType=FeatureElementType::Float32; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.modelIdentity.clear(); assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.distinctInputOutput=false; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  bad=c; bad.sameCommandBuffer=false; assert(InspectNeuralCoreContract(&bad,"profile",layout,layout,256)==NeuralCoreReadiness::InvalidContract);
  assert(InspectNeuralCoreContract(&c,"profile",layout,layout,100)==NeuralCoreReadiness::InvalidContract);
  layout.strides[0]=UINT64_MAX; assert(!layout.bounded());
  NativeFeatureLayouts native;
  assert(NativeHWCFeatureLayouts(736,352,256,20725760+512,7,native));
  assert(native.input.spanBytes()==8290304 && native.output.spanBytes()==20725760);
  NeuralCoreContract full{"same-CB","caller-model","native-profile",native.input,native.output,0xff,0,
      native.input.spanBytes()+native.output.spanBytes(),true,true};
  assert(InspectNeuralCoreContract(&full,"native-profile",native.input,native.output,full.residentBytes)==NeuralCoreReadiness::Unqualified);
  assert(native.input.offset==256 && native.output.offset==256 && native.input.dimensions[3]==16 && native.output.dimensions[3]==40);
  assert(!NativeHWCFeatureLayouts(736,352,256,8290304+256,0,native));
  assert(!NativeHWCFeatureLayouts(UINT64_MAX,352,256,UINT64_MAX,0,native));
  assert(!NativeHWCFeatureLayouts(736,352,256,UINT64_MAX,8,native));
}
int main() {
  provisional_and_exact_ownership(); admission_and_cuts(); failed_and_unknown_work_is_not_drain();
  early_feedback_and_slots(); bounded_missing_callback(); qualification_is_not_capability();
  std::cout << "synthetic_cpu_contract=pass metal_runtime=unverified model=none\n";
}
