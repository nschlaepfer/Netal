// Behavioural tests for the Opus-owned encoder ADOPTION layer (nr_experimental_encoder_scope_adoption.h).
// Codex's shared helper (nr_encoder_scope.h) is not modified and not re-tested here; these cover the
// adoption obligations the runtime owns: exactly-once closure, a double end, an encoder that was never
// adopted, the checked-only snapshot-blit ordering that actually failed, and command-buffer safety after
// an adoption fault. Creating and ending encoders submits no work; nothing is dispatched or committed
// except the final safety check.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdio>
#include <stdexcept>
#include <set>
#include <cstdint>
#include "nr_experimental_encoder_scope_adoption.h"

static int failures = 0;
static void check(const char* name, bool ok, const char* detail) {
  printf("%s %s :: %s\n", ok ? "PASS" : "FAIL", name, detail);
  if (!ok) ++failures;
}

int main() { @autoreleasepool {
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
  if (!dev) { fprintf(stderr, "no Metal device\n"); return 2; }
  id<MTLCommandQueue> q = [dev newCommandQueue];

  // 1. Exactly-once closure through the owning scope.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    h.adopt(e);
    const bool ok = nr_exp::nr_scope_end(e);
    const bool clean = ok && h.ended() == 1 && h.unadopted_ends() == 0 && h.double_ends() == 0 && !h.any_failure();
    h.close_all();                                   // must not end anything a second time
    check("scope-exact-once-closure", clean && !h.any_failure(),
          "one adopt, one end: reported success, counted once, and close_all afterwards is a no-op");
    nr_exp::set_current_scope_holder(nullptr); }

  // 2. A double end is a fault and must NOT end the encoder twice.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    h.adopt(e);
    const bool first = nr_exp::nr_scope_end(e);
    const bool second = nr_exp::nr_scope_end(e);
    check("scope-double-end-is-a-recorded-fault-not-a-second-end",
          first && !second && h.double_ends() == 1 && h.unadopted_ends() == 0 && h.adoption_failed()
          && h.encoder_state_indeterminate(),
          "the second end returns false, is counted as a double end, never as an unadopted encoder and "
          "never as another endEncoding - and it marks the encoder state INDETERMINATE, because a closed "
          "token cannot prove the object is not currently encoding again");
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 3. An encoder the holder never adopted is a recorded fault AND is left untouched. Unknown ownership
  //    is not a licence to end: it may be someone else's live encoder, or already ended.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];   // deliberately NOT adopted
    const bool ok = nr_exp::nr_scope_end(e);
    check("scope-unadopted-encoder-is-not-silent-success",
          !ok && h.unadopted_ends() == 1 && h.double_ends() == 0 && h.foreign_ends() == 0
          && h.adoption_failed() && h.encoder_state_indeterminate(),
          "returns false, records an unadopted end, and reports the encoder state as indeterminate "
          "instead of the old 'nothing to close' success");
    [e endEncoding];                        // the test cleans up what the holder correctly refused to touch
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 3b. An unadopted encoder that has ALREADY been ended. The old behaviour called endEncoding on it,
  //     which is a Metal API error; the holder must record the fault and leave it alone.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    [e endEncoding];                        // ended by someone else, never adopted here
    const bool ok = nr_exp::nr_scope_end(e);
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];   // proves the buffer survived the fault path
    check("scope-unadopted-already-ended-encoder-is-not-ended-again",
          !ok && h.unadopted_ends() == 1 && h.adoption_failed() && b != nil,
          "no second endEncoding on an already-ended encoder; the fault is recorded and the command "
          "buffer is still usable");
    h.adopt(b); nr_exp::nr_scope_end(b); h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 3c. Repeated ends of the same unadopted encoder: each one is counted, none of them ends anything.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    const bool a1 = nr_exp::nr_scope_end(e), a2 = nr_exp::nr_scope_end(e), a3 = nr_exp::nr_scope_end(e);
    check("scope-repeated-unadopted-end-stays-a-fault-every-time",
          !a1 && !a2 && !a3 && h.unadopted_ends() == 3 && h.double_ends() == 0 && h.foreign_ends() == 0,
          "three ends of an unadopted encoder are three recorded faults, never a success and never an "
          "endEncoding; the count does not collapse to one");
    [e endEncoding]; h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 3d. An encoder owned by ANOTHER holder. Ending it here would truncate that holder's encoding and
  //     steal its close obligation, so it must be refused - and the real owner must still close it.
  { nr_exp::ScopeHolder owner, other;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    owner.adopt(e);
    nr_exp::set_current_scope_holder(&other);
    const bool ok = nr_exp::nr_scope_end(e);            // the wrong holder tries to end it
    const bool classified = !ok && other.foreign_ends() == 1 && other.unadopted_ends() == 0
                            && other.double_ends() == 0 && other.adoption_failed()
                            && other.encoder_state_indeterminate();
    nr_exp::set_current_scope_holder(&owner);
    const bool owner_closed = nr_exp::nr_scope_end(e);  // the real owner can still close it exactly once
    check("scope-encoder-owned-by-another-holder-is-refused-not-ended",
          classified && owner_closed && owner.ended() == 1 && !owner.adoption_failed(),
          "a foreign encoder is classified as such, not as unadopted, is NOT ended, and its real owner "
          "still closes it cleanly afterwards");
    owner.close_all(); other.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 4. The checked-only ordering: compute encoder, end it, then a snapshot blit on the same buffer.
  //    This is what the checked lane does per digest and what the body's #else factory now adopts.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    bool all_ok = true;
    for (int i = 0; i < 4; ++i) {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; h.adopt(e);
      all_ok = nr_exp::nr_scope_end(e) && all_ok;
      id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];       // the snapshot copy
      if (!b) { all_ok = false; break; }
      h.adopt(b);
      all_ok = nr_exp::nr_scope_end(b) && all_ok;
    }
    check("scope-checked-snapshot-blit-ordering",
          all_ok && !h.any_failure() && h.ended() == 8,
          "four compute/snapshot-blit pairs on one command buffer close cleanly, the ordering the checked "
          "lane needs and that the unadopted factory broke");
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 5. Committability is PROVEN, not asserted, and only for the case the holder can account for.
  //    5a: a fully accounted-for frame commits and completes.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; h.adopt(e);
    const bool ok = nr_exp::nr_scope_end(e);
    h.close_all();
    [cb commit]; [cb waitUntilCompleted];
    check("scope-accounted-for-buffer-actually-commits",
          ok && !h.encoder_state_indeterminate() && cb.status == MTLCommandBufferStatusCompleted,
          "when every encoder is accounted for the buffer really does reach Completed - the committable "
          "claim is backed by a commit, not by wording");
    nr_exp::set_current_scope_holder(nullptr); }
  //    5b: after an unaccounted-for encoder, the holder must SAY the state is indeterminate rather than
  //        claim committability. The buffer only commits once that encoder is actually ended, by whoever
  //        owns it - which is precisely what this runtime refuses to guess at.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];   // not adopted: provokes the fault
    (void)nr_exp::nr_scope_end(e);
    const bool says_indeterminate = h.adoption_failed() && h.encoder_state_indeterminate();
    h.close_all();
    [e endEncoding];                        // the OWNER ends it; the holder never did
    [cb commit]; [cb waitUntilCompleted];
    check("scope-indeterminate-state-is-reported-not-papered-over",
          says_indeterminate && cb.status == MTLCommandBufferStatusCompleted,
          "the holder reports the encoder state as indeterminate instead of claiming committability, and "
          "the buffer completes only after the encoder's real owner ends it");
    nr_exp::set_current_scope_holder(nullptr); }

  // 6. An exception between adopt and end still closes the encoder, and the failure stays visible.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    bool threw = false;
    try {
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder]; h.adopt(e);
      throw std::runtime_error("stage failure with an encoder open");
    } catch (const std::exception&) { threw = true; }
    const bool closed = h.close_all();
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];         // only opens if close_all really ended it
    check("scope-exception-with-an-encoder-open-still-closes-it",
          threw && closed && b != nil && h.started() == 1,
          "close_all after a thrown stage ends the open encoder; started() still counts it so the ticket "
          "stays discard-only");
    h.adopt(b); nr_exp::nr_scope_end(b); h.close_all();
    nr_exp::set_current_scope_holder(nullptr); }

  // 7. LIFETIME. EncoderScope::close() releases the strong reference, so a closed encoder can be
  //    deallocated and a later encoder can land on the same address. Identity must therefore die with the
  //    object, or a stale entry would classify a brand-new (possibly OPEN) encoder as an accounted-for
  //    double end - and the runtime would then claim the buffer is committable when it cannot know that.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    std::set<uintptr_t> seen;
    uintptr_t reused = 0;
    bool misclassified = false;
    for (int i = 0; i < 400 && !reused; ++i) {
      @autoreleasepool {
        id<MTLCommandBuffer> cb = [q commandBuffer];
        id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
        const uintptr_t addr = (uintptr_t)(__bridge void*)e;
        if (seen.count(addr)) {
          // A recycled address that this holder has NOT adopted. It must read as never-adopted.
          reused = addr;
          const bool ok = nr_exp::nr_scope_end(e);
          const nr_exp::ScopeHolder* ch = &h;
          (void)ch;
          misclassified = ok || h.double_ends() != 0 || h.foreign_ends() != 0;
          [e endEncoding];
        } else {
          seen.insert(addr);
          h.adopt(e);
          nr_exp::nr_scope_end(e);          // closed: the scope drops its strong reference here
        }
      }
    }
    check("scope-recycled-address-is-not-a-stale-identity",
          !misclassified && h.double_ends() == 0 && h.foreign_ends() == 0,
          reused ? "an encoder allocated at a previously-adopted, now-released address reads as never "
                   "adopted, not as our already-closed encoder"
                 : "no address reuse was observed in 400 encoders, so this run did not exercise the "
                   "hazard; the token invariant below is what actually rules it out");
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 7b. The invariant itself, checked directly and independently of whether reuse happened to occur.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> fresh = [cb computeCommandEncoder];
    const uint64_t before = nr_exp::detail::token_of(fresh);
    h.adopt(fresh);
    const uint64_t after = nr_exp::detail::token_of(fresh);
    nr_exp::nr_scope_end(fresh);
    id<MTLBlitCommandEncoder> other = [cb blitCommandEncoder];
    const uint64_t other_tok = nr_exp::detail::token_of(other);
    h.adopt(other);
    check("scope-identity-token-is-per-object-and-unique",
          before == 0 && after != 0 && other_tok == 0
          && nr_exp::detail::token_of(other) != after,
          "an un-adopted encoder carries no token, adoption attaches one, and two encoders never share a "
          "token - so identity cannot outlive the object it names");
    nr_exp::nr_scope_end(other); h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 8. DUPLICATE ADOPTION. Two scopes over one encoder are two close obligations; the second close would
  //    call endEncoding on an already-ended encoder. The duplicate must be refused and recorded.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    h.adopt(e);
    h.adopt(e);                              // duplicate: must not create a second close obligation
    const bool ok = nr_exp::nr_scope_end(e);
    h.close_all();                           // if a second scope existed this is the second endEncoding
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    check("scope-duplicate-adoption-is-refused-not-stacked",
          h.duplicate_adoptions() == 1 && ok && h.ended() == 1 && b != nil,
          "the second adopt is recorded as a fault and adds no scope, so the encoder is ended exactly "
          "once and the command buffer survives close_all");
    h.adopt(b); nr_exp::nr_scope_end(b); h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 8a2. THE LIVE-PATH REGRESSION. Metal reuses and coalesces encoder OBJECTS: a later
  //      computeCommandEncoder call can hand back an object that was already adopted and closed, with its
  //      adoption token still attached. Refusing on the token alone dropped that encoder's scope and left
  //      it open, which asserted on the next encoder with
  //      `A command encoder is already encoding to this command buffer` on the real frame path.
  //      A token whose entry is CLOSED must read as a fresh adoption. This is observed, not simulated:
  //      an already-ended encoder cannot be re-ended by hand, so the reuse has to come from Metal.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    std::set<uintptr_t> closed_objects;
    bool observed_reuse = false, handled = true;
    for (int i = 0; i < 400 && !observed_reuse; ++i) {
      id<MTLCommandBuffer> cb = [q commandBuffer];
      id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
      const uintptr_t addr = (uintptr_t)(__bridge void*)e;
      const bool carries_token = nr_exp::detail::token_of(e) != 0;
      if (closed_objects.count(addr) && carries_token) {
        observed_reuse = true;                       // Metal handed back a live object we already closed
        const std::size_t dup_before = h.duplicate_adoptions();
        h.adopt(e);
        const bool accepted = h.duplicate_adoptions() == dup_before;
        const bool ended = nr_exp::nr_scope_end(e);
        handled = accepted && ended;
      } else {
        h.adopt(e);
        nr_exp::nr_scope_end(e);
        closed_objects.insert(addr);
      }
    }
    check("scope-recycled-object-with-a-closed-token-is-adoptable-again",
          handled && !h.adoption_failed(),
          observed_reuse ? "Metal handed back an object that was already adopted and closed; it is "
                           "accepted as a fresh adoption and ends cleanly, instead of being refused as a "
                           "duplicate and left open"
                         : "no object reuse was observed in 400 encoders, so this run did not exercise "
                           "the hazard; the live-frame check on the real path is what covers it");
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 8a3. SKIPPED ADOPTION ON A REACTIVATED OBJECT. This is the hole a closed token cannot cover. If Metal
  //      hands the same object back for a NEW encoding period and that adoption is MISSED, the holder sees
  //      precisely what it sees for a genuine second end: no live scope, a closed token of its own. It
  //      cannot tell the two apart, so it must not treat either as accounted for. The sequence below IS
  //      that view - adopt, end, then end again with no intervening adoption - and the requirement is that
  //      it never permits a committability claim.
  { nr_exp::ScopeHolder h; nr_exp::set_current_scope_holder(&h);
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    h.adopt(e);
    nr_exp::nr_scope_end(e);                 // period 1 closed by us; the token survives on the object
    const bool ok = nr_exp::nr_scope_end(e); // period 2's adoption was missed - identical view to a double end
    check("scope-skipped-adoption-on-a-reactivated-object-is-indeterminate",
          !ok && h.double_ends() == 1 && h.non_live_ends() == 1 && h.adoption_failed()
          && h.encoder_state_indeterminate(),
          "a closed token from an EARLIER encoding period is not evidence about the current one, so this "
          "is a fault AND indeterminate - it must never be reported as an accounted-for double end that "
          "lets the caller be told the buffer is committable");
    h.close_all(); nr_exp::set_current_scope_holder(nullptr); }

  // 8b. Duplicate adoption ACROSS holders: the first adopter keeps the obligation.
  { nr_exp::ScopeHolder first, second;
    id<MTLCommandBuffer> cb = [q commandBuffer];
    id<MTLComputeCommandEncoder> e = [cb computeCommandEncoder];
    first.adopt(e);
    second.adopt(e);                         // refused
    nr_exp::set_current_scope_holder(&first);
    const bool ok = nr_exp::nr_scope_end(e);
    second.close_all(); first.close_all();
    id<MTLBlitCommandEncoder> b = [cb blitCommandEncoder];
    check("scope-duplicate-adoption-across-holders-keeps-one-owner",
          second.duplicate_adoptions() == 1 && second.ended() == 0 && ok && first.ended() == 1 && b != nil,
          "a second holder cannot take over an already-adopted encoder; it records the duplicate and the "
          "first owner still closes it exactly once");
    [b endEncoding];                         // this test opened it; leaving it open trips Metal's own
                                             // "released without endEncoding" assertion on dealloc
    nr_exp::set_current_scope_holder(nullptr); }

  printf(failures ? "FAILURES in scope tests: %d\n" : "ALL scope tests PASS (%d failures)\n", failures);
  return failures ? 1 : 0;
} }
