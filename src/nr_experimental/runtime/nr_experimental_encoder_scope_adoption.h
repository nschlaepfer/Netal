#pragma once
// Adoption of Codex's shared EncoderScope helper into this runtime (Opus-owned adoption; the helper itself is
// Codex-owned and is copied into the build snapshot unmodified).
//
// The helper wants one FrameEncoderState per encode INVOCATION and one scope constructed immediately after each
// raw encoder is obtained, with scope.close() replacing raw endEncoding. The body creates ~200 encoders through
// two factory lambdas and ends them at ~200 separate sites, so this holder bridges the two: the factory pushes a
// scope the moment the encoder exists, and `nr_scope_end(encoder)` closes exactly that scope instead of calling
// endEncoding directly. Nothing is global: the holder is per invocation, and the previous process-global
// `g_open_encoder` tracker it replaces is gone.
#import <Metal/Metal.h>
#import <objc/runtime.h>
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>
#include "nr_encoder_scope.h"

namespace nr_exp {

namespace detail {
// Which holder adopted which encoder, and whether that owner has already closed it, so `end()` can tell
// "never adopted by anyone" from "adopted by ANOTHER holder" - two faults with different correct answers.
//
// IDENTITY IS A TOKEN, NOT AN ADDRESS. An earlier version keyed on the encoder's pointer and argued that
// an adopted encoder stays alive until its holder is destroyed. That argument was WRONG:
// EncoderScope::close() sets encoder_ = nil and releases the only strong reference, so the object can be
// deallocated the moment it is closed and a LATER encoder can be allocated at the same address. A stale
// entry would then classify a brand-new, never-adopted (and therefore possibly OPEN) encoder as "ours,
// already closed" - an accounted-for double end - and the runtime would go on to claim the command buffer
// is committable when it cannot know that.
//
// So each adopted encoder carries a unique token as an ObjC associated object. The association dies with
// the object, so a recycled address has NO token and is correctly seen as never adopted. Nothing here
// retains an encoder or extends its lifetime; the strong reference stays exactly where Codex's helper
// puts it.
inline const void* token_key() { static const char k = 0; return &k; }
inline uint64_t next_token() { static std::atomic<uint64_t> n{1}; return n.fetch_add(1, std::memory_order_relaxed); }
// 0 means "this object carries no adoption token".
inline uint64_t token_of(id<MTLCommandEncoder> e) {
  if (!e) return 0;
  id v = objc_getAssociatedObject(e, token_key());
  return v ? [(NSNumber*)v unsignedLongLongValue] : 0;
}
inline uint64_t attach_token(id<MTLCommandEncoder> e) {
  const uint64_t t = next_token();
  objc_setAssociatedObject(e, token_key(), @(t), OBJC_ASSOCIATION_RETAIN_NONATOMIC);
  return t;
}
struct AdoptionRegistry {
  struct Entry { uint64_t token; const void* owner; bool closed; };
  std::mutex m;
  std::vector<Entry> entries;
};
inline AdoptionRegistry& registry() { static AdoptionRegistry r; return r; }
struct Ownership { bool known; bool mine; bool closed; };
inline void registry_add(uint64_t token, const void* owner) {
  AdoptionRegistry& r = registry(); std::lock_guard<std::mutex> g(r.m);
  r.entries.push_back({token, owner, false});
}
inline void registry_mark_closed(uint64_t token, const void* owner) {
  if (!token) return;
  AdoptionRegistry& r = registry(); std::lock_guard<std::mutex> g(r.m);
  for (auto it = r.entries.rbegin(); it != r.entries.rend(); ++it)
    if (it->token == token && it->owner == owner) { it->closed = true; return; }
}
inline Ownership registry_lookup(uint64_t token, const void* self) {
  if (!token) return {false, false, false};
  AdoptionRegistry& r = registry(); std::lock_guard<std::mutex> g(r.m);
  for (auto it = r.entries.rbegin(); it != r.entries.rend(); ++it)
    if (it->token == token) return {true, it->owner == self, it->closed};
  return {false, false, false};
}
inline void registry_drop_owner(const void* owner) {
  AdoptionRegistry& r = registry(); std::lock_guard<std::mutex> g(r.m);
  r.entries.erase(std::remove_if(r.entries.begin(), r.entries.end(),
                                 [owner](const AdoptionRegistry::Entry& x) { return x.owner == owner; }),
                  r.entries.end());
}
} // namespace detail

class ScopeHolder {
  netal_nr_experimental::FrameEncoderState state_;
  std::vector<netal_nr_experimental::EncoderScope> scopes_;
  std::size_t ended_ = 0, unadopted_ends_ = 0, double_ends_ = 0, foreign_ends_ = 0, duplicate_adoptions_ = 0;
  bool failed_ = false;
public:
  ScopeHolder() { scopes_.reserve(512); }
  ScopeHolder(const ScopeHolder&) = delete;
  ScopeHolder& operator=(const ScopeHolder&) = delete;

  // Immediately after the raw encoder is obtained, before any throwing bookkeeping.
  ~ScopeHolder() { detail::registry_drop_owner(this); }
  // A DUPLICATE adoption is refused, not stacked. Two scopes over one encoder are two close obligations,
  // and the second close calls endEncoding on an already-ended encoder - the Metal API error this layer
  // exists to prevent. The first adopter keeps the obligation; the duplicate is recorded as a fault.
  void adopt(id<MTLCommandEncoder> e) {
    if (!e) { scopes_.emplace_back(state_, e); return; }
    // A surviving token does NOT mean "currently adopted". Metal reuses and coalesces encoder OBJECTS, so
    // the same object - with its association intact - can come back from a later
    // computeCommandEncoder call. Refusing on the token alone dropped the scope of a live encoder and
    // left it open, which asserted on the next encoder as
    //   `A command encoder is already encoding to this command buffer`.
    // Refuse only when the existing entry is still OPEN: that is a genuine duplicate adoption of a live
    // encoder, and a second scope over it would be a second close obligation. A token whose entry is
    // already CLOSED belongs to a finished adoption of a recycled object, so this is a fresh adoption and
    // gets a fresh identity.
    const uint64_t existing = detail::token_of(e);
    if (existing != 0) {
      const detail::Ownership own = detail::registry_lookup(existing, this);
      if (own.known && !own.closed) { ++duplicate_adoptions_; failed_ = true; return; }
    }
    scopes_.emplace_back(state_, e);
    detail::registry_add(detail::attach_token(e), this);
  }
  // Closes the scope owning `e`. Returns false if the underlying end failed, if `e` was never adopted,
  // or if `e` was already ended through this holder - none of those may be reported as success.
  //
  // The previous version returned true for an unknown encoder ("nothing to close"). That is how a
  // checked-only creation site that never adopted went unnoticed: nr_scope_end reported success and left
  // the encoder OPEN, and the next blit died on
  //   `A command encoder is already encoding to this command buffer`.
  // An unknown encoder is now ENDED here - leaving it open is what corrupts the command buffer - and the
  // fault is recorded so the runtime can refuse to publish the frame. Ending it keeps the buffer in a
  // committable state, which the caller's discard-only contract still requires on failure.
  bool end(id<MTLCommandEncoder> e) {
    if (!e) return true;
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it)
      if (it->get() == e) {                 // live scope: exactly one close, through the scope that owns it
        ++ended_;
        const bool ok = it->close();
        if (!ok) failed_ = true;
        detail::registry_mark_closed(detail::token_of(e), this);
        return ok;
      }
    // Not live here. NONE of the remaining cases may call endEncoding. Unknown ownership is NOT a licence
    // to end: an encoder we do not own may be another holder's live encoder (ending it truncates their
    // encoding) or may already have been ended (ending it again is a Metal API error). An earlier version
    // of this function ended the unknown case "so it is not left open"; that traded a detectable fault for
    // an undetectable one, and it is exactly the assumption this class must not make.
    const detail::Ownership own = detail::registry_lookup(detail::token_of(e), this);
    // "ours, already closed" is only what we can SEE. It cannot distinguish a genuine second end from a
    // reactivated object whose new encoding period we failed to adopt, so it is a fault and it is
    // indeterminate - never an accounted-for case that permits a committability claim.
    if (own.known && own.mine)  { ++double_ends_;   failed_ = true; return false; }
    if (own.known && !own.mine) { ++foreign_ends_;  failed_ = true; return false; }   // another holder owns the close
    ++unadopted_ends_; failed_ = true;                                                 // state unknown: do not touch it
    return false;
  }
  // Closes everything still open, in reverse order. Safe to call more than once.
  bool close_all() {
    bool ok = true;
    for (auto it = scopes_.rbegin(); it != scopes_.rend(); ++it) {
      id<MTLCommandEncoder> e = it->get();
      ok = it->close() && ok;
      if (e) detail::registry_mark_closed(detail::token_of(e), this);
    }
    return ok;
  }
  std::size_t started() const { return state_.started; }      // conservative: counted at creation, not at end
  bool cleanup_failed() const { return state_.cleanup_failed; }
  // Adoption faults, separate from Metal's own cleanup failure. Any of these means an encoder was NOT
  // handled the way the contract requires, and the frame must not be published on the strength of it.
  std::size_t ended() const { return ended_; }
  std::size_t unadopted_ends() const { return unadopted_ends_; }
  std::size_t double_ends() const { return double_ends_; }
  std::size_t foreign_ends() const { return foreign_ends_; }
  std::size_t duplicate_adoptions() const { return duplicate_adoptions_; }
  // True when a fault left an encoder this holder could not account for. The command buffer may still have
  // an OPEN encoder on it, and no caller may be told it is committable on this holder's word alone.
  // EVERY end that did not go through a live scope leaves the encoder's CURRENT state unknown, and that
  // includes a "double end". A closed registry entry only records that an EARLIER encoding period on that
  // object was closed by us. Metal reactivates encoder objects, so if a new encoding period began and its
  // adoption was MISSED, this holder sees exactly the same thing - no live scope, a closed token of its
  // own - and would call it an accounted-for double end. It is not accounted for: the encoder may be open
  // right now. An old closed token cannot prove current committability, so no non-live end may claim it.
  bool encoder_state_indeterminate() const {
    return unadopted_ends_ > 0 || foreign_ends_ > 0 || double_ends_ > 0;
  }
  std::size_t non_live_ends() const { return unadopted_ends_ + foreign_ends_ + double_ends_; }
  bool adoption_failed() const { return failed_; }
  bool any_failure() const { return failed_ || state_.cleanup_failed; }
  // FrameEncoderState is deliberately non-copyable and non-assignable, so a holder is not "reset":
  // each encode invocation constructs a fresh one. This is left out rather than faked.
};

// The holder for the encode invocation currently on this thread. Set by the runtime around encode(); the body,
// block 0 and the encoder module route their encoders through it.
ScopeHolder* current_scope_holder();
void set_current_scope_holder(ScopeHolder* h);

// Convenience used at the ~200 body end sites in place of `[enc endEncoding]`.
// Returns false when the end did not happen the way the contract requires. Call sites keep the plain
// `nr_exp::nr_scope_end(x);` form - the failure is ALSO recorded in the holder, which the runtime checks
// at the encode boundary, because ~200 call sites cannot each carry an error path without changing code
// the accepted v17 body must still reduce to.
inline bool nr_scope_end(id<MTLCommandEncoder> e) {
  if (!e) return true;
  if (ScopeHolder* h = current_scope_holder()) return h->end(e);
  [e endEncoding];                          // no holder installed (never on the runtime path)
  return true;
}
inline void nr_scope_adopt(id<MTLCommandEncoder> e) {
  if (ScopeHolder* h = current_scope_holder()) h->adopt(e);
}
} // namespace nr_exp
