#pragma once

// CPU-only ownership model. It does not inspect Metal, certify a recording, or prove GPU drain.
#include <array>
#include <cstdint>
#include <map>
#include <limits>
#include <iterator>
#include <vector>

namespace netal::sr {
struct FrameToken {
  std::uint64_t session = 0, epoch = 0, sequence = 0, generation = 0;
  std::uint64_t slot = 0, frame = 0;
  bool operator==(const FrameToken&) const = default;
};
enum class FrameOwnership { Free, Reserved, Encoding, CoreEncoded, Encoded, FailedUnsubmitted, SubmitIssued, SubmitUnknown, Quarantined };
enum class FrameResult { Accepted, Unavailable, Backpressure, NeedsReset, NeedsDrain,
                         WrongAssociation, WrongState, WrongOrder };

class FrameLedger {
 public:
  static constexpr std::size_t capacity = 6;
  explicit FrameLedger(std::uint64_t identity) : identity_(identity) {}
  struct Submission { std::uint64_t identity = 0; std::vector<FrameToken> tokens; };

  FrameResult reserve(std::uintptr_t command, std::uintptr_t queue, bool reset, FrameToken& token) {
    if (!command || !queue || (queue_ && queue_ != queue && (!drained() || !reset))) return FrameResult::WrongAssociation;
    for (const auto& slot : slots_) {
      if (slot.state != FrameOwnership::Free && slot.command == command)
        return FrameResult::WrongAssociation; // a raw CB cannot stand for two outstanding recordings
      if ((slot.state == FrameOwnership::Reserved || slot.state == FrameOwnership::Encoding)) return FrameResult::Backpressure;
    }
    if (reset && !drained()) return FrameResult::NeedsDrain;
    if (!reset && (cut_ || !initialized_)) return FrameResult::NeedsReset;
    std::size_t index = 0;
    while (index < capacity && slots_[index].state != FrameOwnership::Free) ++index;
    if (index == capacity) return FrameResult::Backpressure;
    if (sequence_ == UINT64_MAX || slots_[index].generation == UINT64_MAX ||
        next_frame_ == UINT64_MAX || (reset && epoch_ == UINT64_MAX)) return FrameResult::Unavailable;
    if (reset) { ++epoch_; next_frame_ = 0; initialized_ = false; cut_ = false; }
    queue_ = queue;
    auto& slot = slots_[index];
    token = {identity_, epoch_, ++sequence_, ++slot.generation, index, next_frame_};
    slot.token = token; slot.command = command; slot.batch = 0; slot.state = FrameOwnership::Reserved;
    return FrameResult::Accepted;
  }
  bool reserved(const FrameToken& token, std::uintptr_t command) const {
    const Slot* slot = find(token);
    return slot && slot->state == FrameOwnership::Reserved && slot->command == command && !cut_;
  }
  bool encoding(const FrameToken& token) const {
    const Slot* slot = find(token); return slot && slot->state == FrameOwnership::Encoding && !cut_;
  }
  FrameResult beginEncoding(const FrameToken& token) {
    Slot* slot = find(token);
    if (!slot) return FrameResult::WrongAssociation;
    if (slot->state != FrameOwnership::Reserved || cut_) return FrameResult::WrongState;
    slot->state = FrameOwnership::Encoding; return FrameResult::Accepted;
  }
  FrameResult finalize(const FrameToken& token) {
    Slot* slot = find(token);
    if (!slot) return FrameResult::WrongAssociation;
    if (slot->state != FrameOwnership::CoreEncoded || cut_) return FrameResult::WrongState;
    slot->state = FrameOwnership::Encoded; return FrameResult::Accepted;
  }
  FrameResult encoded(const FrameToken& token) {
    Slot* slot = find(token);
    if (!slot) return FrameResult::WrongAssociation;
    if (slot->state != FrameOwnership::Encoding || cut_) return FrameResult::WrongState;
    slot->state = FrameOwnership::CoreEncoded; initialized_ = true; next_frame_ = token.frame + 1;
    return FrameResult::Accepted; // scheduled parity changes here, never in feedback
  }
  FrameResult failEncoding(const FrameToken& token) {
    Slot* slot = find(token);
    if (!slot) return FrameResult::WrongAssociation;
    if (slot->state != FrameOwnership::Reserved && slot->state != FrameOwnership::Encoding &&
        slot->state != FrameOwnership::CoreEncoded) return FrameResult::WrongState;
    slot->state = FrameOwnership::FailedUnsubmitted; cut_ = true;
    return FrameResult::Accepted;
  }
  FrameResult discard(const FrameToken& token) {
    Slot* slot = find(token);
    if (!slot) return FrameResult::WrongAssociation;
    if (slot->state != FrameOwnership::Reserved && slot->state != FrameOwnership::Encoded &&
        slot->state != FrameOwnership::FailedUnsubmitted) return FrameResult::WrongState;
    cut_ = true; release(*slot);
    return FrameResult::Accepted;
  }
  FrameResult issue(const std::vector<FrameToken>& tokens, std::uintptr_t queue, Submission& batch) {
    if (queue != queue_ || tokens.empty()) return FrameResult::WrongAssociation;
    if (cut_) return FrameResult::NeedsReset;
    std::uint64_t previous = 0;
    for (const auto& token : tokens) {
      const Slot* slot = find(token);
      if (!slot) return FrameResult::WrongAssociation;
      if (slot->state != FrameOwnership::Encoded) return FrameResult::WrongState;
      if (token.sequence <= previous) return FrameResult::WrongOrder;
      previous = token.sequence;
    }
    // Every earlier dependency must already be issued or present earlier in this exact batch.
    for (const auto& slot : slots_) {
      if ((slot.state == FrameOwnership::Reserved || slot.state == FrameOwnership::Encoding ||
           slot.state == FrameOwnership::CoreEncoded || slot.state == FrameOwnership::Encoded ||
           slot.state == FrameOwnership::FailedUnsubmitted) && slot.token.sequence < previous) {
        bool found = false;
        for (const auto& token : tokens) if (token == slot.token) found = true;
        if (!found) return FrameResult::WrongOrder;
      }
    }
    if (batch_sequence_ == UINT64_MAX) return FrameResult::Unavailable;
    batch = {++batch_sequence_, tokens};
    for (const auto& token : tokens) { Slot* slot = find(token); slot->batch = batch.identity;
      slot->state = FrameOwnership::SubmitIssued; }
    return FrameResult::Accepted; // persist BEFORE calling the external void commit API
  }
  FrameResult feedback(const Submission& batch, bool nil_error, const std::vector<bool>& exact_markers) {
    if (batch.tokens.empty() || batch.tokens.size() != exact_markers.size()) return FrameResult::WrongAssociation;
    for (const auto& token : batch.tokens) {
      const Slot* slot = find(token);
      if (!slot || slot->batch != batch.identity || (slot->state != FrameOwnership::SubmitIssued && slot->state != FrameOwnership::SubmitUnknown))
        return FrameResult::WrongAssociation;
    }
    // Partial marker validation must not free any ownership in a suspect commit batch.
    bool verified = nil_error;
    for (bool marker : exact_markers) verified = verified && marker;
    for (const auto& token : batch.tokens) {
      Slot* slot = find(token);
      if (verified) release(*slot);
      else slot->state = FrameOwnership::Quarantined;
    }
    if (!verified) cut_ = true; else completed_ += batch.tokens.size();
    return FrameResult::Accepted;
  }
  void ambiguous(const Submission& batch) {
    for (const auto& token : batch.tokens) {
      Slot* slot = find(token);
      if (slot && slot->batch == batch.identity && slot->state == FrameOwnership::SubmitIssued) {
        slot->state = FrameOwnership::SubmitUnknown; cut_ = true;
      }
    } // feedback-before-return may already have settled everything: never overwrite it
  }
  void cut() { cut_ = true; }
  bool drained() const { for (const auto& slot : slots_) if (slot.state != FrameOwnership::Free) return false; return true; }
  bool needsReset() const { return cut_ || !initialized_; }
  std::uint64_t scheduledFrames() const { return next_frame_; }
  std::uint64_t completedFrames() const { return completed_; }
  std::uint64_t retiredPrefix() const { return retired_prefix_; }
  std::size_t retirementRangeCount() const { return retired_.size(); }
  FrameOwnership ownership(const FrameToken& token) const {
    const Slot* slot = find(token); return slot ? slot->state : FrameOwnership::Free;
  }
 private:
  struct Slot { FrameToken token; FrameOwnership state = FrameOwnership::Free;
                std::uint64_t generation = 0, batch = 0; std::uintptr_t command = 0; };
  Slot* find(const FrameToken& token) {
    if (token.session != identity_ || token.slot >= capacity) return nullptr;
    Slot& slot = slots_[token.slot];
    return slot.state != FrameOwnership::Free && slot.token == token ? &slot : nullptr;
  }
  const Slot* find(const FrameToken& token) const { return const_cast<FrameLedger*>(this)->find(token); }
  void release(Slot& slot) {
    // Merge ranges so a missing first callback cannot grow one set entry per later frame.
    // There are at most capacity outstanding gaps, regardless of how many successors retire.
    std::uint64_t begin = slot.token.sequence, end = begin;
    auto next = retired_.lower_bound(begin);
    if (next != retired_.end() && end != UINT64_MAX && next->first == end + 1) {
      end = next->second; next = retired_.erase(next);
    }
    if (next != retired_.begin()) {
      auto previous = std::prev(next);
      if (previous->second != UINT64_MAX && previous->second + 1 == begin) {
        begin = previous->first; retired_.erase(previous);
      }
    }
    retired_[begin] = end;
    auto first = retired_.begin();
    if (retired_prefix_ != UINT64_MAX && first->first == retired_prefix_ + 1) {
      retired_prefix_ = first->second; retired_.erase(first);
    }
    slot.state = FrameOwnership::Free; slot.command = 0; slot.batch = 0;
  }
  std::array<Slot, capacity> slots_{};
  std::map<std::uint64_t, std::uint64_t> retired_;
  std::uint64_t identity_, epoch_ = 0, sequence_ = 0, batch_sequence_ = 0;
  std::uint64_t next_frame_ = 0, completed_ = 0, retired_prefix_ = 0;
  std::uintptr_t queue_ = 0;
  bool initialized_ = false, cut_ = false;
};
} // namespace netal::sr
