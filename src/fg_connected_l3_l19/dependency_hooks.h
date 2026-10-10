#pragma once

#include <array>
#include <cstddef>
#include <cstdint>

namespace fg_connected_l3_l19 {

enum class Phase : std::uint32_t { kReset = 0, kGenerated = 1 };

struct HookReceipt {
  std::uint32_t first_launch = 0;
  std::uint32_t last_launch = 0;
  std::array<char, 65> checkpoint_sha256{};
  std::size_t input_mismatched_bytes = 0;
  std::size_t canary_mismatched_bytes = 0;
};

// The backend context owns all live resources. Hook implementations must bind
// those resources directly; a hook may not read a checkpoint from disk or
// accept an expected/archive tensor pointer.
using L6L16Hook = int (*)(void* backend_context, Phase phase,
                          HookReceipt* receipt);
using L19Hook = int (*)(void* backend_context, HookReceipt* receipt);

struct DependencyHooks {
  L6L16Hook l6_l16 = nullptr;
  L19Hook l19 = nullptr;

  [[nodiscard]] bool complete() const {
    return l6_l16 != nullptr && l19 != nullptr;
  }
};

inline int FailClosedIfIncomplete(const DependencyHooks& hooks) {
  return hooks.complete() ? 0 : 4;
}

}  // namespace fg_connected_l3_l19
