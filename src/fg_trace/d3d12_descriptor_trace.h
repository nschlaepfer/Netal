#pragma once

#include <cstdint>

namespace neuralstudio::fg_trace {

struct DescriptorTraceRequestSnapshot {
  std::uint64_t descriptor_token = 0;
  std::uint64_t sampler_descriptor_token = 0;
  bool descriptor_kind_matches = false;
  bool sampler_kind_matches = false;
  bool capture_enabled = false;
};

// Mirrors the normalized phase stream into the optional D3D12 descriptor
// capture without exposing the proxy's internal phase storage.
void DescriptorTraceSetPhase(const char* phase);

// Snapshot before the real NvAPI call because the driver may rewrite the CPU
// descriptor slot while materializing its CUDA texture/surface object.
DescriptorTraceRequestSnapshot DescriptorTraceSnapshotResourceObject(
    const char* resource_kind, std::uintptr_t descriptor_a,
    std::uintptr_t descriptor_b, bool has_descriptor_b);

// Bind the completed NvAPI request to the immutable pre-call snapshot. Raw
// descriptor handles remain confined to the descriptor capture's volatile
// sidecar.
bool DescriptorTraceRecordResourceObject(
    const char* call, const char* phase, const char* resource_kind,
    std::uint64_t nvapi_resource_token, std::int32_t status,
    std::uintptr_t descriptor_a, std::uintptr_t descriptor_b,
    bool has_descriptor_b, const DescriptorTraceRequestSnapshot& snapshot);

}  // namespace neuralstudio::fg_trace
