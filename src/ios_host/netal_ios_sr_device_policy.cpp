#include "netal_ios_sr_device_policy.h"

#include <string>

namespace netal::ios_sr {
namespace {
bool IsASCIIWhitespace(unsigned char c) {
  return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v';
}

std::string NormalizeGPUName(std::string_view name) {
  std::string normalized;
  bool pending_space = false;
  for (unsigned char c : name) {
    if (c >= 0x80 || c == '\0') return {};  // Unknown formats fail closed.
    if (IsASCIIWhitespace(c)) {
      pending_space = !normalized.empty();
      continue;
    }
    if (pending_space) normalized.push_back(' ');
    pending_space = false;
    normalized.push_back(c >= 'A' && c <= 'Z' ? static_cast<char>(c + ('a' - 'A'))
                                             : static_cast<char>(c));
  }
  return normalized;
}
}  // namespace

Chip ClassifyChipName(std::string_view gpu_name) {
  const std::string normalized = NormalizeGPUName(gpu_name);
  std::string_view name(normalized);
  constexpr std::string_view prefix = "apple ";
  constexpr std::string_view suffix = " gpu";
  if (name.substr(0, prefix.size()) == prefix) name.remove_prefix(prefix.size());
  if (name.size() >= suffix.size() && name.substr(name.size() - suffix.size()) == suffix) {
    name.remove_suffix(suffix.size());
  }
  if (name == "a19 pro") return Chip::A19Pro;
  if (name == "a20 pro") return Chip::A20Pro;
  return Chip::Unknown;
}

std::string_view ChipName(Chip chip) {
  switch (chip) {
    case Chip::A19Pro: return "A19 Pro";
    case Chip::A20Pro: return "A20 Pro";
    case Chip::Unknown: return "Unknown";
  }
  return "Unknown";
}

std::string_view AdmissionReasonText(AdmissionReason reason) {
  switch (reason) {
    case AdmissionReason::EligibleCandidate:
      return "Eligible mobile hardware candidate; model correctness and performance are unvalidated.";
    case AdmissionReason::UnsupportedPlatform:
      return "Mobile SR requires a physical iOS or iPadOS device; simulators and Catalyst are excluded.";
    case AdmissionReason::NoMetalDevice:
      return "A Metal device is unavailable.";
    case AdmissionReason::UnsupportedChip:
      return "Mobile SR requires an identified A19 Pro or A20 Pro GPU; unknown and other chips are excluded.";
    case AdmissionReason::UnsupportedOS:
      return "Mobile SR requires iOS or iPadOS 26 or later.";
    case AdmissionReason::Metal4Unavailable:
      return "Metal 4 support is unavailable.";
    case AdmissionReason::NeuralAcceleratorFamilyUnavailable:
      return "The GPU does not report the Apple10 neural-accelerator capability required by this path.";
    case AdmissionReason::TensorUnavailable:
      return "A required FP16 Metal tensor could not be allocated.";
  }
  return "Unsupported mobile SR configuration.";
}

Admission EvaluateDevice(const DeviceFacts& facts) {
  Admission result;
  result.chip = ClassifyChipName(facts.gpu_name);
  result.neural_accelerator_candidate =
      facts.platform == Platform::PhysicalIOS && facts.metal_device_available &&
      result.chip != Chip::Unknown && facts.supports_apple10;
  if (facts.platform != Platform::PhysicalIOS) {
    result.reason = AdmissionReason::UnsupportedPlatform;
  } else if (!facts.metal_device_available) {
    result.reason = AdmissionReason::NoMetalDevice;
  } else if (result.chip == Chip::Unknown) {
    result.reason = AdmissionReason::UnsupportedChip;
  } else if (facts.os_major < 26) {
    result.reason = AdmissionReason::UnsupportedOS;
  } else if (!facts.supports_metal4) {
    result.reason = AdmissionReason::Metal4Unavailable;
  } else if (!facts.supports_apple10) {
    result.reason = AdmissionReason::NeuralAcceleratorFamilyUnavailable;
  } else if (!facts.fp16_tensor_allocation_succeeded) {
    result.reason = AdmissionReason::TensorUnavailable;
  } else {
    result.reason = AdmissionReason::EligibleCandidate;
    result.eligible = true;
    result.fp8_tensor_types_available =
        facts.os_major >= 27 && facts.fp8_tensor_allocations_succeeded;
  }
  return result;
}

}  // namespace netal::ios_sr
