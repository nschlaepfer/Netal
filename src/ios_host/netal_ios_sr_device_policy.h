#pragma once

#include <string_view>

namespace netal::ios_sr {

// This is the mobile product policy, separate from model and inference validation.
// GPU family alone also admits devices outside the requested Pro-chip scope.
enum class Platform { Other, PhysicalIOS, Simulator, Catalyst };
enum class Chip { Unknown, A19Pro, A20Pro };
enum class AdmissionReason {
  EligibleCandidate,
  UnsupportedPlatform,
  NoMetalDevice,
  UnsupportedChip,
  UnsupportedOS,
  Metal4Unavailable,
  NeuralAcceleratorFamilyUnavailable,
  TensorUnavailable,
};

struct DeviceFacts {
  Platform platform = Platform::Other;
  std::string_view gpu_name;
  unsigned os_major = 0;
  bool metal_device_available = false;
  bool supports_metal4 = false;
  bool supports_apple10 = false;
  bool fp16_tensor_allocation_succeeded = false;
  bool fp8_tensor_allocations_succeeded = false;
};

struct Admission {
  Chip chip = Chip::Unknown;
  AdmissionReason reason = AdmissionReason::UnsupportedPlatform;
  bool eligible = false;
  // Hardware/API candidate indicators, not evidence of a DLSS model or dispatch.
  bool neural_accelerator_candidate = false;
  bool fp8_tensor_types_available = false;
};

Chip ClassifyChipName(std::string_view gpu_name);
std::string_view ChipName(Chip chip);
std::string_view AdmissionReasonText(AdmissionReason reason);
Admission EvaluateDevice(const DeviceFacts& facts);

}  // namespace netal::ios_sr
