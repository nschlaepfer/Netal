#include "netal_ios_sr_device_policy.h"

#include <cassert>
#include <iostream>
#include <string>

using namespace netal::ios_sr;

int main() {
  unsigned checks = 0;
  const auto check = [&](bool condition) { assert(condition); ++checks; };
  DeviceFacts supported;
  supported.platform = Platform::PhysicalIOS;
  supported.gpu_name = "Apple A20 Pro GPU";
  supported.os_major = 27;
  supported.metal_device_available = true;
  supported.supports_metal4 = true;
  supported.supports_apple10 = true;
  supported.fp16_tensor_allocation_succeeded = true;

  for (const auto name : {"Apple A19 Pro GPU", "Apple A20 Pro GPU", "A19 Pro", "A20 Pro",
                          "  apple\tA20  PRO\nGPU  ", "A19 Pro GPU", "Apple A20 Pro"}) {
    auto facts = supported; facts.gpu_name = name;
    const auto result = EvaluateDevice(facts);
    check(result.eligible && result.neural_accelerator_candidate);
    check(!result.fp8_tensor_types_available);
  }
  check(ClassifyChipName("Apple A19 Pro GPU") == Chip::A19Pro);
  check(ClassifyChipName("Apple A20 Pro GPU") == Chip::A20Pro);
  for (const auto name : {"Apple A19 GPU", "Apple A20 GPU", "Apple M5 GPU", "Apple GPU",
                          "A19Pro", "A20 Pro Max", "Apple A20 Pro GPU Extra",
                          "Apple Apple A20 Pro GPU", "Apple A20 Pro GPU GPU", "", "a20 pro gpu!"}) {
    auto facts = supported; facts.gpu_name = name;
    const auto result = EvaluateDevice(facts);
    check(!result.eligible && result.chip == Chip::Unknown);
    check(result.reason == AdmissionReason::UnsupportedChip);
    check(!result.neural_accelerator_candidate && !result.fp8_tensor_types_available);
  }
  const std::string nul_name("Apple A20 Pro GPU\0Extra", 23);
  check(ClassifyChipName(nul_name) == Chip::Unknown);
  check(ClassifyChipName("Apple\xC2\xA0" "A20 Pro GPU") == Chip::Unknown);

  for (const auto platform : {Platform::Other, Platform::Simulator, Platform::Catalyst}) {
    auto facts = supported; facts.platform = platform;
    const auto result = EvaluateDevice(facts);
    check(!result.eligible && result.reason == AdmissionReason::UnsupportedPlatform);
    check(!result.neural_accelerator_candidate);
  }
  const auto denied = [&](DeviceFacts facts, AdmissionReason reason) {
    const auto result = EvaluateDevice(facts);
    check(!result.eligible && result.reason == reason && !result.fp8_tensor_types_available);
  };
  { auto f = supported; f.metal_device_available = false; denied(f, AdmissionReason::NoMetalDevice); }
  { auto f = supported; f.os_major = 25; denied(f, AdmissionReason::UnsupportedOS); }
  { auto f = supported; f.supports_metal4 = false; denied(f, AdmissionReason::Metal4Unavailable); }
  { auto f = supported; f.supports_apple10 = false; denied(f, AdmissionReason::NeuralAcceleratorFamilyUnavailable); }
  { auto f = supported; f.fp16_tensor_allocation_succeeded = false; denied(f, AdmissionReason::TensorUnavailable); }
  { auto f = supported; f.os_major = 26; check(EvaluateDevice(f).eligible); }
  { auto f = supported; f.os_major = 26; f.fp8_tensor_allocations_succeeded = true;
    check(EvaluateDevice(f).eligible && !EvaluateDevice(f).fp8_tensor_types_available); }
  { auto f = supported; f.fp8_tensor_allocations_succeeded = true;
    check(EvaluateDevice(f).eligible && EvaluateDevice(f).fp8_tensor_types_available); }
  { auto f = supported; f.fp8_tensor_allocations_succeeded = true; f.supports_metal4 = false;
    denied(f, AdmissionReason::Metal4Unavailable); }
  check(!EvaluateDevice(DeviceFacts{}).eligible);
  check(ChipName(Chip::A19Pro) == "A19 Pro" && ChipName(Chip::A20Pro) == "A20 Pro");
  check(AdmissionReasonText(AdmissionReason::EligibleCandidate).find("unvalidated") != std::string_view::npos);
  std::cout << checks << " mobile SR eligibility checks passed\n";
}
