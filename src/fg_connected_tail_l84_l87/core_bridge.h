#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
struct FgConnectedCoreBridgeResult {
  std::array<std::string,27> checkpoint_sha256;
  std::array<std::string,3> external_input_sha256;
  std::size_t input_model_mismatched_bytes=0;
  std::size_t canary_mismatched_bytes=0;
};
int RunFgConnectedCoreL56L83(const char*module_directory,
                             const char*model_directory,
                             std::uint8_t poison,
                             std::uint64_t l83_output_gpu_va,
                             FgConnectedCoreBridgeResult*result);
