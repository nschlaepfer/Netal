#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

struct FgConnectedPrefixResult {
  std::array<std::string, 32> checkpoint_sha256;
  std::string external_input_sha256;
  std::size_t input_model_mismatched_bytes = 0;
  std::size_t canary_mismatched_bytes = 0;
};

int RunFgConnectedPrefixL20L51(const char* module_directory,
                               const char* model_directory,
                               std::uint8_t poison,
                               std::uint64_t l51_output_cuda_ptr,
                               FgConnectedPrefixResult* result);

struct FgLiveInputsL56 {
  std::uint64_t input_a_gpu_va;
  std::uint64_t input_b_gpu_va;
  std::uint64_t input_c_gpu_va;
};
