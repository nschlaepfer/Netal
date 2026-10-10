#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

struct FgSubrectPrefixResult {
  std::array<std::string, 32> checkpoint_sha256;
  std::string external_input_sha256;
  std::size_t input_model_mismatched_bytes = 0;
  std::size_t canary_mismatched_bytes = 0;
};

int RunFgSubrectPrefixL18L49(const char* module_directory,
                             const char* model_directory,
                             std::uint8_t poison,
                             std::uint64_t l49_output_cuda_ptr,
                             FgSubrectPrefixResult* result);

struct FgSubrectCoreResult {
  std::array<std::string, 27> checkpoint_sha256;
  std::array<std::string, 3> external_input_sha256;
  std::size_t input_model_mismatched_bytes = 0;
  std::size_t canary_mismatched_bytes = 0;
};

int RunFgSubrectCoreL54L80Live(
    const char* module_directory, const char* model_directory,
    std::uint8_t poison, std::uint64_t input_a_cuda_ptr,
    std::uint64_t input_b_cuda_ptr, std::uint64_t input_c_cuda_ptr,
    std::uint64_t l81_output_cuda_ptr, FgSubrectCoreResult* result);
