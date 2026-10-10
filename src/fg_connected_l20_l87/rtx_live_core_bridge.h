#pragma once

#include "../fg_connected_tail_l84_l87/core_bridge.h"

int RunFgConnectedCoreL56L83Live(
    const char* module_directory, const char* model_directory,
    std::uint8_t poison, std::uint64_t input_a_cuda_ptr,
    std::uint64_t input_b_cuda_ptr, std::uint64_t input_c_cuda_ptr,
    std::uint64_t l83_output_cuda_ptr,
    FgConnectedCoreBridgeResult* result);
