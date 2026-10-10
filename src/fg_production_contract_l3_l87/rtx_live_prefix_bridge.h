#pragma once

#include <cstdint>

// Binds the accepted L20-L51 executor to the live allocation37 CUDA mapping.
// A zero pointer clears the binding after the nested executor returns.
void FgConnectedSetRtxLivePrefixInput(std::uint64_t input_cuda_ptr);
void FgConnectedSetRtxLivePrefixDebugPath(const char* path);

