// Standalone arithmetic-table regeneration; no model, frame, DLL or captured input.
// Kernels reproduce the table-producing instruction chains documented in
// runtime/fg-arithmetic-calibration/manifest.json. The separate normalization
// experiments from the historical probes are deliberately not part of this tool.
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__device__ float Ex2(float value) {
  float result;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__global__ void Table(std::uint32_t* output, std::uint32_t count, int mode) {
  const std::uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= count) return;
  if (mode == 0) {
    output[index] = __float_as_uint(Ex2(-ldexpf(static_cast<float>(index), -24)));
  } else if (mode == 1) {
    output[index] = __float_as_uint(Ex2(-2.0f + ldexpf(static_cast<float>(index), -23)));
  } else {
    __half_raw raw{};
    raw.x = static_cast<std::uint16_t>(index);
    const float value = __half2float(raw);
    float scaled, exponential, denominator, reciprocal;
    asm("mul.ftz.f32 %0, %1, 0fBFB8AA3B;" : "=f"(scaled) : "f"(value));
    exponential = Ex2(scaled);
    asm("add.ftz.f32 %0, %1, 0f3F800000;" : "=f"(denominator) : "f"(exponential));
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(reciprocal) : "f"(denominator));
    output[index] = __float_as_uint(reciprocal);
  }
}

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage: fg-scalar-tables <unit|fraction|sigmoid> <new-output-file>\n";
    return 2;
  }
  try {
    const std::string name = argv[1];
    const int mode = name == "unit" ? 0 : name == "fraction" ? 1 : name == "sigmoid" ? 2 : -1;
    if (mode < 0) throw std::runtime_error("invalid table name");
    if (std::filesystem::exists(argv[2])) throw std::runtime_error("output exists");
    const std::uint32_t count = mode == 0 ? 16777217u : mode == 1 ? 8388609u : 65536u;
    std::vector<std::uint32_t> words(count);
    std::uint32_t* device = nullptr;
    Check(cudaMalloc(&device, count * sizeof(std::uint32_t)));
    Check(cudaMemset(device, 0xa5, count * sizeof(std::uint32_t)));
    Table<<<(count + 255u) / 256u, 256>>>(device, count, mode);
    Check(cudaGetLastError());
    Check(cudaMemcpy(words.data(), device, count * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
    Check(cudaFree(device));
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(words.data()), count * sizeof(std::uint32_t));
    output.close();
    if (!output) throw std::runtime_error("output write failed");
    // The caller must compare the bytes/hash to the pinned identity. Generation
    // alone does not certify numerical parity or a compatible GPU/compiler.
    std::cout << "generated_records=" << count << " identity_verified=false\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
