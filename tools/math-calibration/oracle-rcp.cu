#include <cuda_runtime.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

__global__ void rcp_file_kernel(const uint32_t* inputs, uint32_t* outputs,
                                uint64_t count) {
  const uint64_t index = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
  if (index >= count) return;
  const float input = __uint_as_float(inputs[index]);
  float output;
  asm volatile("rcp.approx.f32 %0, %1;" : "=f"(output) : "f"(input));
  outputs[index] = __float_as_uint(output);
}

__global__ void rcp_sweep_kernel(uint32_t* outputs, uint64_t count) {
  const uint64_t index = blockIdx.x * uint64_t(blockDim.x) + threadIdx.x;
  if (index >= count) return;
  const uint32_t mantissa = uint32_t(index) & 0x7fffffu;
  const uint32_t exponent_band = uint32_t(index >> 23u);
  const uint32_t input_bits = ((127u + exponent_band) << 23u) | mantissa;
  const float input = __uint_as_float(input_bits);
  float output;
  asm volatile("rcp.approx.f32 %0, %1;" : "=f"(output) : "f"(input));
  outputs[index] = __float_as_uint(output);
}

void check(cudaError_t status, const char* action) {
  if (status != cudaSuccess) {
    throw std::runtime_error(std::string(action) + ": " +
                             cudaGetErrorString(status));
  }
}

std::vector<uint32_t> read_words(const std::string& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("opening input failed");
  const auto bytes = stream.tellg();
  if (bytes <= 0 || bytes % 4 != 0) {
    throw std::runtime_error("input geometry failed");
  }
  std::vector<uint32_t> words(size_t(bytes) / 4u);
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(words.data()), bytes);
  if (!stream) throw std::runtime_error("reading input failed");
  return words;
}

void write_words(const std::string& path, const std::vector<uint32_t>& words) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  stream.write(reinterpret_cast<const char*>(words.data()),
               std::streamsize(words.size() * sizeof(uint32_t)));
  if (!stream) throw std::runtime_error("writing output failed");
}

int main(int argc, char** argv) {
  try {
    bool sweep = false;
    std::string input_path;
    std::string output_path;
    if (argc == 3 && std::string(argv[1]) == "--sweep") {
      sweep = true;
      output_path = argv[2];
    } else if (argc == 4 && std::string(argv[1]) == "--file") {
      input_path = argv[2];
      output_path = argv[3];
    } else {
      std::cerr << "usage: sm120-rcp-sweep --file INPUT OUTPUT | --sweep OUTPUT\n";
      return 2;
    }
    const uint64_t count = sweep ? (uint64_t(2) << 23u)
                                 : read_words(input_path).size();
    std::vector<uint32_t> inputs;
    if (!sweep) inputs = read_words(input_path);
    std::vector<uint32_t> outputs(static_cast<size_t>(count), 0u);
    uint32_t* device_inputs = nullptr;
    uint32_t* device_outputs = nullptr;
    if (!sweep) {
      check(cudaMalloc(&device_inputs, count * sizeof(uint32_t)),
            "allocating inputs");
      check(cudaMemcpy(device_inputs, inputs.data(), count * sizeof(uint32_t),
                       cudaMemcpyHostToDevice),
            "copying inputs");
    }
    check(cudaMalloc(&device_outputs, count * sizeof(uint32_t)),
          "allocating outputs");
    constexpr uint32_t threads = 256;
    const uint32_t blocks = uint32_t((count + threads - 1u) / threads);
    if (sweep) {
      rcp_sweep_kernel<<<blocks, threads>>>(device_outputs, count);
    } else {
      rcp_file_kernel<<<blocks, threads>>>(device_inputs, device_outputs, count);
    }
    check(cudaGetLastError(), "launching RCP kernel");
    check(cudaDeviceSynchronize(), "synchronizing RCP kernel");
    check(cudaMemcpy(outputs.data(), device_outputs, count * sizeof(uint32_t),
                     cudaMemcpyDeviceToHost),
          "copying outputs");
    write_words(output_path, outputs);
    cudaFree(device_outputs);
    if (device_inputs != nullptr) cudaFree(device_inputs);
    cudaDeviceProp properties{};
    check(cudaGetDeviceProperties(&properties, 0), "reading GPU properties");
    std::cout << "gpu=" << properties.name << "\n";
    std::cout << "compute_capability=" << properties.major << "."
              << properties.minor << "\n";
    std::cout << "mode=" << (sweep ? "sweep" : "file") << "\n";
    std::cout << "count=" << count << "\n";
    std::cout << "result=pass\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << "\n";
    return 1;
  }
}
