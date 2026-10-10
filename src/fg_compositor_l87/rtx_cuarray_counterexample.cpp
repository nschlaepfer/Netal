#include <windows.h>
#include <bcrypt.h>
#include <cuda.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "compositor_l87_contract.h"

// Rejected counterexample only: a generic CUDA array cannot carry the D3D12
// RGBA8_UNORM surface semantics required by the selected SUST.P instruction.
// The accepted RTX oracle is rtx_nvapi_oracle.cpp.

#pragma comment(lib, "bcrypt.lib")

namespace fs = std::filesystem;
using namespace fg_compositor_l87;

namespace {

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t RgbHalf(std::uint64_t pixel, std::uint32_t channel) {
  const auto bits = SplitMix64(0x6a09e667f3bcc909ull ^
                               (pixel * 0x9e3779b97f4a7c15ull) ^
                               (std::uint64_t{channel + 1u} *
                                0xbf58476d1ce4e5b9ull));
  const std::uint16_t sign = (bits >> 63u) != 0u ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 17u) & 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (bits & 0x03ffu));
}

std::uint16_t AlphaHalf(std::uint64_t pixel) {
  const auto bits = SplitMix64(0xbb67ae8584caa73bull ^
                               (pixel * 0x94d049bb133111ebull));
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((bits >> 23u) & 3u));
  return static_cast<std::uint16_t>((exponent << 10u) |
                                    (bits & 0x03ffu));
}

std::vector<std::uint8_t> GenerateHalf4(std::size_t bytes,
                                        std::uint64_t salt) {
  if ((bytes % 8u) != 0u) throw std::runtime_error("half4 byte drift");
  std::vector<std::uint8_t> result(bytes);
  for (std::size_t pixel = 0; pixel < bytes / 8u; ++pixel) {
    const std::uint64_t identity = pixel ^ salt;
    Store16(&result, pixel * 8u, RgbHalf(identity, 0));
    Store16(&result, pixel * 8u + 2u, RgbHalf(identity, 1));
    Store16(&result, pixel * 8u + 4u, RgbHalf(identity, 2));
    Store16(&result, pixel * 8u + 6u, AlphaHalf(identity));
  }
  return result;
}

std::vector<std::uint8_t> GenerateMask(bool sparse) {
  std::vector<std::uint8_t> mask(kMaskBytes);
  for (std::size_t i = 0; i < mask.size(); ++i) {
    const auto value = SplitMix64(0x3c6ef372fe94f82bull ^ i);
    mask[i] = sparse && ((i % 17u) == 0u || (i % 101u) == 7u)
                  ? 0u
                  : static_cast<std::uint8_t>(1u + (value % 255u));
  }
  return mask;
}

std::vector<std::uint8_t> Read(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid size " + path.string());
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const fs::path& path, const std::vector<std::uint8_t>& bytes) {
  std::ofstream stream(path, std::ios::binary | std::ios::trunc);
  if (!stream) throw std::runtime_error("unable to write " + path.string());
  stream.write(reinterpret_cast<const char*>(bytes.data()),
               static_cast<std::streamsize>(bytes.size()));
  if (!stream) throw std::runtime_error("short write " + path.string());
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  BCRYPT_ALG_HANDLE algorithm = nullptr;
  BCRYPT_HASH_HANDLE hash = nullptr;
  DWORD object_bytes = 0, digest_bytes = 0, written = 0;
  auto require = [](NTSTATUS status, const char* op) {
    if (status < 0) throw std::runtime_error(std::string(op) + " failed");
  };
  require(BCryptOpenAlgorithmProvider(&algorithm, BCRYPT_SHA256_ALGORITHM,
                                      nullptr, 0),
          "BCryptOpenAlgorithmProvider");
  std::vector<std::uint8_t> object;
  std::vector<std::uint8_t> digest;
  try {
    require(BCryptGetProperty(algorithm, BCRYPT_OBJECT_LENGTH,
                              reinterpret_cast<PUCHAR>(&object_bytes),
                              sizeof(object_bytes), &written, 0),
            "BCryptGetProperty(object)");
    require(BCryptGetProperty(algorithm, BCRYPT_HASH_LENGTH,
                              reinterpret_cast<PUCHAR>(&digest_bytes),
                              sizeof(digest_bytes), &written, 0),
            "BCryptGetProperty(digest)");
    object.resize(object_bytes);
    digest.resize(digest_bytes);
    require(BCryptCreateHash(algorithm, &hash, object.data(), object_bytes,
                             nullptr, 0, 0),
            "BCryptCreateHash");
    require(BCryptHashData(hash, const_cast<PUCHAR>(bytes.data()),
                           static_cast<ULONG>(bytes.size()), 0),
            "BCryptHashData");
    require(BCryptFinishHash(hash, digest.data(), digest_bytes, 0),
            "BCryptFinishHash");
    BCryptDestroyHash(hash);
    BCryptCloseAlgorithmProvider(algorithm, 0);
  } catch (...) {
    if (hash != nullptr) BCryptDestroyHash(hash);
    if (algorithm != nullptr) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

void Check(CUresult result, const char* operation) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* description = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &description);
  throw std::runtime_error(std::string(operation) + " failed: " +
                           (name ? name : "unknown") + " / " +
                           (description ? description : "unknown"));
}

struct GuardedBuffer {
  CUdeviceptr base = 0;
  std::size_t payload_bytes = 0;
  CUdeviceptr payload() const { return base + kGuardBytes; }
  void Allocate(const std::vector<std::uint8_t>& source) {
    payload_bytes = source.size();
    Check(cuMemAlloc(&base, source.size() + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, source.size() + 2u * kGuardBytes),
          "cuMemsetD8(guard)");
    Check(cuMemcpyHtoD(payload(), source.data(), source.size()),
          "cuMemcpyHtoD(payload)");
  }
  std::vector<std::uint8_t> ReadAll() const {
    std::vector<std::uint8_t> bytes(payload_bytes + 2u * kGuardBytes);
    Check(cuMemcpyDtoH(bytes.data(), base, bytes.size()), "cuMemcpyDtoH");
    return bytes;
  }
  void Free() {
    if (base != 0) Check(cuMemFree(base), "cuMemFree");
    base = 0;
  }
  ~GuardedBuffer() {
    if (base != 0) cuMemFree(base);
  }
};

std::vector<std::uint8_t> Payload(const std::vector<std::uint8_t>& guarded,
                                  std::size_t bytes) {
  return {guarded.begin() + kGuardBytes,
          guarded.begin() + kGuardBytes + bytes};
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

std::size_t CanaryMismatches(const std::vector<std::uint8_t>& bytes,
                             std::size_t payload_bytes) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < kGuardBytes; ++i) {
    count += bytes[i] != kGuardValue;
    count += bytes[kGuardBytes + payload_bytes + i] != kGuardValue;
  }
  return count;
}

std::size_t PoisonPixels(const std::vector<std::uint8_t>& output,
                         std::uint8_t poison) {
  std::size_t count = 0;
  for (std::size_t i = 0; i < output.size(); i += 4u) {
    count += output[i] == poison && output[i + 1u] == poison &&
             output[i + 2u] == poison && output[i + 3u] == poison;
  }
  return count;
}

std::size_t ZeroMaskPixels(const std::vector<std::uint8_t>& mask) {
  std::size_t count = 0;
  for (const auto value : mask) count += value == 0;
  return count;
}

void Pack64(std::array<std::uint8_t, kPackedParameterBytes>* packed,
            std::size_t offset, std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

void Pack32(std::array<std::uint8_t, kPackedParameterBytes>* packed,
            std::size_t offset, std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

std::uint8_t ParsePoison(const std::string& text) {
  const unsigned long value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison byte");
  return static_cast<std::uint8_t>(value);
}

void CopyHostToArray(CUarray array, const std::vector<std::uint8_t>& bytes,
                     std::size_t row_bytes, std::size_t height) {
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_HOST;
  copy.srcHost = bytes.data();
  copy.srcPitch = row_bytes;
  copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
  copy.dstArray = array;
  copy.WidthInBytes = row_bytes;
  copy.Height = height;
  Check(cuMemcpy2D(&copy), "cuMemcpy2D(host-to-array)");
}

std::vector<std::uint8_t> CopyArrayToHost(CUarray array,
                                           std::size_t row_bytes,
                                           std::size_t height) {
  std::vector<std::uint8_t> bytes(row_bytes * height);
  CUDA_MEMCPY2D copy{};
  copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
  copy.srcArray = array;
  copy.dstMemoryType = CU_MEMORYTYPE_HOST;
  copy.dstHost = bytes.data();
  copy.dstPitch = row_bytes;
  copy.WidthInBytes = row_bytes;
  copy.Height = height;
  Check(cuMemcpy2D(&copy), "cuMemcpy2D(array-to-host)");
  return bytes;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-compositor-l87-rtx <module> <output-dir> <run-id> <poison-hex> <full|sparse>\n";
    return 2;
  }
  try {
    const fs::path module_path = argv[1];
    const fs::path output_directory = argv[2];
    const std::string run_id = argv[3];
    const std::uint8_t poison = ParsePoison(argv[4]);
    const std::string mask_mode = argv[5];
    if (mask_mode != "full" && mask_mode != "sparse")
      throw std::runtime_error("unsupported mask mode");
    const bool sparse = mask_mode == "sparse";
    fs::create_directories(output_directory);

    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != kModuleBytes ||
        Sha256(module_bytes) != kModuleSha256)
      throw std::runtime_error("selected module identity mismatch");
    std::array<std::vector<std::uint8_t>, 4> pyramid;
    for (std::size_t i = 0; i < pyramid.size(); ++i)
      pyramid[i] = GenerateHalf4(kPyramidBytes[i], 0x100000001b3ull * (i + 1));
    const auto mask = GenerateMask(sparse);
    const std::vector<std::uint8_t> initial_output(kOutputBytes, poison);

    Check(cuInit(0), "cuInit");
    CUdevice device = 0;
    Check(cuDeviceGet(&device, 0), "cuDeviceGet");
    CUcontext context = nullptr;
    Check(cuDevicePrimaryCtxRetain(&context, device), "cuDevicePrimaryCtxRetain");
    Check(cuCtxSetCurrent(context), "cuCtxSetCurrent");
    CUmodule module = nullptr;
    Check(cuModuleLoadData(&module, module_bytes.data()), "cuModuleLoadData");
    CUfunction function = nullptr;
    Check(cuModuleGetFunction(&function, module, "main_kernel"),
          "cuModuleGetFunction");

    std::array<GuardedBuffer, 4> buffers;
    for (std::size_t i = 0; i < buffers.size(); ++i) buffers[i].Allocate(pyramid[i]);

    CUDA_ARRAY3D_DESCRIPTOR mask_description{};
    mask_description.Width = kWidth;
    mask_description.Height = kHeight;
    mask_description.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    mask_description.NumChannels = 1;
    CUarray mask_array = nullptr;
    Check(cuArray3DCreate(&mask_array, &mask_description), "cuArray3DCreate(mask)");
    CopyHostToArray(mask_array, mask, kWidth, kHeight);

    CUDA_ARRAY3D_DESCRIPTOR output_description{};
    output_description.Width = kWidth;
    output_description.Height = kHeight;
    output_description.Format = CU_AD_FORMAT_UNSIGNED_INT8;
    output_description.NumChannels = 4;
    output_description.Flags = CUDA_ARRAY3D_SURFACE_LDST;
    CUarray output_array = nullptr;
    Check(cuArray3DCreate(&output_array, &output_description),
          "cuArray3DCreate(output)");
    CopyHostToArray(output_array, initial_output, kWidth * 4u, kHeight);

    CUDA_RESOURCE_DESC mask_resource{};
    mask_resource.resType = CU_RESOURCE_TYPE_ARRAY;
    mask_resource.res.array.hArray = mask_array;
    CUDA_TEXTURE_DESC texture_description{};
    texture_description.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
    texture_description.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
    texture_description.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
    texture_description.filterMode = CU_TR_FILTER_MODE_POINT;
    texture_description.mipmapFilterMode = CU_TR_FILTER_MODE_POINT;
    CUtexObject texture = 0;
    Check(cuTexObjectCreate(&texture, &mask_resource, &texture_description,
                            nullptr),
          "cuTexObjectCreate");

    CUDA_RESOURCE_DESC output_resource{};
    output_resource.resType = CU_RESOURCE_TYPE_ARRAY;
    output_resource.res.array.hArray = output_array;
    CUsurfObject surface = 0;
    Check(cuSurfObjectCreate(&surface, &output_resource), "cuSurfObjectCreate");

    std::array<std::uint8_t, kPackedParameterBytes> packed{};
    Pack64(&packed, 0, texture);
    Pack64(&packed, 8, surface);
    Pack32(&packed, 16, 0);
    Pack32(&packed, 20, 0);
    for (std::size_t i = 0; i < buffers.size(); ++i)
      Pack64(&packed, 24u + i * 8u, buffers[i].payload());
    Pack32(&packed, 56, kSelectedLevel);
    Pack32(&packed, 60, kWidth);
    Pack32(&packed, 64, kHeight);
    Pack32(&packed, 68, 0);
    std::size_t packed_size = packed.size();
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, kGridX, kGridY, 1, kBlockX, 1, 1, 0,
                         nullptr, nullptr, extra),
          "cuLaunchKernel(main_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto actual_mask = CopyArrayToHost(mask_array, kWidth, kHeight);
    const auto output = CopyArrayToHost(output_array, kWidth * 4u, kHeight);
    std::size_t input_mismatches = Mismatches(mask, actual_mask);
    std::size_t canary_mismatches = 0;
    std::array<std::vector<std::uint8_t>, 4> actual_pyramid;
    for (std::size_t i = 0; i < buffers.size(); ++i) {
      const auto guarded = buffers[i].ReadAll();
      canary_mismatches += CanaryMismatches(guarded, pyramid[i].size());
      actual_pyramid[i] = Payload(guarded, pyramid[i].size());
      input_mismatches += Mismatches(pyramid[i], actual_pyramid[i]);
    }
    const std::size_t zero_mask_pixels = ZeroMaskPixels(mask);
    const std::size_t poison_pixels = PoisonPixels(output, poison);
    const bool coverage_ok = poison_pixels == zero_mask_pixels;
    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        coverage_ok;

    Write(output_directory / "input-level0-after.rgba16f", actual_pyramid[0]);
    Write(output_directory / "input-mask-after.r8", actual_mask);
    Write(output_directory / "output.rgba8", output);
    std::ofstream receipt(output_directory / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    if (!receipt) throw std::runtime_error("unable to write receipt");
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.compositor-l87-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-driver\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"mask_mode\": \"" << mask_mode << "\",\n"
            << "  \"selected_launch_token\": 87,\n"
            << "  \"descriptor_trace_launch_alias\": 85,\n"
            << "  \"module_token\": 20,\n"
            << "  \"module_bytes\": " << module_bytes.size() << ",\n"
            << "  \"module_sha256\": \"" << kModuleSha256 << "\",\n"
            << "  \"raw_parameter_sha256\": \"" << kRawParameterSha256
            << "\",\n"
            << "  \"address_free_parameter_sha256\": \""
            << kAddressFreeParameterSha256 << "\",\n"
            << "  \"packed_parameter_bytes\": 72,\n"
            << "  \"abi\": {\"texture\": \"resource-000060 R8_UNORM 640x360 point-border\", \"surface\": \"resource-000055 RGBA8_UNORM 640x360\", \"output_origin\": [0, 0], \"pyramid_buffers\": [\"320x180\", \"160x90\", \"80x45\", \"40x23\"], \"selected_level\": 10, \"width\": 640, \"height\": 360},\n"
            << "  \"grid\": [40, 23, 1],\n"
            << "  \"block\": [256, 1, 1],\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"numerical_domain\": \"finite-normal-half4-positive-normal-alpha\",\n"
            << "  \"level0_bytes\": " << actual_pyramid[0].size() << ",\n"
            << "  \"level0_sha256\": \"" << Sha256(actual_pyramid[0])
            << "\",\n"
            << "  \"mask_bytes\": " << actual_mask.size() << ",\n"
            << "  \"mask_sha256\": \"" << Sha256(actual_mask) << "\",\n"
            << "  \"output_bytes\": " << output.size() << ",\n"
            << "  \"output_sha256\": \"" << Sha256(output) << "\",\n"
            << "  \"zero_mask_pixels\": " << zero_mask_pixels << ",\n"
            << "  \"poison_pixels_after\": " << poison_pixels << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"canary_mismatched_bytes\": " << canary_mismatches
            << ",\n"
            << "  \"external_expected_tensors_consumed\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail")
            << "\"\n}\n";
    receipt.close();
    std::cout << "probe=fg_compositor_l87_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "output_sha256=" << Sha256(output) << '\n'
              << "zero_mask_pixels=" << zero_mask_pixels << '\n'
              << "poison_pixels_after=" << poison_pixels << '\n'
              << "canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    Check(cuSurfObjectDestroy(surface), "cuSurfObjectDestroy");
    Check(cuTexObjectDestroy(texture), "cuTexObjectDestroy");
    Check(cuArrayDestroy(output_array), "cuArrayDestroy(output)");
    Check(cuArrayDestroy(mask_array), "cuArrayDestroy(mask)");
    for (auto& buffer : buffers) buffer.Free();
    Check(cuModuleUnload(module), "cuModuleUnload");
    Check(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
