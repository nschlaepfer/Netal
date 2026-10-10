// PRESERVED COUNTEREXAMPLE ONLY.
// Generic CUDA arrays do not reproduce the D3D12 typed-UNORM surface
// semantics of selected launch 84. Acceptance uses rtx_nvapi_oracle.cpp.
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

#pragma comment(lib, "bcrypt.lib")

namespace {

constexpr std::uint32_t kWidth = 640;
constexpr std::uint32_t kHeight = 360;
constexpr std::uint32_t kPaddedHeight = 368;
constexpr std::size_t kMotionBytes =
    static_cast<std::size_t>(kWidth) * kPaddedHeight * 16u;
constexpr std::size_t kColorBytes =
    static_cast<std::size_t>(kWidth) * kHeight * 8u;
constexpr std::size_t kMaskBytes =
    static_cast<std::size_t>(kWidth) * kHeight;
constexpr std::array<std::size_t, 4> kPyramidBytes = {
    320u * 180u * 8u, 160u * 90u * 8u, 80u * 45u * 8u, 40u * 23u * 8u};
constexpr std::size_t kGuardBytes = 256;
constexpr std::uint8_t kGuardValue = 0xC7;
constexpr char kModuleSha[] =
    "3d8c6aeba3a82fdeb99672243a3d392776bd1742576ab44e8ef42af105c2fd76";
constexpr char kRawParameterSha[] =
    "31cfc2016f76ad79fe92717d4bb3b38ceb6c8b3f14968f6b5f1e2f66503e548a";
constexpr char kGenerator[] =
    "splitmix64-fg-postprocess-l84-independent-textures-motion-v1";

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

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  if (size < 0) throw std::runtime_error("invalid file size");
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return bytes;
}

void Write(const std::filesystem::path& path,
           const std::vector<std::uint8_t>& bytes) {
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
  auto require = [](NTSTATUS status, const char* operation) {
    if (status < 0) throw std::runtime_error(std::string(operation) + " failed");
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
    if (hash) BCryptDestroyHash(hash);
    if (algorithm) BCryptCloseAlgorithmProvider(algorithm, 0);
    throw;
  }
  std::ostringstream text;
  text << std::hex << std::setfill('0');
  for (const auto byte : digest) text << std::setw(2) << unsigned(byte);
  return text.str();
}

std::uint64_t SplitMix64(std::uint64_t value) {
  value += 0x9E3779B97F4A7C15ull;
  value = (value ^ (value >> 30u)) * 0xBF58476D1CE4E5B9ull;
  value = (value ^ (value >> 27u)) * 0x94D049BB133111EBull;
  return value ^ (value >> 31u);
}

void Store16(std::vector<std::uint8_t>* bytes, std::size_t offset,
             std::uint16_t value) {
  (*bytes)[offset] = static_cast<std::uint8_t>(value);
  (*bytes)[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

std::uint16_t PositiveUnitHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(11u + ((mixed >> 17u) % 4u));
  return static_cast<std::uint16_t>((exponent << 10u) | (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateColor(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kColorBytes);
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 3; ++channel) {
        Store16(&bytes, pixel * 8u + channel * 2u,
                PositiveUnitHalf(seed ^ pixel * 0xD6E8FEB86659FD93ull ^
                                 channel * 0xA0761D6478BD642Full));
      }
      Store16(&bytes, pixel * 8u + 6u, 0x3c00u);
    }
  }
  return bytes;
}

std::uint16_t SmallSignedHalf(std::uint64_t key) {
  const auto mixed = SplitMix64(key);
  const std::uint16_t sign = (mixed >> 63u) ? 0x8000u : 0u;
  const std::uint16_t exponent =
      static_cast<std::uint16_t>(3u + ((mixed >> 11u) % 3u));
  return static_cast<std::uint16_t>(sign | (exponent << 10u) |
                                    (mixed & 0x03ffu));
}

std::vector<std::uint8_t> GenerateMotion(std::uint64_t seed) {
  std::vector<std::uint8_t> bytes(kMotionBytes);
  for (std::uint32_t y = 0; y < kPaddedHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const std::size_t pixel = static_cast<std::size_t>(y) * kWidth + x;
      for (std::uint32_t channel = 0; channel < 4; ++channel) {
        Store16(&bytes, pixel * 16u + channel * 2u,
                SmallSignedHalf(seed ^ pixel * 0xE7037ED1A0B428DBull ^
                                channel * 0x8EBC6AF09C88C6E3ull));
      }
      const auto mixed = SplitMix64(seed ^ pixel * 0x589965CC75374CC3ull);
      const std::uint16_t logit = static_cast<std::uint16_t>(
          ((mixed >> 63u) ? 0x8000u : 0u) | (14u << 10u) |
          (mixed & 0x03ffu));
      Store16(&bytes, pixel * 16u + 8u, logit);
      Store16(&bytes, pixel * 16u + 10u, SmallSignedHalf(mixed ^ 0x11u));
      Store16(&bytes, pixel * 16u + 12u, SmallSignedHalf(mixed ^ 0x22u));
      Store16(&bytes, pixel * 16u + 14u, SmallSignedHalf(mixed ^ 0x33u));
    }
  }
  return bytes;
}

std::vector<std::uint8_t> GenerateMask(const std::string& fixture) {
  std::vector<std::uint8_t> bytes(kMaskBytes, 0u);
  if (fixture == "open") return bytes;
  if (fixture != "islands") throw std::runtime_error("unsupported fixture");
  for (std::uint32_t y = 0; y < kHeight; ++y) {
    for (std::uint32_t x = 0; x < kWidth; ++x) {
      const bool island =
          ((x >= 104u && x < 224u && y >= 72u && y < 184u) ||
           (x >= 376u && x < 552u && y >= 196u && y < 320u));
      const bool holes = ((x * 13u + y * 7u) % 47u) == 0u;
      bytes[static_cast<std::size_t>(y) * kWidth + x] =
          island && !holes ? 0xffu : 0u;
    }
  }
  return bytes;
}

std::size_t Mismatches(const std::vector<std::uint8_t>& a,
                       const std::vector<std::uint8_t>& b) {
  if (a.size() != b.size()) throw std::runtime_error("comparison size drift");
  std::size_t count = 0;
  for (std::size_t i = 0; i < a.size(); ++i) count += a[i] != b[i];
  return count;
}

struct GuardedBuffer {
  CUdeviceptr base = 0;
  std::size_t payload_bytes = 0;
  CUdeviceptr payload() const { return base + kGuardBytes; }
  void Allocate(std::size_t bytes, std::uint8_t fill) {
    payload_bytes = bytes;
    Check(cuMemAlloc(&base, bytes + 2u * kGuardBytes), "cuMemAlloc");
    Check(cuMemsetD8(base, kGuardValue, bytes + 2u * kGuardBytes),
          "cuMemsetD8(guard)");
    Check(cuMemsetD8(payload(), fill, bytes), "cuMemsetD8(payload)");
  }
  void Upload(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() != payload_bytes) throw std::runtime_error("upload size");
    Check(cuMemcpyHtoD(payload(), bytes.data(), bytes.size()), "cuMemcpyHtoD");
  }
  std::vector<std::uint8_t> DownloadPayload() const {
    std::vector<std::uint8_t> bytes(payload_bytes);
    Check(cuMemcpyDtoH(bytes.data(), payload(), bytes.size()), "cuMemcpyDtoH");
    return bytes;
  }
  std::size_t CanaryMismatches() const {
    std::vector<std::uint8_t> bytes(payload_bytes + 2u * kGuardBytes);
    Check(cuMemcpyDtoH(bytes.data(), base, bytes.size()), "cuMemcpyDtoH(all)");
    std::size_t count = 0;
    for (std::size_t i = 0; i < kGuardBytes; ++i) {
      count += bytes[i] != kGuardValue;
      count += bytes[kGuardBytes + payload_bytes + i] != kGuardValue;
    }
    return count;
  }
  ~GuardedBuffer() {
    if (base) cuMemFree(base);
  }
};

struct Array2D {
  CUarray array = nullptr;
  std::size_t width = 0;
  std::size_t height = 0;
  std::size_t pixel_bytes = 0;
  void Create(CUarray_format format, unsigned channels, std::size_t w,
              std::size_t h, bool surface) {
    width = w;
    height = h;
    const std::size_t component =
        format == CU_AD_FORMAT_HALF ? 2u : 1u;
    pixel_bytes = component * channels;
    CUDA_ARRAY3D_DESCRIPTOR descriptor{};
    descriptor.Width = w;
    descriptor.Height = h;
    descriptor.Depth = 0;
    descriptor.Format = format;
    descriptor.NumChannels = channels;
    descriptor.Flags = surface ? CUDA_ARRAY3D_SURFACE_LDST : 0u;
    Check(cuArray3DCreate(&array, &descriptor), "cuArray3DCreate");
  }
  void Upload(const std::vector<std::uint8_t>& bytes) {
    if (bytes.size() != width * height * pixel_bytes)
      throw std::runtime_error("array upload size");
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_HOST;
    copy.srcHost = bytes.data();
    copy.srcPitch = width * pixel_bytes;
    copy.dstMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.dstArray = array;
    copy.WidthInBytes = width * pixel_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(upload)");
  }
  std::vector<std::uint8_t> Download() const {
    std::vector<std::uint8_t> bytes(width * height * pixel_bytes);
    CUDA_MEMCPY2D copy{};
    copy.srcMemoryType = CU_MEMORYTYPE_ARRAY;
    copy.srcArray = array;
    copy.dstMemoryType = CU_MEMORYTYPE_HOST;
    copy.dstHost = bytes.data();
    copy.dstPitch = width * pixel_bytes;
    copy.WidthInBytes = width * pixel_bytes;
    copy.Height = height;
    Check(cuMemcpy2D(&copy), "cuMemcpy2D(download)");
    return bytes;
  }
  ~Array2D() {
    if (array) cuArrayDestroy(array);
  }
};

CUtexObject MakeTexture(CUarray array) {
  CUDA_RESOURCE_DESC resource{};
  resource.resType = CU_RESOURCE_TYPE_ARRAY;
  resource.res.array.hArray = array;
  CUDA_TEXTURE_DESC texture{};
  texture.addressMode[0] = CU_TR_ADDRESS_MODE_BORDER;
  texture.addressMode[1] = CU_TR_ADDRESS_MODE_BORDER;
  texture.addressMode[2] = CU_TR_ADDRESS_MODE_BORDER;
  texture.filterMode = CU_TR_FILTER_MODE_POINT;
  texture.flags = 0;
  texture.maxAnisotropy = 1;
  texture.minMipmapLevelClamp = 0.0f;
  texture.maxMipmapLevelClamp = 0.0f;
  CUtexObject object = 0;
  Check(cuTexObjectCreate(&object, &resource, &texture, nullptr),
        "cuTexObjectCreate");
  return object;
}

CUsurfObject MakeSurface(CUarray array) {
  CUDA_RESOURCE_DESC resource{};
  resource.resType = CU_RESOURCE_TYPE_ARRAY;
  resource.res.array.hArray = array;
  CUsurfObject object = 0;
  Check(cuSurfObjectCreate(&object, &resource), "cuSurfObjectCreate");
  return object;
}

void Pack64(std::array<std::uint8_t, 152>* packed, std::size_t offset,
            std::uint64_t value) {
  for (std::size_t i = 0; i < 8; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

void Pack32(std::array<std::uint8_t, 152>* packed, std::size_t offset,
            std::uint32_t value) {
  for (std::size_t i = 0; i < 4; ++i)
    (*packed)[offset + i] = static_cast<std::uint8_t>(value >> (8u * i));
}

std::uint8_t ParsePoison(const std::string& text) {
  const auto value = std::stoul(text, nullptr, 16);
  if (value > 255u) throw std::runtime_error("invalid poison");
  return static_cast<std::uint8_t>(value);
}

std::string JsonBool(bool value) { return value ? "true" : "false"; }

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: fg-postprocess-l84-rtx <module> <output-dir> <run-id> <fixture> <poison-hex>\n";
    return 2;
  }
  try {
    const std::filesystem::path module_path = argv[1];
    const std::filesystem::path output_root = argv[2];
    const std::string run_id = argv[3];
    const std::string fixture = argv[4];
    const std::uint8_t poison = ParsePoison(argv[5]);
    std::filesystem::create_directories(output_root);

    const auto module_bytes = Read(module_path);
    if (module_bytes.size() != 33936u || Sha256(module_bytes) != kModuleSha)
      throw std::runtime_error("selected module identity mismatch");
    const std::uint64_t fixture_seed =
        fixture == "open" ? 0xA0761D6478BD642Full :
        fixture == "islands" ? 0xE7037ED1A0B428DBull : 0ull;
    if (fixture_seed == 0) throw std::runtime_error("unsupported fixture");
    const auto color_a = GenerateColor(fixture_seed ^ 0x11111111ull);
    const auto color_b = GenerateColor(fixture_seed ^ 0x22222222ull);
    const auto mask = GenerateMask(fixture);
    const auto motion = GenerateMotion(fixture_seed ^ 0x33333333ull);

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

    Array2D color_a_array, color_b_array, mask_array;
    Array2D output_color_array, output_mask_array;
    color_a_array.Create(CU_AD_FORMAT_HALF, 4, kWidth, kHeight, false);
    color_b_array.Create(CU_AD_FORMAT_HALF, 4, kWidth, kHeight, false);
    mask_array.Create(CU_AD_FORMAT_UNSIGNED_INT8, 1, kWidth, kHeight, false);
    output_color_array.Create(CU_AD_FORMAT_UNSIGNED_INT8, 4, kWidth, kHeight,
                              true);
    output_mask_array.Create(CU_AD_FORMAT_UNSIGNED_INT8, 1, kWidth, kHeight,
                             true);
    color_a_array.Upload(color_a);
    color_b_array.Upload(color_b);
    mask_array.Upload(mask);
    output_color_array.Upload(std::vector<std::uint8_t>(kWidth * kHeight * 4u,
                                                        poison));
    output_mask_array.Upload(std::vector<std::uint8_t>(kMaskBytes, poison));

    const CUtexObject color_a_texture = MakeTexture(color_a_array.array);
    const CUtexObject color_b_texture = MakeTexture(color_b_array.array);
    const CUtexObject mask_texture = MakeTexture(mask_array.array);
    const CUsurfObject output_color_surface =
        MakeSurface(output_color_array.array);
    const CUsurfObject output_mask_surface = MakeSurface(output_mask_array.array);

    GuardedBuffer motion_buffer;
    std::array<GuardedBuffer, 4> pyramid;
    motion_buffer.Allocate(kMotionBytes, 0);
    motion_buffer.Upload(motion);
    for (std::size_t i = 0; i < pyramid.size(); ++i)
      pyramid[i].Allocate(kPyramidBytes[i], poison);

    std::array<std::uint8_t, 152> packed{};
    Pack64(&packed, 0, output_color_surface);
    Pack64(&packed, 16, color_a_texture);
    Pack64(&packed, 24, color_b_texture);
    Pack64(&packed, 64, motion_buffer.payload());
    Pack64(&packed, 72, mask_texture);
    for (std::size_t i = 0; i < pyramid.size(); ++i)
      Pack64(&packed, 80u + i * 8u, pyramid[i].payload());
    Pack64(&packed, 112, output_mask_surface);
    Pack32(&packed, 128, kWidth);
    Pack32(&packed, 132, kPaddedHeight);
    Pack32(&packed, 136, kWidth);
    Pack32(&packed, 140, kHeight);
    Pack32(&packed, 144, 0x3f000000u);
    Pack32(&packed, 148, 1u);
    std::size_t packed_size = packed.size();
    void* extra[] = {CU_LAUNCH_PARAM_BUFFER_POINTER, packed.data(),
                     CU_LAUNCH_PARAM_BUFFER_SIZE, &packed_size,
                     CU_LAUNCH_PARAM_END};
    Check(cuLaunchKernel(function, 40, 23, 1, 256, 1, 1, 0, nullptr, nullptr,
                         extra),
          "cuLaunchKernel(main_kernel)");
    Check(cuCtxSynchronize(), "cuCtxSynchronize");

    const auto color_a_after = color_a_array.Download();
    const auto color_b_after = color_b_array.Download();
    const auto mask_after = mask_array.Download();
    const auto motion_after = motion_buffer.DownloadPayload();
    const auto output_color = output_color_array.Download();
    const auto output_mask = output_mask_array.Download();
    std::array<std::vector<std::uint8_t>, 4> pyramid_after;
    std::size_t canary_mismatches = motion_buffer.CanaryMismatches();
    for (std::size_t i = 0; i < pyramid.size(); ++i) {
      pyramid_after[i] = pyramid[i].DownloadPayload();
      canary_mismatches += pyramid[i].CanaryMismatches();
    }
    const std::size_t input_mismatches =
        Mismatches(color_a, color_a_after) + Mismatches(color_b, color_b_after) +
        Mismatches(mask, mask_after) + Mismatches(motion, motion_after);

    Write(output_root / "color-a-after.bin", color_a_after);
    Write(output_root / "color-b-after.bin", color_b_after);
    Write(output_root / "mask-after.bin", mask_after);
    Write(output_root / "motion-after.bin", motion_after);
    Write(output_root / "output-color.bin", output_color);
    Write(output_root / "output-mask.bin", output_mask);
    for (std::size_t i = 0; i < pyramid_after.size(); ++i)
      Write(output_root / ("pyramid-" + std::to_string(i) + ".bin"),
            pyramid_after[i]);

    std::vector<std::uint8_t> output_combined = output_color;
    output_combined.insert(output_combined.end(), output_mask.begin(),
                           output_mask.end());
    for (const auto& level : pyramid_after)
      output_combined.insert(output_combined.end(), level.begin(), level.end());
    std::vector<std::uint8_t> input_combined = color_a;
    input_combined.insert(input_combined.end(), color_b.begin(), color_b.end());
    input_combined.insert(input_combined.end(), mask.begin(), mask.end());
    input_combined.insert(input_combined.end(), motion.begin(), motion.end());

    std::size_t output_mask_zero_bytes = 0;
    std::size_t output_mask_one_bytes = 0;
    std::size_t output_mask_other_bytes = 0;
    for (const auto value : output_mask) {
      output_mask_zero_bytes += value == 0u;
      output_mask_one_bytes += value == 0xffu;
      output_mask_other_bytes += value != 0u && value != 0xffu;
    }
    const bool passed = input_mismatches == 0 && canary_mismatches == 0 &&
                        output_mask_other_bytes == 0;
    char device_name[256]{};
    Check(cuDeviceGetName(device_name, sizeof(device_name), device),
          "cuDeviceGetName");
    int major = 0, minor = 0;
    Check(cuDeviceGetAttribute(&major,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR,
                               device),
          "cuDeviceGetAttribute(major)");
    Check(cuDeviceGetAttribute(&minor,
                               CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR,
                               device),
          "cuDeviceGetAttribute(minor)");
    std::ofstream receipt(output_root / "run-receipt.json",
                          std::ios::binary | std::ios::trunc);
    receipt << "{\n"
            << "  \"schema\": \"neuralstudio.fg.postprocess-l84-run.v1\",\n"
            << "  \"backend\": \"rtx-cuda-driver\",\n"
            << "  \"run_id\": \"" << run_id << "\",\n"
            << "  \"process_id\": " << GetCurrentProcessId() << ",\n"
            << "  \"fixture\": \"" << fixture << "\",\n"
            << "  \"poison_byte\": " << unsigned(poison) << ",\n"
            << "  \"device\": \"" << device_name << "\",\n"
            << "  \"compute_capability\": \"" << major << '.' << minor
            << "\",\n"
            << "  \"selected_launch_token\": 84,\n"
            << "  \"module_token\": 17,\n"
            << "  \"module_bytes\": 33936,\n"
            << "  \"module_sha256\": \"" << kModuleSha << "\",\n"
            << "  \"raw_live_parameter_sha256\": \"" << kRawParameterSha
            << "\",\n"
            << "  \"packed_parameter_bytes\": 152,\n"
            << "  \"grid\": [40, 23, 1],\n"
            << "  \"block\": [256, 1, 1],\n"
            << "  \"input_generator\": \"" << kGenerator << "\",\n"
            << "  \"input_root_sha256\": \"" << Sha256(input_combined)
            << "\",\n"
            << "  \"output_root_sha256\": \"" << Sha256(output_combined)
            << "\",\n"
            << "  \"output_color_sha256\": \"" << Sha256(output_color)
            << "\",\n"
            << "  \"output_mask_sha256\": \"" << Sha256(output_mask)
            << "\",\n"
            << "  \"pyramid_sha256\": [\"" << Sha256(pyramid_after[0])
            << "\", \"" << Sha256(pyramid_after[1]) << "\", \""
            << Sha256(pyramid_after[2]) << "\", \""
            << Sha256(pyramid_after[3]) << "\"],\n"
            << "  \"output_bytes\": " << output_combined.size() << ",\n"
            << "  \"input_mismatched_bytes\": " << input_mismatches << ",\n"
            << "  \"buffer_canary_mismatched_bytes\": "
            << canary_mismatches << ",\n"
            << "  \"output_mask_zero_bytes\": " << output_mask_zero_bytes
            << ",\n"
            << "  \"output_mask_one_bytes\": " << output_mask_one_bytes
            << ",\n"
            << "  \"output_mask_other_bytes\": " << output_mask_other_bytes
            << ",\n"
            << "  \"expected_output_fed_to_execution\": false,\n"
            << "  \"archived_intermediates_consumed\": false,\n"
            << "  \"address_fields_recorded\": false,\n"
            << "  \"result\": \"" << (passed ? "pass" : "fail")
            << "\"\n}\n";
    receipt.close();

    std::cout << "probe=fg_postprocess_l84_rtx\n"
              << "run_id=" << run_id << '\n'
              << "process_id=" << GetCurrentProcessId() << '\n'
              << "fixture=" << fixture << '\n'
              << "output_root_sha256=" << Sha256(output_combined) << '\n'
              << "input_mismatched_bytes=" << input_mismatches << '\n'
              << "buffer_canary_mismatched_bytes=" << canary_mismatches << '\n'
              << "result=" << (passed ? "pass" : "fail") << '\n';

    cuSurfObjectDestroy(output_mask_surface);
    cuSurfObjectDestroy(output_color_surface);
    cuTexObjectDestroy(mask_texture);
    cuTexObjectDestroy(color_b_texture);
    cuTexObjectDestroy(color_a_texture);
    Check(cuModuleUnload(module), "cuModuleUnload");
    // Buffer and array RAII objects below still need the primary context while
    // their destructors execute. The fresh-process oracle intentionally lets
    // process teardown release the retained primary context after those CUDA
    // resources are destroyed.
    return passed ? 0 : 3;
  } catch (const std::exception& exception) {
    std::cerr << "error=" << exception.what() << '\n';
    return 1;
  }
}
