#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include "fg_runtime.h"
#include "span_recorder.h"
#include "metal4_shim.h"
#include "tail_output_bridge.h"

#include <array>
#include <atomic>
#include <bit>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <unistd.h>

#include "../fg_production_contract_l3_l87/contract.h"
#include "../fg_production_contract_l3_l87/macos_live_handoff.h"

int FgRuntimeFullTailMain(int argc, const char* argv[]);

namespace {

namespace fs = std::filesystem;
namespace contract = fg_production_contract_l3_l87;

std::atomic_flag g_session_live = ATOMIC_FLAG_INIT;

constexpr std::size_t kGuardBytes = 256;
constexpr std::size_t kRomBytes = 2097152;
constexpr std::array<std::size_t, 5> kPyramidBytes = {
    57600, 86400, 21600, 5520, 1440};

struct L3Parameters {
  std::uint32_t width, height, half_width, half_height;
};
struct L4Parameters {
  std::uint32_t reset_mode, input_width, input_height, output_width,
      output_height;
};
struct L5Parameters {
  std::uint32_t width, height;
  float motion_divisor, neighbor_divisor, depth_scale;
  std::uint32_t invert_depth;
};
struct L8Parameters {
  std::uint32_t width, height, mode;
};
struct L9Parameters {
  std::uint32_t width, height;
};
struct StatusParameters {
  std::uint32_t count, value;
};
struct L11Parameters {
  std::uint32_t input_x, input_y, output_x, output_y, width, height;
};
struct L19Parameters {
  std::uint32_t input_width, input_height, tensor_width, tensor_height;
  std::int32_t input_y_offset;
};

static_assert(sizeof(L3Parameters) == 16);
static_assert(sizeof(L4Parameters) == 20);
static_assert(sizeof(L5Parameters) == 24);
static_assert(sizeof(L8Parameters) == 12);
static_assert(sizeof(L9Parameters) == 8);
static_assert(sizeof(StatusParameters) == 8);
static_assert(sizeof(L11Parameters) == 24);
static_assert(sizeof(L19Parameters) == 20);

std::vector<std::uint8_t> ReadFile(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read " + path.string());
  const auto size = stream.tellg();
  std::vector<std::uint8_t> result(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(result.data()), size);
  if (!stream) throw std::runtime_error("short read " + path.string());
  return result;
}

std::string Sha256(const std::vector<std::uint8_t>& bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> digest{};
  CC_SHA256(bytes.data(), static_cast<CC_LONG>(bytes.size()), digest.data());
  std::ostringstream result;
  result << std::hex << std::setfill('0');
  for (const auto byte : digest) result << std::setw(2) << unsigned(byte);
  return result.str();
}

std::size_t Mismatches(const std::vector<std::uint8_t>& lhs,
                       const std::vector<std::uint8_t>& rhs) {
  if (lhs.size() != rhs.size()) return std::max(lhs.size(), rhs.size());
  std::size_t result = 0;
  for (std::size_t index = 0; index < lhs.size(); ++index)
    result += lhs[index] != rhs[index];
  return result;
}

id<MTLComputePipelineState> MakePipeline(id<MTLDevice> device,
                                         const fs::path& path,
                                         const char* function_name) {
  NSError* error = nil;
  id<MTLLibrary> library = [device
      newLibraryWithURL:[NSURL fileURLWithPath:
                                  [NSString stringWithUTF8String:
                                                path.string().c_str()]]
                    error:&error];
  if (library == nil)
    throw std::runtime_error("metallib load failed: " + path.string());
  id<MTLFunction> function = [library
      newFunctionWithName:[NSString stringWithUTF8String:function_name]];
  id<MTLComputePipelineState> result =
      [device newComputePipelineStateWithFunction:function error:&error];
  if (function == nil || result == nil)
    throw std::runtime_error(std::string("pipeline failed: ") + function_name);
  return result;
}

void Wait(id<MTLCommandBuffer> command) {
  NSFGWaitAndRecord(command, NSFG_SPAN_EARLY);
}

template <std::size_t N>
void Store32(std::array<std::uint8_t, N>* bytes, std::size_t offset,
             std::uint32_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

template <std::size_t N>
void Store64(std::array<std::uint8_t, N>* bytes, std::size_t offset,
             std::uint64_t value) {
  std::memcpy(bytes->data() + offset, &value, sizeof(value));
}

class RuntimeCore final {
 public:
  explicit RuntimeCore(const NSFGCreateInfo& info)
      : device_((__bridge id<MTLDevice>)info.metal_device),
        queue_(info.metal4 ? (id<MTLCommandQueue>)NSFGMetal4ShimCreateQueue(
                                 (__bridge id<MTLDevice>)info.metal_device)
                           : (__bridge id<MTLCommandQueue>)info.metal_command_queue),
        metal4_(info.metal4),
        libs_(info.metallib_directory),
        model_(info.model_directory),
        rom_path_(info.rcp_rom_path),
        sigmoid_path_(info.sigmoid_lut_path),
        ex2_mode_(info.ex2_mode),
        auxiliary_mode_(info.auxiliary_mode),
        poison_(info.poison_byte),
        audit_enabled_(info.enable_audit),
        profiling_enabled_(info.enable_profiling) {
    if (device_ == nil || queue_ == nil || queue_.device != device_)
      throw std::invalid_argument("Metal device/queue mismatch");
    if (info.width != 640 || info.height != 360)
      throw std::invalid_argument("only the pinned 640x360 profile is accepted");
    if (auxiliary_mode_ != NSFG_AUXILIARY_INTERNAL_SMALL_FLOW &&
        auxiliary_mode_ != NSFG_AUXILIARY_LIVE_EARLY_RESOURCES)
      throw std::invalid_argument("unsupported auxiliary mode");
    LoadPipelines();
    LoadArchitectureData(info);
    AllocateResources();
    RestoreInitialState();
    phase_ = NSFG_SESSION_NEEDS_RESET;
  }

  ~RuntimeCore() { NSFGTailReleasePersistentResources(); }

  NSFGStatus EvaluateMetal4(const NSFGFrameInfo& frame, void* command_buffer,
                            NSFGFrameResult* result) {
    if (!metal4_) return Fail(NSFG_STATUS_INVALID_ARGUMENT, "session was not created for Metal 4");
    if (command_buffer == nullptr) return Fail(NSFG_STATUS_INVALID_ARGUMENT, "missing MTL4 command buffer");
    if (audit_enabled_ || profiling_enabled_)
      return Fail(NSFG_STATUS_INVALID_ARGUMENT, "Metal 4 frames cannot audit or profile");
    NSFGMetal4ShimBeginFrame(queue_, (__bridge id)command_buffer);
    const auto status = Evaluate(frame, result);
    NSFGMetal4ShimEndFrame(queue_);
    return status;
  }

  NSFGStatus Evaluate(const NSFGFrameInfo& frame, NSFGFrameResult* result) {
    ClearError();
    if (phase_ == NSFG_SESSION_EVALUATING) return Fail(NSFG_STATUS_BUSY, "session is busy");
    const auto validation = ValidateFrame(frame);
    if (validation != NSFG_STATUS_OK) return validation;
    const auto allocations_before = resource_allocations_;
    last_checkpoints_.clear();
    NSFGSpanBegin(profiling_enabled_);
    NSFGSpanSetDeferWaits(!audit_enabled_ && !profiling_enabled_ &&
                          std::getenv("NSFG_SYNC_WAITS") == nullptr);
    phase_ = NSFG_SESSION_EVALUATING;
    try {
      const auto current = TextureRef(frame.current_backbuffer_rgba8);
      const auto motion = TextureRef(frame.motion_rg16f);
      const auto depth = TextureRef(frame.depth_r32f);
      std::vector<std::uint8_t> current_before;
      std::vector<std::uint8_t> motion_before;
      std::vector<std::uint8_t> depth_before;
      if (audit_enabled_) {
        current_before = ReadTexture(current, 4);
        motion_before = ReadTexture(motion, 4);
        depth_before = ReadTexture(depth, 4);
      }
      if (frame.reset) {
        if (!metal4_) RestoreInitialState();   // CPU writes; unsafe with game frames in flight
        ExecuteReset(current, motion, depth);
        generated_count_ = 0;
        ++reset_count_;
        if (result != nullptr) {
          *result = {false, 9, frame.frame_id, resource_allocations_,
                     resource_allocations_ - allocations_before};
        }
      } else {
        const auto target = TextureRef(frame.generated_target_rgba8);
        ExecuteGenerated(current, motion, depth, target, frame.frame_id);
        ++generated_count_;
        ++generated_frame_total_;
        if (result != nullptr) {
          *result = {true, 76, frame.frame_id, resource_allocations_,
                     resource_allocations_ - allocations_before};
        }
      }
      if (audit_enabled_) {
        input_mismatches_ += Mismatches(current_before, ReadTexture(current, 4));
        input_mismatches_ += Mismatches(motion_before, ReadTexture(motion, 4));
        input_mismatches_ += Mismatches(depth_before, ReadTexture(depth, 4));
        canary_mismatches_ += AuditCanaries();
      }
      NSFGSpanFlush();
      last_current_ = current;
      last_frame_id_ = frame.frame_id;
      has_frame_ = true;
      last_spans_ = NSFGRecordedSpans();
      phase_ = NSFG_SESSION_READY;
      return NSFG_STATUS_OK;
    } catch (const std::exception& error) {
      NSFGSetTailOutputTarget(nil);
      NSFGSetTailContext(nil, nil, false, false);
      NSFGSetTailLiveResources(nullptr);
      try { NSFGSpanFlush(); } catch (const std::exception&) {}
      last_spans_ = NSFGRecordedSpans();
      phase_ = NSFG_SESSION_FAULTED;
      return Fail(NSFG_STATUS_DEVICE_ERROR, error.what());
    }
  }

  NSFGStatus Invalidate() {
    ClearError();
    if (phase_ == NSFG_SESSION_EVALUATING) return Fail(NSFG_STATUS_BUSY, "session is busy");
    last_current_ = nil;
    has_frame_ = false;
    generated_count_ = 0;
    phase_ = NSFG_SESSION_NEEDS_RESET;
    return NSFG_STATUS_OK;
  }

  NSFGSessionStats Stats() const {
    return {phase_, reset_count_, generated_frame_total_, last_frame_id_,
            resource_allocations_, resource_allocations_after_warmup_,
            input_mismatches_, canary_mismatches_};
  }

  const std::string& last_error() const { return last_error_; }
  const std::vector<std::string>& last_checkpoints() const {
    return last_checkpoints_;
  }
  const std::vector<NSFGCommandSpan>& last_spans() const {
    return last_spans_;
  }
  NSFGTailResources last_tail_resources() const {
    const auto& e = last_exports_;
    return {(__bridge void*)e.motion_half8, e.motion_offset, (__bridge void*)e.candidate_a,
            (__bridge void*)e.candidate_b, e.candidate_offset, (__bridge void*)e.blend_rg16f,
            (__bridge void*)e.sigmoid_lut, e.sigmoid_offset};
  }

 private:
  static id<MTLTexture> TextureRef(void* value) {
    return (__bridge id<MTLTexture>)value;
  }

  NSFGStatus Fail(NSFGStatus status, std::string message) {
    last_error_ = std::move(message);
    return status;
  }

  void ClearError() { last_error_.clear(); }

  bool ValidTexture(id<MTLTexture> texture, MTLPixelFormat format,
                    MTLTextureUsage required_usage) const {
    if (texture == nil || texture.device != device_ || texture.width != 640 ||
        texture.height != 360 || texture.depth != 1 ||
        texture.pixelFormat != format)
      return false;
    return required_usage == MTLTextureUsageUnknown ||
           (texture.usage & required_usage) == required_usage;
  }

  NSFGStatus ValidateFrame(const NSFGFrameInfo& frame) {
    const auto current = TextureRef(frame.current_backbuffer_rgba8);
    const auto previous = TextureRef(frame.previous_backbuffer_rgba8);
    const auto motion = TextureRef(frame.motion_rg16f);
    const auto depth = TextureRef(frame.depth_r32f);
    const auto target = TextureRef(frame.generated_target_rgba8);
    if (!ValidTexture(current, MTLPixelFormatRGBA8Unorm,
                      MTLTextureUsageShaderRead) ||
        !ValidTexture(previous, MTLPixelFormatRGBA8Unorm,
                      MTLTextureUsageShaderRead) ||
        !ValidTexture(motion, MTLPixelFormatRG16Float,
                      MTLTextureUsageShaderRead) ||
        !ValidTexture(depth, MTLPixelFormatR32Float,
                      MTLTextureUsageShaderRead))
      return Fail(NSFG_STATUS_UNSUPPORTED_PROFILE, "input texture contract mismatch");
    if (phase_ == NSFG_SESSION_FAULTED && !frame.reset)
      return Fail(NSFG_STATUS_NEEDS_RESET, "faulted session requires reset");
    if (!frame.reset) {
      if (!has_frame_ || phase_ == NSFG_SESSION_NEEDS_RESET)
        return Fail(NSFG_STATUS_NEEDS_RESET, "non-reset frame before warmup");
      if (frame.frame_id != last_frame_id_ + 1u)
        return Fail(NSFG_STATUS_OUT_OF_ORDER_FRAME, "frame_id is not adjacent");
      if (previous != last_current_)
        return Fail(NSFG_STATUS_OUT_OF_ORDER_FRAME,
                    "previous backbuffer does not match session lineage");
      if (!ValidTexture(target, MTLPixelFormatRGBA8Unorm,
                        MTLTextureUsageUnknown))
        return Fail(NSFG_STATUS_UNSUPPORTED_PROFILE, "output texture contract mismatch");
      if (target == current || target == previous || target == motion ||
          target == depth)
        return Fail(NSFG_STATUS_INVALID_ARGUMENT, "output aliases an input");
    }
    return NSFG_STATUS_OK;
  }

  id<MTLBuffer> MakeBuffer(const std::vector<std::uint8_t>& payload) {
    id<MTLBuffer> result = [device_
        newBufferWithLength:payload.size() + 2 * kGuardBytes
                    options:MTLResourceStorageModeShared];
    if (result == nil) throw std::runtime_error("buffer allocation failed");
    ++resource_allocations_;
    std::memset(result.contents, contract::kCanaryValue, result.length);
    std::memcpy(static_cast<std::uint8_t*>(result.contents) + kGuardBytes,
                payload.data(), payload.size());
    return result;
  }

  id<MTLBuffer> MakeRawBuffer(const void* bytes, std::size_t size) {
    id<MTLBuffer> result = [device_ newBufferWithBytes:bytes
                                               length:size
                                              options:MTLResourceStorageModeShared];
    if (result == nil) throw std::runtime_error("parameter buffer allocation failed");
    ++resource_allocations_;
    return result;
  }

  id<MTLTexture> MakeTexture(MTLPixelFormat format, std::uint32_t width,
                             std::uint32_t height, MTLTextureUsage usage,
                             const std::vector<std::uint8_t>& initial,
                             std::size_t bytes_per_pixel) {
    if (initial.size() != std::size_t{width} * height * bytes_per_pixel)
      throw std::runtime_error("texture initialization size drift");
    auto* descriptor = [MTLTextureDescriptor
        texture2DDescriptorWithPixelFormat:format
                                      width:width
                                     height:height
                                  mipmapped:NO];
    descriptor.storageMode = MTLStorageModeShared;
    descriptor.usage = usage;
    id<MTLTexture> result = [device_ newTextureWithDescriptor:descriptor];
    if (result == nil) throw std::runtime_error("texture allocation failed");
    ++resource_allocations_;
    WriteTexture(result, initial, bytes_per_pixel);
    return result;
  }

  static void WriteTexture(id<MTLTexture> texture,
                           const std::vector<std::uint8_t>& bytes,
                           std::size_t bytes_per_pixel) {
    [texture replaceRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
               mipmapLevel:0
                 withBytes:bytes.data()
               bytesPerRow:texture.width * bytes_per_pixel];
  }

  static std::vector<std::uint8_t> ReadTexture(id<MTLTexture> texture,
                                                std::size_t bytes_per_pixel) {
    std::vector<std::uint8_t> result(texture.width * texture.height *
                                     bytes_per_pixel);
    [texture getBytes:result.data()
          bytesPerRow:texture.width * bytes_per_pixel
           fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
          mipmapLevel:0];
    return result;
  }

  static std::vector<std::uint8_t> BufferPayload(id<MTLBuffer> buffer,
                                                 std::size_t size) {
    const auto* begin = static_cast<const std::uint8_t*>(buffer.contents) +
                        kGuardBytes;
    return {begin, begin + size};
  }

  static std::vector<std::uint8_t> Join(
      std::initializer_list<std::vector<std::uint8_t>> values) {
    std::size_t size = 0;
    for (const auto& value : values) size += value.size();
    std::vector<std::uint8_t> result;
    result.reserve(size);
    for (const auto& value : values)
      result.insert(result.end(), value.begin(), value.end());
    return result;
  }

  void RecordCheckpoint(std::vector<std::uint8_t> bytes) {
    if (audit_enabled_) last_checkpoints_.push_back(Sha256(bytes));
  }

  static void ResetPayload(id<MTLBuffer> buffer,
                           const std::vector<std::uint8_t>& payload) {
    std::memset(buffer.contents, contract::kCanaryValue, buffer.length);
    std::memcpy(static_cast<std::uint8_t*>(buffer.contents) + kGuardBytes,
                payload.data(), payload.size());
  }

  static std::size_t CanaryMismatches(id<MTLBuffer> buffer,
                                      std::size_t payload_size) {
    const auto* bytes = static_cast<const std::uint8_t*>(buffer.contents);
    std::size_t result = 0;
    for (std::size_t index = 0; index < kGuardBytes; ++index) {
      result += bytes[index] != contract::kCanaryValue;
      result += bytes[kGuardBytes + payload_size + index] !=
                contract::kCanaryValue;
    }
    return result;
  }

  void LoadPipelines() {
    l3_ = MakePipeline(device_, libs_ / "l3_l13.metallib",
                       "fg_early_l3_l13_mode2");
    l4_ = MakePipeline(device_, libs_ / "l4_l14.metallib", "fg_early_l4_l14");
    l5_high_ = MakePipeline(device_, libs_ / "l5_l15.metallib",
                            "fg_early_l5_l15_high");
    l5_low_ = MakePipeline(device_, libs_ / "l5_l15.metallib",
                           "fg_early_l5_l15_low");
    l5_pack_ = MakePipeline(device_, libs_ / "l5_l15.metallib",
                            "fg_early_l5_l15_pack");
    l6_ = MakePipeline(device_, libs_ / "l6_l16.metallib", "fg_early_l6_l16");
    l7_ = MakePipeline(device_, libs_ / "l7_l17.metallib", "fg_early_l7_l17");
    l8_ = MakePipeline(device_, libs_ / "l8_l18.metallib", "fg_early_l8_l18");
    l9_ = MakePipeline(device_, libs_ / "l9.metallib", "fg_early_l9");
    status_ = MakePipeline(device_, libs_ / "l10_l12.metallib",
                           "fg_early_status_write");
    l11_ = MakePipeline(device_, libs_ / "l11.metallib", "fg_early_l11");
    l19_surfaces_ = MakePipeline(device_, libs_ / "l19.metallib",
                                 "fg_early_l19_surfaces");
    l19_tensor_ = MakePipeline(device_, libs_ / "l19.metallib",
                               "fg_early_l19_tensor");
  }

  void LoadArchitectureData(const NSFGCreateInfo& info) {
    rom_bytes_ = ReadFile(info.rcp_rom_path);
    if (rom_bytes_.size() != kRomBytes ||
        Sha256(rom_bytes_) !=
            "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750")
      throw std::runtime_error("RCP ROM identity mismatch");
    sigmoid_bytes_ = ReadFile(info.sigmoid_lut_path);
    if (sigmoid_bytes_.size() != 262144u ||
        Sha256(sigmoid_bytes_) !=
            "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850")
      throw std::runtime_error("sigmoid LUT identity mismatch");
    ex2_unit_bytes_ = ReadFile(info.ex2_unit_table_path);
    ex2_fraction_bytes_ = ReadFile(info.ex2_fraction_table_path);
    if (ex2_mode_ == NSFG_EX2_REFERENCE) {
      if (ex2_unit_bytes_.size() != 67108868u ||
          Sha256(ex2_unit_bytes_) !=
              "40088ff59b58c4018114b4a2dd1fb5b124488b91e7cad79bdf3bf143bf267b75" ||
          ex2_fraction_bytes_.size() != 33554436u ||
          Sha256(ex2_fraction_bytes_) !=
              "f7cfb2fd9df97ae01bc747b866e4928a79311e8443700f938859f43772f73d86")
        throw std::runtime_error("reference EX2 table identity mismatch");
    } else if (ex2_mode_ == NSFG_EX2_COMPACT) {
      if (ex2_unit_bytes_.size() != 4194305u ||
          Sha256(ex2_unit_bytes_) !=
              "60c2ae1a0869ce6f10ba093a33def68ecdbf4ba0d789cd186289586b63b9f234" ||
          ex2_fraction_bytes_.size() != 2097153u ||
          Sha256(ex2_fraction_bytes_) !=
              "76bacb1adf8e53accdd6a40a64940c073f001e091232e9119b155f2ab25f1a13")
        throw std::runtime_error("compact EX2 correction identity mismatch");
    } else {
      throw std::invalid_argument("unsupported EX2 mode");
    }
  }

  void AllocateResources() {
    const auto poisoned = [this](std::size_t size) {
      return std::vector<std::uint8_t>(size, poison_);
    };
    rom_buffer_ = MakeBuffer(rom_bytes_);
    ex2_unit_buffer_ = MakeBuffer(ex2_unit_bytes_);
    ex2_fraction_buffer_ = MakeBuffer(ex2_fraction_bytes_);
    history_a_ = MakeTexture(MTLPixelFormatRGBA16Float, 640, 360,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             contract::InitialHistory(), 8);
    history_b_ = MakeTexture(MTLPixelFormatRGBA16Float, 640, 360,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             poisoned(contract::kHistoryBytes), 8);
    feature_a_ = MakeTexture(MTLPixelFormatRGBA16Float, 320, 180,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             poisoned(contract::kHalfColorBytes), 8);
    feature_b_ = MakeTexture(MTLPixelFormatRGBA16Float, 320, 180,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             poisoned(contract::kHalfColorBytes), 8);
    shared_metric_ = MakeTexture(MTLPixelFormatR16Float, 320, 180,
                                 MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                 poisoned(contract::kHalfScalarBytes), 2);
    flow_ = MakeTexture(MTLPixelFormatRG32Float, 320, 180,
                        MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                        poisoned(contract::kDepthBytes * 2u), 8);
    depth_a_ = MakeTexture(MTLPixelFormatR32Uint, 320, 180,
                           MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                           poisoned(contract::kDepthBytes), 4);
    depth_b_ = MakeTexture(MTLPixelFormatR32Uint, 320, 180,
                           MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                           contract::InitialGeneratedDepth(), 4);
    mask_a_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                          poisoned(contract::kMaskBytes), 1);
    mask_b_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                          contract::InitialGeneratedMask(), 1);
    working_vector_ = MakeTexture(MTLPixelFormatRG16Float, 320, 180,
                                  MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                  poisoned(contract::kVectorBytes), 4);
    working_mask_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                poisoned(contract::kMaskBytes), 1);
    scatter_a_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             std::vector<std::uint8_t>(contract::kMaskBytes, 255), 1);
    scatter_b_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                             MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                             std::vector<std::uint8_t>(contract::kMaskBytes, 255), 1);
    projection_a_ = MakeTexture(MTLPixelFormatRG16Float, 320, 180,
                                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                poisoned(contract::kVectorBytes), 4);
    projection_b_ = MakeTexture(MTLPixelFormatRG16Float, 320, 180,
                                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                poisoned(contract::kVectorBytes), 4);
    allocation29_ = MakeTexture(MTLPixelFormatR16Float, 320, 180,
                                MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
                                contract::InitialAllocation29(), 2);
    auxiliary_mask_ = MakeTexture(MTLPixelFormatR8Unorm, 320, 180,
                                  MTLTextureUsageShaderRead |
                                      MTLTextureUsageShaderWrite,
                                  poisoned(contract::kAuxMaskBytes), 1);
    reset_present_ = MakeTexture(MTLPixelFormatRGBA8Unorm, 640, 360,
                                 MTLTextureUsageShaderWrite,
                                 poisoned(contract::kSourceBytes), 4);
    model_input_ = MakeBuffer(poisoned(contract::kModelInputBytes));
    atomic_ = MakeBuffer(contract::InitialAtomic());
    l5_high_buffer_ = MakeBuffer(std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    l5_low_buffer_ = MakeBuffer(std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    for (std::size_t index = 0; index < pyramid_.size(); ++index)
      pyramid_[index] = MakeBuffer(poisoned(kPyramidBytes[index]));
    vector_buffer_ = MakeBuffer(poisoned(contract::kVectorBytes));
    status_buffer_ = MakeBuffer(poisoned(4));
    resource_allocations_after_warmup_ = 0;
  }

  void RestoreInitialState() {
    const auto poisoned = [this](std::size_t size) {
      return std::vector<std::uint8_t>(size, poison_);
    };
    WriteTexture(history_a_, contract::InitialHistory(), 8);
    WriteTexture(history_b_, poisoned(contract::kHistoryBytes), 8);
    WriteTexture(feature_a_, poisoned(contract::kHalfColorBytes), 8);
    WriteTexture(feature_b_, poisoned(contract::kHalfColorBytes), 8);
    WriteTexture(shared_metric_, poisoned(contract::kHalfScalarBytes), 2);
    WriteTexture(flow_, poisoned(contract::kDepthBytes * 2u), 8);
    WriteTexture(depth_a_, poisoned(contract::kDepthBytes), 4);
    WriteTexture(depth_b_, contract::InitialGeneratedDepth(), 4);
    WriteTexture(mask_a_, poisoned(contract::kMaskBytes), 1);
    WriteTexture(mask_b_, contract::InitialGeneratedMask(), 1);
    WriteTexture(working_vector_, poisoned(contract::kVectorBytes), 4);
    WriteTexture(working_mask_, poisoned(contract::kMaskBytes), 1);
    WriteTexture(scatter_a_, std::vector<std::uint8_t>(contract::kMaskBytes, 255), 1);
    WriteTexture(scatter_b_, std::vector<std::uint8_t>(contract::kMaskBytes, 255), 1);
    WriteTexture(projection_a_, poisoned(contract::kVectorBytes), 4);
    WriteTexture(projection_b_, poisoned(contract::kVectorBytes), 4);
    WriteTexture(allocation29_, contract::InitialAllocation29(), 2);
    WriteTexture(auxiliary_mask_, poisoned(contract::kAuxMaskBytes), 1);
    WriteTexture(reset_present_, poisoned(contract::kSourceBytes), 4);
    ResetPayload(model_input_, poisoned(contract::kModelInputBytes));
    ResetPayload(atomic_, contract::InitialAtomic());
    ResetPayload(l5_high_buffer_, std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    ResetPayload(l5_low_buffer_, std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    for (std::size_t index = 0; index < pyramid_.size(); ++index)
      ResetPayload(pyramid_[index], poisoned(kPyramidBytes[index]));
    ResetPayload(vector_buffer_, poisoned(contract::kVectorBytes));
    ResetPayload(status_buffer_, poisoned(4));
  }

  void EncodeL3(id<MTLTexture> source, id<MTLTexture> history,
                id<MTLTexture> feature, id<MTLTexture> next_history) {
    const L3Parameters parameters{640, 360, 320, 180};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l3_];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:history atIndex:1];
    [encoder setTexture:feature atIndex:2];
    [encoder setTexture:shared_metric_ atIndex:3];
    [encoder setTexture:next_history atIndex:4];
    [encoder dispatchThreads:MTLSizeMake(320, 180, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 8, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL4(bool reset, id<MTLTexture> motion, id<MTLTexture> depth,
                id<MTLTexture> output_depth) {
    const L4Parameters parameters{reset ? 127u : 0u, 640, 360, 320, 180};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l4_];
    [encoder setTexture:motion atIndex:0];
    [encoder setTexture:depth atIndex:1];
    [encoder setTexture:flow_ atIndex:2];
    [encoder setTexture:output_depth atIndex:3];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(320, 180, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL5(id<MTLTexture> depth, id<MTLTexture> mask) {
    ResetPayload(l5_high_buffer_, std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    ResetPayload(l5_low_buffer_, std::vector<std::uint8_t>(contract::kDepthBytes, 0xff));
    const L5Parameters parameters{320, 180, 144.0f, 12.96f, 1.0f, 0};
    for (id<MTLComputePipelineState> pipeline in @[ l5_high_, l5_low_ ]) {
      id<MTLCommandBuffer> command = [queue_ commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setTexture:flow_ atIndex:0];
      [encoder setTexture:depth atIndex:1];
      [encoder setTexture:mask atIndex:2];
      [encoder setBuffer:l5_high_buffer_ offset:kGuardBytes atIndex:0];
      [encoder setBuffer:l5_low_buffer_ offset:kGuardBytes atIndex:1];
      [encoder setBuffer:rom_buffer_ offset:kGuardBytes atIndex:2];
      [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
      [encoder dispatchThreads:MTLSizeMake(320, 180, 1)
          threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
      [encoder endEncoding];
      Wait(command);
    }
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l5_pack_];
    [encoder setBuffer:l5_high_buffer_ offset:kGuardBytes atIndex:0];
    [encoder setBuffer:l5_low_buffer_ offset:kGuardBytes atIndex:1];
    [encoder setBuffer:atomic_ offset:kGuardBytes atIndex:2];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:3];
    [encoder dispatchThreads:MTLSizeMake(320u * 180u, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL6(id<MTLTexture> opposite_depth,
                id<MTLTexture> opposite_mask) {
    std::array<std::uint8_t, 224> parameters{};
    Store64(&parameters, 24, atomic_.gpuAddress + kGuardBytes);
    for (std::size_t index = 0; index < pyramid_.size(); ++index)
      Store64(&parameters, 48 + index * 8,
              pyramid_[index].gpuAddress + kGuardBytes);
    Store32(&parameters, 128, std::bit_cast<std::uint32_t>(1.0f / 640.0f));
    Store32(&parameters, 132, std::bit_cast<std::uint32_t>(1.0f / 360.0f));
    Store32(&parameters, 136, 320);
    Store32(&parameters, 140, 180);
    Store32(&parameters, 144, std::bit_cast<std::uint32_t>(40.0f));
    Store32(&parameters, 148, std::bit_cast<std::uint32_t>(1.0f));
    Store32(&parameters, 152, std::bit_cast<std::uint32_t>(600.0f));
    const std::array<float, 16> identity = {
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
    std::memcpy(parameters.data() + 160, identity.data(), 64);
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l6_];
    [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
    [encoder setBuffer:rom_buffer_ offset:kGuardBytes atIndex:1];
    [encoder setTexture:flow_ atIndex:0];
    [encoder setTexture:opposite_depth atIndex:1];
    [encoder setTexture:opposite_mask atIndex:2];
    [encoder setTexture:working_vector_ atIndex:3];
    [encoder setTexture:working_mask_ atIndex:4];
    [encoder useResource:atomic_ usage:MTLResourceUsageRead];
    for (id<MTLBuffer> item : pyramid_)
      [encoder useResource:item usage:MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void BlitTextureToBuffer(id<MTLTexture> texture, id<MTLBuffer> buffer,
                           std::size_t bytes_per_pixel) {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    [encoder copyFromTexture:texture
                 sourceSlice:0
                 sourceLevel:0
                sourceOrigin:MTLOriginMake(0, 0, 0)
                  sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                    toBuffer:buffer
           destinationOffset:kGuardBytes
      destinationBytesPerRow:texture.width * bytes_per_pixel
    destinationBytesPerImage:texture.width * texture.height * bytes_per_pixel];
    [encoder endEncoding];
    Wait(command);
  }

  void BlitBufferToTexture(id<MTLBuffer> buffer, id<MTLTexture> texture,
                           std::size_t bytes_per_pixel) {
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLBlitCommandEncoder> encoder = [command blitCommandEncoder];
    [encoder copyFromBuffer:buffer
               sourceOffset:kGuardBytes
          sourceBytesPerRow:texture.width * bytes_per_pixel
        sourceBytesPerImage:texture.width * texture.height * bytes_per_pixel
                 sourceSize:MTLSizeMake(texture.width, texture.height, 1)
                  toTexture:texture
           destinationSlice:0
           destinationLevel:0
          destinationOrigin:MTLOriginMake(0, 0, 0)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL7() {
    BlitTextureToBuffer(working_vector_, vector_buffer_, 4);
    std::array<std::uint8_t, 72> parameters{};
    Store64(&parameters, 8, pyramid_[1].gpuAddress + kGuardBytes);
    Store64(&parameters, 16, pyramid_[2].gpuAddress + kGuardBytes);
    Store64(&parameters, 24, pyramid_[3].gpuAddress + kGuardBytes);
    Store64(&parameters, 32, pyramid_[4].gpuAddress + kGuardBytes);
    Store32(&parameters, 40, 320);
    Store32(&parameters, 44, 180);
    Store32(&parameters, 48, 1);
    Store64(&parameters, 56, vector_buffer_.gpuAddress + kGuardBytes);
    Store64(&parameters, 64, pyramid_[0].gpuAddress + kGuardBytes);
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l7_];
    [encoder setBytes:parameters.data() length:parameters.size() atIndex:0];
    [encoder setBuffer:rom_buffer_ offset:kGuardBytes atIndex:1];
    for (id<MTLBuffer> item : pyramid_)
      [encoder useResource:item usage:MTLResourceUsageRead];
    [encoder useResource:vector_buffer_ usage:MTLResourceUsageWrite];
    [encoder dispatchThreadgroups:MTLSizeMake(20, 12, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
    [encoder endEncoding];
    Wait(command);
    BlitBufferToTexture(vector_buffer_, working_vector_, 4);
  }

  void EncodeL8(id<MTLTexture> scatter) {
    const L8Parameters parameters{320, 180, 0};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l8_];
    [encoder setTexture:working_vector_ atIndex:0];
    [encoder setTexture:scatter atIndex:1];
    [encoder setBuffer:rom_buffer_ offset:kGuardBytes atIndex:0];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(320, 180, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL9() {
    const L9Parameters parameters{320, 180};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l9_];
    [encoder setTexture:feature_a_ atIndex:0];
    [encoder setTexture:projection_a_ atIndex:1];
    [encoder setBuffer:rom_buffer_ offset:kGuardBytes atIndex:0];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(320, 180, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeStatus(std::uint32_t value) {
    const StatusParameters parameters{1, value};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:status_];
    [encoder setBuffer:status_buffer_ offset:kGuardBytes atIndex:0];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:1];
    [encoder dispatchThreads:MTLSizeMake(1, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL11(id<MTLTexture> source) {
    const L11Parameters parameters{0, 0, 0, 0, 640, 360};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:l11_];
    [encoder setTexture:source atIndex:0];
    [encoder setTexture:reset_present_ atIndex:1];
    [encoder setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [encoder dispatchThreads:MTLSizeMake(640, 360, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [encoder endEncoding];
    Wait(command);
  }

  void EncodeL19(const std::array<id<MTLTexture>, 10>& inputs,
                 id<MTLTexture> vector_output) {
    const L19Parameters parameters{320, 180, 160, 96, -6};
    id<MTLCommandBuffer> command = [queue_ commandBuffer];
    id<MTLComputeCommandEncoder> surface = [command computeCommandEncoder];
    [surface setComputePipelineState:l19_surfaces_];
    for (NSUInteger index = 0; index < inputs.size(); ++index)
      [surface setTexture:inputs[index] atIndex:index];
    [surface setTexture:auxiliary_mask_ atIndex:10];
    [surface setTexture:vector_output atIndex:11];
    [surface setBytes:&parameters length:sizeof(parameters) atIndex:0];
    [surface dispatchThreads:MTLSizeMake(320, 180, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [surface endEncoding];
    id<MTLComputeCommandEncoder> tensor = [command computeCommandEncoder];
    [tensor setComputePipelineState:l19_tensor_];
    for (NSUInteger index = 0; index < inputs.size(); ++index)
      [tensor setTexture:inputs[index] atIndex:index];
    [tensor setBuffer:model_input_ offset:kGuardBytes atIndex:0];
    [tensor setBytes:&parameters length:sizeof(parameters) atIndex:1];
    [tensor dispatchThreads:MTLSizeMake(160, 96, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [tensor endEncoding];
    Wait(command);
  }

  void ExecuteReset(id<MTLTexture> current, id<MTLTexture> motion,
                    id<MTLTexture> depth) {
    EncodeL3(current, history_a_, feature_a_, history_b_);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(feature_a_, 8), ReadTexture(shared_metric_, 2),
        ReadTexture(history_b_, 8)}));
    EncodeL4(true, motion, depth, depth_a_);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(flow_, 8), ReadTexture(depth_a_, 4)}));
    EncodeL5(depth_a_, mask_a_);
    if (audit_enabled_) RecordCheckpoint(Join({
        BufferPayload(atomic_, contract::kAtomicBytes),
        ReadTexture(mask_a_, 1)}));
    EncodeL6(depth_b_, mask_b_);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(working_vector_, 4), ReadTexture(working_mask_, 1),
        BufferPayload(pyramid_[0], kPyramidBytes[0]),
        BufferPayload(pyramid_[1], kPyramidBytes[1]),
        BufferPayload(pyramid_[2], kPyramidBytes[2]),
        BufferPayload(pyramid_[3], kPyramidBytes[3]),
        BufferPayload(pyramid_[4], kPyramidBytes[4])}));
    EncodeL7();
    if (audit_enabled_) RecordCheckpoint(ReadTexture(working_vector_, 4));
    EncodeL8(scatter_a_);
    if (audit_enabled_) RecordCheckpoint(ReadTexture(scatter_a_, 1));
    EncodeL9();
    if (audit_enabled_) RecordCheckpoint(ReadTexture(projection_a_, 4));
    EncodeStatus(1);
    if (audit_enabled_) RecordCheckpoint(BufferPayload(status_buffer_, 4));
    EncodeL11(current);
    if (audit_enabled_) RecordCheckpoint(ReadTexture(reset_present_, 4));
  }

  void ExecuteGenerated(id<MTLTexture> current, id<MTLTexture> motion,
                        id<MTLTexture> depth, id<MTLTexture> target,
                        std::uint64_t frame_id) {
    EncodeStatus(0);
    if (audit_enabled_) RecordCheckpoint(BufferPayload(status_buffer_, 4));
    const bool odd = ((generated_count_ + 1u) & 1u) != 0u;
    id<MTLTexture> previous_history = odd ? history_b_ : history_a_;
    id<MTLTexture> next_history = odd ? history_a_ : history_b_;
    id<MTLTexture> current_feature = odd ? feature_b_ : feature_a_;
    id<MTLTexture> previous_feature = odd ? feature_a_ : feature_b_;
    id<MTLTexture> current_depth = odd ? depth_b_ : depth_a_;
    id<MTLTexture> previous_depth = odd ? depth_a_ : depth_b_;
    id<MTLTexture> current_mask = odd ? mask_b_ : mask_a_;
    id<MTLTexture> previous_mask = odd ? mask_a_ : mask_b_;
    id<MTLTexture> current_scatter = odd ? scatter_b_ : scatter_a_;
    id<MTLTexture> previous_scatter = odd ? scatter_a_ : scatter_b_;
    id<MTLTexture> current_projection = odd ? projection_b_ : projection_a_;
    id<MTLTexture> previous_projection = odd ? projection_a_ : projection_b_;
    EncodeL3(current, previous_history, current_feature, next_history);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(current_feature, 8), ReadTexture(shared_metric_, 2),
        ReadTexture(next_history, 8)}));
    EncodeL4(false, motion, depth, current_depth);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(flow_, 8), ReadTexture(current_depth, 4)}));
    EncodeL5(current_depth, current_mask);
    if (audit_enabled_) RecordCheckpoint(Join({
        BufferPayload(atomic_, contract::kAtomicBytes),
        ReadTexture(current_mask, 1)}));
    EncodeL6(previous_depth, previous_mask);
    if (audit_enabled_) RecordCheckpoint(Join({
        ReadTexture(working_vector_, 4), ReadTexture(working_mask_, 1),
        BufferPayload(pyramid_[0], kPyramidBytes[0]),
        BufferPayload(pyramid_[1], kPyramidBytes[1]),
        BufferPayload(pyramid_[2], kPyramidBytes[2]),
        BufferPayload(pyramid_[3], kPyramidBytes[3]),
        BufferPayload(pyramid_[4], kPyramidBytes[4])}));
    EncodeL7();
    if (audit_enabled_) RecordCheckpoint(ReadTexture(working_vector_, 4));
    EncodeL8(current_scatter);
    if (audit_enabled_) RecordCheckpoint(ReadTexture(current_scatter, 1));
    const std::array<id<MTLTexture>, 10> inputs = {
        previous_feature, current_feature, current_scatter, previous_scatter,
        previous_projection, shared_metric_, allocation29_, flow_,
        working_vector_, working_mask_};
    EncodeL19(inputs, current_projection);
    if (audit_enabled_) RecordCheckpoint(Join({
        BufferPayload(model_input_, contract::kModelInputBytes),
        ReadTexture(auxiliary_mask_, 1), ReadTexture(current_projection, 4)}));
    const NSFGTailLiveResources live_resources = {
        auxiliary_mask_, flow_, working_vector_, current_depth, previous_depth,
        previous_history, next_history, current_projection};
    RunTail(target, frame_id, live_resources);
  }

  void RunTail(id<MTLTexture> target, std::uint64_t frame_id,
               const NSFGTailLiveResources& live_resources) {
    const auto root = fs::temp_directory_path() /
        ("neuralstudio-fg-runtime-" + std::to_string(getpid())) /
        ("frame-" + std::to_string(frame_id));
    std::string program = "fg-runtime-tail";
    std::string libs = libs_.string();
    std::string model = model_.string();
    std::string rom = rom_path_.string();
    std::string sigmoid = sigmoid_path_.string();
    std::string output = root.string();
    std::string run = "fg-runtime-frame-" + std::to_string(frame_id);
    std::ostringstream poison_text;
    poison_text << std::hex << unsigned(poison_);
    std::string poison = poison_text.str();
    std::array<const char*, 8> arguments = {
        program.c_str(), libs.c_str(), model.c_str(), rom.c_str(),
        sigmoid.c_str(), output.c_str(), run.c_str(), poison.c_str()};
    FgConnectedSetMacLivePrefixInput(model_input_);
    FgConnectedSetMacAllocation29(allocation29_);
    FgConnectedSetMacEx2Tables(ex2_unit_buffer_, ex2_fraction_buffer_);
    NSFGSetTailLiveResources(&live_resources);
    NSFGSetTailContext(
        device_, queue_, audit_enabled_,
        auxiliary_mode_ == NSFG_AUXILIARY_LIVE_EARLY_RESOURCES);
    NSFGSetTailOutputTarget(target);
    const int status = FgRuntimeFullTailMain(
        static_cast<int>(arguments.size()), arguments.data());
    NSFGSetTailOutputTarget(nil);
    last_exports_ = NSFGTailLastExports();
    const auto tail_allocations = NSFGTailLastResourceAllocations();
    std::vector<std::string> tail_checkpoints;
    if (status == 0 && audit_enabled_) {
      const auto count = NSFGTailCheckpointCount();
      tail_checkpoints.reserve(count);
      for (std::size_t index = 0; index < count; ++index) {
        const char* checkpoint = NSFGTailCheckpoint(index);
        if (checkpoint != nullptr) tail_checkpoints.emplace_back(checkpoint);
      }
    }
    NSFGSetTailContext(nil, nil, false, false);
    FgConnectedSetMacLivePrefixInput(nil);
    FgConnectedSetMacAllocation29(nil);
    FgConnectedSetMacEx2Tables(nil, nil);
    NSFGSetTailLiveResources(nullptr);
    if (status != 0) throw std::runtime_error("selected L20-L87 runtime tail failed");
    if (audit_enabled_ && tail_checkpoints.size() != 68)
      throw std::runtime_error("selected L20-L87 audit checkpoint count drift");
    last_checkpoints_.insert(last_checkpoints_.end(), tail_checkpoints.begin(),
                             tail_checkpoints.end());
    resource_allocations_ += tail_allocations;
    if (generated_count_ != 0) resource_allocations_after_warmup_ += tail_allocations;
  }

  std::size_t AuditCanaries() const {
    std::size_t result = 0;
    result += CanaryMismatches(rom_buffer_, rom_bytes_.size());
    result += CanaryMismatches(ex2_unit_buffer_, ex2_unit_bytes_.size());
    result += CanaryMismatches(ex2_fraction_buffer_, ex2_fraction_bytes_.size());
    result += CanaryMismatches(model_input_, contract::kModelInputBytes);
    result += CanaryMismatches(atomic_, contract::kAtomicBytes);
    result += CanaryMismatches(l5_high_buffer_, contract::kDepthBytes);
    result += CanaryMismatches(l5_low_buffer_, contract::kDepthBytes);
    result += CanaryMismatches(vector_buffer_, contract::kVectorBytes);
    result += CanaryMismatches(status_buffer_, 4);
    for (std::size_t index = 0; index < pyramid_.size(); ++index)
      result += CanaryMismatches(pyramid_[index], kPyramidBytes[index]);
    return result;
  }

  id<MTLDevice> device_ = nil;
  id<MTLCommandQueue> queue_ = nil;
  bool metal4_ = false;
  NSFGTailExports last_exports_{};
  fs::path libs_;
  fs::path model_;
  fs::path rom_path_;
  fs::path sigmoid_path_;
  NSFGEx2Mode ex2_mode_ = NSFG_EX2_REFERENCE;
  NSFGAuxiliaryMode auxiliary_mode_ = NSFG_AUXILIARY_INTERNAL_SMALL_FLOW;
  std::uint8_t poison_ = 0;
  bool audit_enabled_ = false;
  bool profiling_enabled_ = false;
  NSFGSessionPhase phase_ = NSFG_SESSION_UNINITIALIZED;
  std::uint64_t reset_count_ = 0;
  std::uint64_t generated_count_ = 0;
  std::uint64_t generated_frame_total_ = 0;
  std::uint64_t last_frame_id_ = 0;
  std::uint64_t resource_allocations_ = 0;
  std::uint64_t resource_allocations_after_warmup_ = 0;
  std::uint64_t input_mismatches_ = 0;
  std::uint64_t canary_mismatches_ = 0;
  bool has_frame_ = false;
  id<MTLTexture> last_current_ = nil;
  std::string last_error_;
  std::vector<std::string> last_checkpoints_;
  std::vector<NSFGCommandSpan> last_spans_;
  std::vector<std::uint8_t> rom_bytes_;
  std::vector<std::uint8_t> sigmoid_bytes_;
  std::vector<std::uint8_t> ex2_unit_bytes_;
  std::vector<std::uint8_t> ex2_fraction_bytes_;
  id<MTLComputePipelineState> l3_ = nil;
  id<MTLComputePipelineState> l4_ = nil;
  id<MTLComputePipelineState> l5_high_ = nil;
  id<MTLComputePipelineState> l5_low_ = nil;
  id<MTLComputePipelineState> l5_pack_ = nil;
  id<MTLComputePipelineState> l6_ = nil;
  id<MTLComputePipelineState> l7_ = nil;
  id<MTLComputePipelineState> l8_ = nil;
  id<MTLComputePipelineState> l9_ = nil;
  id<MTLComputePipelineState> status_ = nil;
  id<MTLComputePipelineState> l11_ = nil;
  id<MTLComputePipelineState> l19_surfaces_ = nil;
  id<MTLComputePipelineState> l19_tensor_ = nil;
  id<MTLBuffer> rom_buffer_ = nil;
  id<MTLBuffer> ex2_unit_buffer_ = nil;
  id<MTLBuffer> ex2_fraction_buffer_ = nil;
  id<MTLBuffer> model_input_ = nil;
  id<MTLBuffer> atomic_ = nil;
  id<MTLBuffer> l5_high_buffer_ = nil;
  id<MTLBuffer> l5_low_buffer_ = nil;
  std::array<id<MTLBuffer>, 5> pyramid_{};
  id<MTLBuffer> vector_buffer_ = nil;
  id<MTLBuffer> status_buffer_ = nil;
  id<MTLTexture> history_a_ = nil;
  id<MTLTexture> history_b_ = nil;
  id<MTLTexture> feature_a_ = nil;
  id<MTLTexture> feature_b_ = nil;
  id<MTLTexture> shared_metric_ = nil;
  id<MTLTexture> flow_ = nil;
  id<MTLTexture> depth_a_ = nil;
  id<MTLTexture> depth_b_ = nil;
  id<MTLTexture> mask_a_ = nil;
  id<MTLTexture> mask_b_ = nil;
  id<MTLTexture> working_vector_ = nil;
  id<MTLTexture> working_mask_ = nil;
  id<MTLTexture> scatter_a_ = nil;
  id<MTLTexture> scatter_b_ = nil;
  id<MTLTexture> projection_a_ = nil;
  id<MTLTexture> projection_b_ = nil;
  id<MTLTexture> allocation29_ = nil;
  id<MTLTexture> auxiliary_mask_ = nil;
  id<MTLTexture> reset_present_ = nil;
};

}  // namespace

struct NSFGSession {
  std::unique_ptr<RuntimeCore> core;
  bool owns_process_gate = false;
};

NSFGStatus NSFGSessionCreate(const NSFGCreateInfo* info,
                             NSFGSession** out_session) {
  if (out_session == nullptr) return NSFG_STATUS_INVALID_ARGUMENT;
  *out_session = nullptr;
  if (info == nullptr || info->metal_device == nullptr ||
      (info->metal_command_queue == nullptr && !info->metal4) ||
      info->metallib_directory == nullptr || info->model_directory == nullptr ||
      info->rcp_rom_path == nullptr || info->sigmoid_lut_path == nullptr ||
      info->ex2_unit_table_path == nullptr ||
      info->ex2_fraction_table_path == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  if (g_session_live.test_and_set()) return NSFG_STATUS_BUSY;
  try {
    auto session = std::make_unique<NSFGSession>();
    session->owns_process_gate = true;
    session->core = std::make_unique<RuntimeCore>(*info);
    *out_session = session.release();
    return NSFG_STATUS_OK;
  } catch (const std::invalid_argument&) {
    g_session_live.clear();
    return NSFG_STATUS_UNSUPPORTED_PROFILE;
  } catch (...) {
    g_session_live.clear();
    return NSFG_STATUS_INTERNAL_ERROR;
  }
}

NSFGStatus NSFGSessionEvaluate(NSFGSession* session,
                               const NSFGFrameInfo* frame,
                               NSFGFrameResult* result) {
  if (session == nullptr || session->core == nullptr || frame == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  if (result != nullptr) *result = {};
  return session->core->Evaluate(*frame, result);
}

NSFGStatus NSFGSessionCopyLastTailResources(const NSFGSession* session,
                                            NSFGTailResources* resources) {
  if (session == nullptr || session->core == nullptr || resources == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  *resources = session->core->last_tail_resources();
  return resources->motion_half8_buffer == nullptr ? NSFG_STATUS_NEEDS_RESET : NSFG_STATUS_OK;
}

NSFGStatus NSFGSessionEvaluateMetal4(NSFGSession* session,
                               const NSFGFrameInfo* frame,
                               void* mtl4_command_buffer,
                                     NSFGFrameResult* result) {
  if (session == nullptr || session->core == nullptr || frame == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  if (result != nullptr) *result = {};
  return session->core->EvaluateMetal4(*frame, mtl4_command_buffer, result);
}

NSFGStatus NSFGSessionInvalidate(NSFGSession* session) {
  if (session == nullptr || session->core == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  return session->core->Invalidate();
}

NSFGStatus NSFGSessionGetStats(const NSFGSession* session,
                               NSFGSessionStats* stats) {
  if (session == nullptr || session->core == nullptr || stats == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  *stats = session->core->Stats();
  return NSFG_STATUS_OK;
}

size_t NSFGSessionLastCheckpointCount(const NSFGSession* session) {
  return session != nullptr && session->core != nullptr
             ? session->core->last_checkpoints().size()
             : 0;
}

NSFGStatus NSFGSessionCopyLastCheckpoint(const NSFGSession* session,
                                         size_t index, char* sha256,
                                         size_t capacity) {
  if (session == nullptr || session->core == nullptr || sha256 == nullptr ||
      capacity < 65)
    return NSFG_STATUS_INVALID_ARGUMENT;
  const auto& checkpoints = session->core->last_checkpoints();
  if (index >= checkpoints.size()) return NSFG_STATUS_INVALID_ARGUMENT;
  std::memcpy(sha256, checkpoints[index].c_str(), 65);
  return NSFG_STATUS_OK;
}

size_t NSFGSessionLastCommandSpanCount(const NSFGSession* session) {
  return session != nullptr && session->core != nullptr
             ? session->core->last_spans().size()
             : 0;
}

NSFGStatus NSFGSessionCopyLastCommandSpan(const NSFGSession* session,
                                          size_t index,
                                          NSFGCommandSpan* span) {
  if (session == nullptr || session->core == nullptr || span == nullptr)
    return NSFG_STATUS_INVALID_ARGUMENT;
  const auto& spans = session->core->last_spans();
  if (index >= spans.size()) return NSFG_STATUS_INVALID_ARGUMENT;
  *span = spans[index];
  return NSFG_STATUS_OK;
}

const char* NSFGSessionLastError(const NSFGSession* session) {
  if (session == nullptr || session->core == nullptr) return "invalid session";
  return session->core->last_error().c_str();
}

void NSFGSessionDestroy(NSFGSession* session) {
  if (session == nullptr) return;
  const bool owns_process_gate = session->owns_process_gate;
  delete session;
  if (owns_process_gate) g_session_live.clear();
}

const char* NSFGStatusName(NSFGStatus status) {
  switch (status) {
    case NSFG_STATUS_OK: return "ok";
    case NSFG_STATUS_INVALID_ARGUMENT: return "invalid_argument";
    case NSFG_STATUS_UNSUPPORTED_PROFILE: return "unsupported_profile";
    case NSFG_STATUS_NEEDS_RESET: return "needs_reset";
    case NSFG_STATUS_OUT_OF_ORDER_FRAME: return "out_of_order_frame";
    case NSFG_STATUS_BUSY: return "busy";
    case NSFG_STATUS_DEVICE_ERROR: return "device_error";
    case NSFG_STATUS_INTERNAL_ERROR: return "internal_error";
  }
  return "unknown";
}
