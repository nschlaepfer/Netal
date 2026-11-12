#pragma once

// A full learned-core feature boundary, not scene-color or a single PWin replacement.
#ifdef __cplusplus
#include <array>
#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

namespace netal::sr {
enum class FeatureElementType { Float16, Float32, FP8E4M3, FP8E5M2 };
struct FeatureLayout {
  std::string abi;
  std::uint64_t bytes = 0, offset = 0, elementBytes = 0;
  std::array<std::uint64_t, 4> dimensions{}, strides{};
  FeatureElementType elementType = FeatureElementType::Float16;
  bool operator==(const FeatureLayout&) const = default;
  std::uint64_t spanBytes() const {
    std::uint64_t span = elementBytes;
    for (std::size_t i = 0; i < 4; ++i) {
      if (!dimensions[i] || (dimensions[i]-1 && strides[i] > (UINT64_MAX-span)/(dimensions[i]-1))) return 0;
      span += (dimensions[i]-1)*strides[i];
    }
    return span;
  }
  bool bounded() const {
    if (static_cast<unsigned>(elementType) > static_cast<unsigned>(FeatureElementType::FP8E5M2)) return false;
    const std::uint64_t required = elementType == FeatureElementType::Float16 ? 2
        : elementType == FeatureElementType::Float32 ? 4 : 1;
    if (abi.empty() || !bytes || elementBytes != required) return false;
    std::array<std::size_t, 4> axes{0,1,2,3};
    std::sort(axes.begin(), axes.end(), [&](auto a, auto b) { return strides[a] < strides[b]; });
    std::uint64_t footprint = elementBytes;
    for (auto axis : axes) {
      if (dimensions[axis] > 1) {
        if (strides[axis] < footprint || dimensions[axis] - 1 > (UINT64_MAX - footprint) / strides[axis]) return false;
        footprint += (dimensions[axis] - 1) * strides[axis];
      }
    }
    std::uint64_t end = offset;
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
      if (!dimensions[i] || !strides[i]) return false;
      const auto count = dimensions[i] - 1;
      if (count && strides[i] > (std::numeric_limits<std::uint64_t>::max() - end) / count) return false;
      end += count * strides[i];
    }
    return end <= bytes && elementBytes <= bytes - end;
  }
};
struct NativeFeatureLayouts { FeatureLayout input, output; std::uint64_t phase = 0; };
inline bool NativeHWCFeatureLayouts(std::uint64_t width, std::uint64_t height,
                                    std::uint64_t offset, std::uint64_t backingBytes,
                                    std::uint64_t phase, NativeFeatureLayouts& layouts) {
  if (!width || !height || phase >= 8 || width > UINT64_MAX/80 || height > UINT64_MAX/(width*80)) return false;
  auto layout = [&](std::uint64_t channels) {
    FeatureLayout v; v.abi = "netal.native-hwc-f16.v1"; v.bytes = backingBytes; v.offset = offset; v.elementBytes = 2;
    v.dimensions = {1,height,width,channels}; v.strides = {height*width*channels*2,width*channels*2,channels*2,2};
    return v;
  };
  layouts = {layout(16),layout(40),phase};
  return layouts.input.bounded() && layouts.output.bounded();
}
struct NeuralCoreContract {
  std::string backendIdentity, modelIdentity, profileIdentity;
  FeatureLayout input, output;
  // Backend-owned residency/scratch budget; external aliased arena bounds are not provider tensor sizes.
  std::uint64_t phaseMask = 0, scratchBytes = 0, residentBytes = 0;
  // distinctInputOutput describes provider-owned MODEL backing, not the aliased native views.
  // Its Shared/Private storage mode is separately validated by the provider.
  bool sameCommandBuffer = false, distinctInputOutput = true;
};
enum class NeuralCoreReadiness { NativeDefault, InvalidContract, Unqualified };
#ifdef __OBJC__
#define NETAL_SR_OPTIONAL _Nullable
#else
#define NETAL_SR_OPTIONAL
#endif
inline NeuralCoreReadiness InspectNeuralCoreContract(const NeuralCoreContract* NETAL_SR_OPTIONAL candidate,
                                                    const std::string& profile,
                                                    const FeatureLayout& input,
                                                    const FeatureLayout& output,
                                                    std::uint64_t memoryBudget) {
  if (!candidate) return NeuralCoreReadiness::NativeDefault;
  if (candidate->backendIdentity.empty() || candidate->modelIdentity.empty() ||
      profile.empty() || candidate->profileIdentity != profile || candidate->input != input || candidate->output != output ||
      !candidate->input.bounded() || !candidate->output.bounded() || candidate->phaseMask != 0xff ||
      !candidate->sameCommandBuffer || !candidate->distinctInputOutput ||
      candidate->residentBytes > memoryBudget || candidate->scratchBytes > memoryBudget - candidate->residentBytes ||
      candidate->input.spanBytes() > candidate->residentBytes ||
      candidate->output.spanBytes() > candidate->residentBytes - candidate->input.spanBytes())
    return NeuralCoreReadiness::InvalidContract;
  // Deliberately no caller-supplied ready/numerics/temporal booleans. This repository has no
  // qualification authority for an external full-core backend yet. Capability and encode do not qualify it.
  return NeuralCoreReadiness::Unqualified;
}
#undef NETAL_SR_OPTIONAL
} // namespace netal::sr
#endif // __cplusplus

#ifdef __OBJC__
#import <Metal/Metal.h>
NS_ASSUME_NONNULL_BEGIN
@protocol NeuralStudioDLSSSRPreparedNeuralCoreInvocation;
// Created only by the session for its authentic outer lease. No completion or ready setter.
__attribute__((objc_subclassing_restricted))
@interface NeuralStudioDLSSSRNeuralCoreEncodingContext : NSObject
- (instancetype)init NS_UNAVAILABLE;
@property(nonatomic, readonly) NSUInteger phase;
@property(nonatomic, readonly) BOOL reset;
@property(nonatomic, readonly) id<MTLBuffer> inputBuffer;
@property(nonatomic, readonly) id<MTLBuffer> outputBuffer;
#ifdef __cplusplus
@property(nonatomic, readonly) netal::sr::FeatureLayout inputLayout;
@property(nonatomic, readonly) netal::sr::FeatureLayout outputLayout;
#endif
/// Exact invocation association, one claim only, unavailable after encode returns or throws.
- (nullable id<MTL4CommandBuffer>)claimCommandBufferForInvocation:(id<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>)invocation error:(NSError**)error;
@end
@protocol NeuralStudioDLSSSRPreparedNeuralCoreInvocation <NSObject>
/// Complete immutable resource set, including self, all temporary packing/model backing,
/// residency sets, tensors, pipelines, heaps, tables and fences, BEFORE any GPU emission.
@property(nonatomic, readonly, copy) NSArray* retainedResources;
- (BOOL)encodeNeuralCoreWithContext:(NeuralStudioDLSSSRNeuralCoreEncodingContext*)context error:(NSError**)error;
@end
@protocol NeuralStudioDLSSSRNeuralCoreBackend <NSObject>
#ifdef __cplusplus
- (netal::sr::NeuralCoreContract)neuralCoreContract;
/// Preparation MUST NOT emit GPU commands. Views may alias the same native arena; fully pack
/// all HWC16 input into separate internal model backing before any HWC40 consumer writeback.
/// bytes is the accessible end-bound from buffer base; offset+spanBytes must fit it and length.
- (nullable id<NeuralStudioDLSSSRPreparedNeuralCoreInvocation>)prepareNeuralCoreForCommandBuffer:(id<MTL4CommandBuffer>)commandBuffer
    inputBuffer:(id<MTLBuffer>)input inputLayout:(netal::sr::FeatureLayout)inputLayout
    outputBuffer:(id<MTLBuffer>)output outputLayout:(netal::sr::FeatureLayout)outputLayout
    phase:(NSUInteger)phase reset:(BOOL)reset error:(NSError**)error;
#endif
// Prepared invocation owns validated layout pack/ML/copyback on the exact claimed CB.
// Provider must apply all-stage queue consumer barriers before packing/ML/copyback and all
// intra-pass dependencies. No private queue/commit, second frame lease, or blocking prediction.
@end
NS_ASSUME_NONNULL_END
#endif
