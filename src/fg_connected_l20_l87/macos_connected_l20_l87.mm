#include <map>
#define main FgAcceptedConnectedTailUnusedMain
#include "../fg_connected_tail_l84_l87/macos_connected_l56_l87.mm"
#undef main

#include "../fg_conv/conv_1x1_contract.h"
#include "guarded_block.inc"
#include "../fg_conv/conv_3x3_contract.h"
#include "../fg_mid_l53/contract.h"
#include "../fg_mid_l54/contract.h"
#include "../fg_mid_l55/fg_mid_l55_contract.h"
#include "../fg_mid_l55/fg_mid_l55_inputs.h"

namespace {

namespace m53 = fg_mid_l53;
namespace m55 = fg_mid_l55;

struct PrefixWeightSpec {
  std::uint32_t token;
  std::size_t bytes;
  const char* sha256;
};

constexpr std::array<PrefixWeightSpec, 17> kPrefixWeights = {{
    {63,56320,"44595b02307a6b934a124abe476cfd327440e2acba7f5ed69f73195ca743e3d9"},
    {65,10240,"eb99328524fddc9f5f3fac12b8e3a4011e66a75e071cd07126a70c553458bc60"},
    {67,299008,"4191924ff335d5018987f2526f933cb4cdbb9145411e3ce34f63c8206321be55"},
    {69,593920,"c0c5648f9cb839e41d0e786c8034aa0bcc931b6ac84c58cfbde0502dc8a53404"},
    {71,1187840,"fe2c3057f2e894219601d3d7200eabfa97d990e08ccb23b43d52dfcba2d2fb93"},
    {73,540672,"e23316a86707c1d3d377aa0711df47d05976ad1a30f6428524d030dd17145de9"},
    {75,532480,"9f20d6589cc9269ad020233e68196d66eba0e8a76497374cf6d445385f94dbc4"},
    {77,270336,"9e5da09b93650c74e89d5262aa8a9a6a3dcbd0d753dc8c9d9aed39731e39b2a0"},
    {79,135168,"3969384493583e98dcf357d3046a10a5b836585261f3e584605bf11e23fb8d46"},
    {81,69632,"a0c6690faee2838521c6ad0f9445425d639446c0ef1b7fed9b43d21e7a8ec650"},
    {83,69632,"3a13970b482f1f78ede1bfe531956c203f08f99077c81bda2d1e4d0f453cf9a0"},
    {85,69632,"47361a699fa6514332dc560cf2416872256e61d1b36b1ca29df323a53ee2be77"},
    {87,34816,"327bd9b02da4aaac8b251320f60b4aeccb0df7df84bba44d7f1f6a9fdf62c934"},
    {89,18432,"e24f67110bab401ab7f70a861f3f09d2cd3cd6abf8d1aa158ba3300efb64797e"},
    {91,9216,"325d7dfac5620a89edc1d2315731414cf755c50506ea5758a8c3cf563649577c"},
    {93,5120,"bb98eb15be7d255a15a46994589705560e0691e273e286a36a68fd679a1890a9"},
    {95,2560,"477c68ebace499e2e1ac331f214fae3e9fb4df815a9e7d8142a0f1fb414b340e"},
}};

struct PrefixTensorSpec {
  std::uint32_t width;
  std::uint32_t height;
  std::uint32_t channels;
};

constexpr std::array<PrefixTensorSpec, 32> kPrefixTensors = {{
    {160,96,32},{80,48,32},{80,48,64},{40,24,64},
    {40,24,128},{20,12,128},{20,12,128},{10,6,128},
    {10,6,256},{5,3,256},{5,3,512},{5,3,256},
    {10,6,256},{10,6,256},{10,6,256},{10,6,128},
    {20,12,128},{20,12,128},{20,12,128},{20,12,128},
    {40,24,128},{40,24,128},{40,24,128},{40,24,64},
    {80,48,64},{80,48,64},{80,48,64},{80,48,32},
    {160,96,32},{160,96,32},{160,96,32},{160,96,16},
}};

struct PrefixConv3x3Params {
  std::uint32_t width, height, input_channels, output_channels;
  std::uint32_t active_weight_half_count, column_stage_schedule;
};
struct PrefixConv1x1Params {
  std::uint32_t width, height, input_channels, output_channels;
  std::uint32_t active_weight_half_count, apply_leaky_relu;
};
struct PrefixPoolParams {
  std::uint32_t input_width, input_height, output_width, output_height;
  std::uint32_t channels;
};
struct PrefixUpscaleParams {
  std::uint32_t input_height, input_width, output_height, output_width;
  std::uint32_t physical_output_height, physical_output_width, channels;
};
struct PrefixAddParams {
  std::uint32_t output_width, output_height, channels;
  std::uint32_t input_a_width, input_a_height, input_b_width, input_b_height;
  std::uint32_t vector_thread_stride;
};
struct L52Params {
  std::uint32_t width, height, input_width, input_height;
  std::uint32_t input_offset_x, input_offset_y, crop_top, crop_bottom;
};
struct L53Params {
  std::uint32_t width, height;
  float interpolation_factor, motion_threshold, depth_threshold, depth_scale;
  std::uint8_t invert_depth, reserved0;
  std::uint16_t reserved1;
  std::uint32_t low_phase;
};
struct L54Params { std::uint32_t width, height; };

static_assert(sizeof(PrefixConv3x3Params) == 24);
static_assert(sizeof(PrefixConv1x1Params) == 24);
static_assert(sizeof(PrefixPoolParams) == 20);
static_assert(sizeof(PrefixUpscaleParams) == 28);
static_assert(sizeof(PrefixAddParams) == 32);
static_assert(sizeof(L52Params) == 32);
static_assert(sizeof(L53Params) == 32);
static_assert(sizeof(L54Params) == 8);

std::size_t PrefixTensorBytes(const PrefixTensorSpec& tensor) {
  return std::size_t{tensor.width} * tensor.height * tensor.channels * 2u;
}

std::uint64_t PrefixMix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

// Per-process caches. This program is the runtime's per-frame tail: without these it re-read and
// re-hashed ~7 MB of model files and copied several MB of buffers every frame, on the game's render
// thread (under Rosetta for a CrossOver game). The audit path still sees fresh copies.
#ifndef NSFG_TAIL_AUDIT_HOOK
#define NSFG_TAIL_AUDIT_HOOK() true
#endif
static const std::vector<std::uint8_t>& CachedRead(const std::filesystem::path& path) {
  static std::map<std::string, std::vector<std::uint8_t>> cache;
  auto it = cache.find(path.string());
  if (it == cache.end()) it = cache.emplace(path.string(), ReadB(path)).first;
  return it->second;
}
static const std::string& CachedSha256(const std::filesystem::path& path) {
  static std::map<std::string, std::string> cache;
  auto it = cache.find(path.string());
  if (it == cache.end()) it = cache.emplace(path.string(), Sha256(CachedRead(path))).first;
  return it->second;
}
template <typename F>
static std::vector<std::uint8_t> AuditOnly(F&& read) { return NSFG_TAIL_AUDIT_HOOK() ? read() : std::vector<std::uint8_t>{}; }

std::vector<std::uint8_t> PrefixInput() {
  constexpr std::array<std::uint16_t, 13> values = {
      0x0000,0x8000,0x2000,0xa000,0x2400,0xa400,0x2800,
      0xa800,0x2c00,0xac00,0x3000,0xb000,0x3400};
  std::vector<std::uint8_t> result(kInputBytes);
  for (std::size_t index = 0; index < kInputBytes / 2u; ++index) {
    const auto mixed = PrefixMix(0x243f6a8885a308d3ull +
                                 index * 0x9e3779b97f4a7c15ull);
    StoreHalf(&result, index * 2u, values[mixed % values.size()]);
  }
  return result;
}

std::vector<std::uint8_t> BinaryDepthMac() {
  std::vector<std::uint8_t> result(320u * 180u);
  for (std::size_t pixel = 0; pixel < result.size(); ++pixel)
    result[pixel] = ((pixel / 320u + pixel % 320u) & 1u) ? 0xffu : 0u;
  return result;
}

id<MTLTexture> ConnectedTexture(id<MTLDevice> device, MTLPixelFormat format,
                                NSUInteger width, NSUInteger height,
                                MTLTextureUsage usage,
                                const std::vector<std::uint8_t>& initial,
                                NSUInteger bytes_per_pixel) {
  if (initial.size() != width * height * bytes_per_pixel)
    throw std::runtime_error("connected Metal texture size drift");
  auto* descriptor =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:format
                                                         width:width
                                                        height:height
                                                     mipmapped:NO];
  descriptor.storageMode = MTLStorageModeShared;
  descriptor.usage = usage;
  id<MTLTexture> result = [device newTextureWithDescriptor:descriptor];
  if (result == nil) throw std::runtime_error("connected texture allocation");
  [result replaceRegion:MTLRegionMake2D(0, 0, width, height)
             mipmapLevel:0
               withBytes:initial.data()
             bytesPerRow:width * bytes_per_pixel];
  return result;
}

std::vector<std::uint8_t> ConnectedReadTexture(id<MTLTexture> texture,
                                               NSUInteger bytes_per_pixel) {
  std::vector<std::uint8_t> result(texture.width * texture.height *
                                   bytes_per_pixel);
  [texture getBytes:result.data()
         bytesPerRow:texture.width * bytes_per_pixel
          fromRegion:MTLRegionMake2D(0, 0, texture.width, texture.height)
         mipmapLevel:0];
  return result;
}

// The prefix's 33 launches share one compute encoder when this is set: Metal orders dependent
// dispatches inside an encoder through hazard tracking, and an encoder boundary per launch was most of
// the prefix's time (the arithmetic is under 0.1 ms).
static id<MTLComputeCommandEncoder> g_prefix_encoder = nil;
static id<MTLComputeCommandEncoder> PrefixEncoder(id<MTLCommandBuffer> command) {
  return g_prefix_encoder != nil ? g_prefix_encoder : [command computeCommandEncoder];
}
static void PrefixEnd(id<MTLComputeCommandEncoder> encoder) { if (g_prefix_encoder == nil) [encoder endEncoding]; }
// The hardware conv kernels are launched output-channel fastest (x = oc, y = px, z = py) so that a
// pixel's eight output channels sit in one simdgroup octet: their weight blocks are consecutive and
// they share the input taps.
static bool PrefixHW() { static const bool hw = std::getenv("NSFG_PREFIX_MODE") != nullptr && std::strcmp(std::getenv("NSFG_PREFIX_MODE"), "hw") == 0; return hw; }
static void PrefixDispatchConv(id<MTLComputeCommandEncoder> encoder, uint32_t width, uint32_t height, uint32_t output_channels) {
  if (PrefixHW()) {
    [encoder dispatchThreads:MTLSizeMake(output_channels, width, height) threadsPerThreadgroup:MTLSizeMake(std::min<uint32_t>(output_channels, 32u), 4, 1)];
  } else {
    [encoder dispatchThreads:MTLSizeMake(width, height, output_channels) threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
  }
}

void PrefixEncode3x3(id<MTLCommandBuffer> command,
                     id<MTLComputePipelineState> pipeline,
                     const fg_conv_3x3::Layer& layer, id<MTLBuffer> weight,
                     id<MTLBuffer> input, id<MTLBuffer> output) {
  const PrefixConv3x3Params params = {
      layer.width, layer.height, layer.input_channels, layer.output_channels,
      layer.input_channels * layer.output_channels * 9u,
      layer.column_stage_schedule ? 1u : 0u};
  id<MTLComputeCommandEncoder> encoder = PrefixEncoder(command);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  PrefixDispatchConv(encoder, layer.width, layer.height, layer.output_channels);
  PrefixEnd(encoder);
}

void PrefixEncode1x1(id<MTLCommandBuffer> command,
                     id<MTLComputePipelineState> pipeline,
                     const fg_conv_1x1::Layer& layer, id<MTLBuffer> weight,
                     id<MTLBuffer> input, id<MTLBuffer> output) {
  const PrefixConv1x1Params params = {
      layer.width, layer.height, layer.input_channels, layer.output_channels,
      layer.input_channels * layer.output_channels,
      layer.leaky_relu ? 1u : 0u};
  id<MTLComputeCommandEncoder> encoder = PrefixEncoder(command);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:weight offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  PrefixDispatchConv(encoder, layer.width, layer.height, layer.output_channels);
  PrefixEnd(encoder);
}

void PrefixEncodePool(id<MTLCommandBuffer> command,
                      id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> input, id<MTLBuffer> output,
                      std::uint32_t width, std::uint32_t height,
                      std::uint32_t channels) {
  const PrefixPoolParams params = {width, height, (width + 1u) / 2u,
                                   (height + 1u) / 2u, channels};
  id<MTLComputeCommandEncoder> encoder = PrefixEncoder(command);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:output offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input offset:kGuardBytes atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  [encoder dispatchThreadgroups:MTLSizeMake((params.output_width + 3u) / 4u,
                                            (params.output_height + 3u) / 4u,
                                            1)
             threadsPerThreadgroup:MTLSizeMake(4, 4, 16)];
  PrefixEnd(encoder);
}

void PrefixEncodeUpscale(id<MTLCommandBuffer> command,
                         id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> input, id<MTLBuffer> output,
                         std::uint32_t input_width,
                         std::uint32_t input_height,
                         std::uint32_t output_width,
                         std::uint32_t output_height,
                         std::uint32_t channels) {
  const PrefixUpscaleParams params = {input_height, input_width,
                                      output_height, output_width,
                                      output_height, output_width, channels};
  id<MTLComputeCommandEncoder> encoder = PrefixEncoder(command);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input offset:kGuardBytes atIndex:0];
  [encoder setBuffer:output offset:kGuardBytes atIndex:1];
  [encoder setBytes:&params length:sizeof(params) atIndex:2];
  [encoder dispatchThreads:MTLSizeMake(channels / 2u, output_width,
                                       output_height)
          threadsPerThreadgroup:MTLSizeMake(16, 4, 1)];
  PrefixEnd(encoder);
}

void PrefixEncodeAdd(id<MTLCommandBuffer> command,
                     id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> input_a, id<MTLBuffer> input_b,
                     id<MTLBuffer> output, std::uint32_t width,
                     std::uint32_t height, std::uint32_t channels,
                     std::uint32_t grid_x, std::uint32_t grid_y,
                     std::uint32_t block_z) {
  const PrefixAddParams params = {width, height, channels, width, height,
                                  width, height, block_z};
  id<MTLComputeCommandEncoder> encoder = PrefixEncoder(command);
  [encoder setComputePipelineState:pipeline];
  [encoder setBuffer:input_a offset:kGuardBytes atIndex:0];
  [encoder setBuffer:input_b offset:kGuardBytes atIndex:1];
  [encoder setBuffer:output offset:kGuardBytes atIndex:2];
  [encoder setBytes:&params length:sizeof(params) atIndex:3];
  [encoder dispatchThreadgroups:MTLSizeMake(grid_x, grid_y, 1)
             threadsPerThreadgroup:MTLSizeMake(4, 4, block_z)];
  PrefixEnd(encoder);
}

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 8) {
      std::cerr << "usage=fg-connected-l20-l87-mac <libs> <model> <rom> "
                   "<sigmoid> <output> <run-id> <poison-hex>\n";
      return 2;
    }
    try {
      const std::filesystem::path libs = argv[1];
      const std::filesystem::path model = argv[2];
      const std::filesystem::path output = argv[5];
      std::filesystem::create_directories(output);
      char* end = nullptr;
      const auto parsed = std::strtoul(argv[7], &end, 16);
      if (end == argv[7] || *end != '\0' || parsed > 255u)
        throw std::runtime_error("invalid poison");
      const auto poison = static_cast<std::uint8_t>(parsed);
      const auto& rom = CachedRead(argv[3]);
      if (rom.size() != 2097152u ||
          CachedSha256(argv[3]) !=
              "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750")
        throw std::runtime_error("RCP architecture ROM identity mismatch");
      const auto& sigmoid = CachedRead(argv[4]);
      if (sigmoid.size() != 262144u ||
          CachedSha256(argv[4]) !=
              "394394a5258bad437495d68076be75d8413fa0ed5b752400e3947c332437b850")
        throw std::runtime_error("sigmoid architecture LUT identity mismatch");
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      id<MTLCommandQueue> queue = [device newCommandQueue];
      if (queue == nil) throw std::runtime_error("Metal queue unavailable");
      auto conv3 = Pipeline(device, libs / "conv_3x3.metallib",
                            "fg_conv_3x3");
      auto conv1 = Pipeline(device, libs / "conv_1x1.metallib",
                            "fg_conv_1x1");
      auto pool = Pipeline(device, libs / "pooling.metallib", "fg_pooling");
      auto upscale = Pipeline(device, libs / "upscale-prefix.metallib",
                              "fg_upscale");
      auto add = Pipeline(device, libs / "elementwise.metallib",
                          "fg_elementwise_add_family");
      auto l52_pipeline =
          Pipeline(device, libs / "mid_l52.metallib", "fg_mid_l52");
      auto l53_candidates = Pipeline(device, libs / "mid_l53.metallib",
                                     "fg_mid_l53_candidates");
      auto l53_pack = Pipeline(device, libs / "mid_l53.metallib",
                               "fg_mid_l53_pack");
      if (l53_candidates.maxTotalThreadsPerThreadgroup < 324u)
        throw std::runtime_error("L53 324-thread path unsupported");
      auto l54_pipeline =
          Pipeline(device, libs / "mid_l54.metallib", "fg_mid_l54");
      auto l55_pipeline = Pipeline(device, libs / "mid_l55.metallib",
                                   "fg_mid_l55_exact");
      auto initial = Pipeline(device, libs / "initial_merge.metallib",
                              "fg_initial_merge");
      auto block0_pre = Pipeline(device, libs / "conv_pre.metallib",
                                 "fg_custom_block0_conv_pre");
      auto block0_conv0 = Pipeline(device, libs / "conv0.metallib",
                                   "fg_custom_block0_conv0_c8");
      auto block0_conv1 = Pipeline(device, libs / "conv1.metallib",
                                   "fg_custom_block0_conv1");
      auto block0_conv2 = Pipeline(device, libs / "conv2.metallib",
                                   "fg_custom_block0_conv2");
      auto block0_bot0 = Pipeline(device, libs / "bot0.metallib",
                                  "fg_connected_block0_bot0");
      auto block0_bot1 = Pipeline(device, libs / "bot1.metallib",
                                  "fg_connected_block0_bot1");
      auto custom_upsample = Pipeline(device, libs / "upsample.metallib",
                                      "fg_custom_upsample_plane");
      auto central = Pipeline(device, libs / "central.metallib",
                              "fg_central_block");
      auto block1_conv0 = Pipeline(device, libs / "block1_conv0.metallib",
                                   "fg_custom_block1_conv0");
      auto block1_conv1 = Pipeline(device, libs / "block1_conv1.metallib",
                                   "fg_custom_block1_conv1");
      auto block1_conv2 = Pipeline(device, libs / "block1_conv2.metallib",
                                   "fg_custom_block1_conv2");
      auto block1_bot0 = Pipeline(device, libs / "block1_bot0.metallib",
                                  "fg_connected_block1_bot0");
      auto block1_bot1 = Pipeline(device, libs / "block1_bot1.metallib",
                                  "fg_connected_block1_bot1");
      auto l84_pixels = Pipeline(device, libs / "postprocess_l84.metallib",
                                 "fg_postprocess_l84_pixels");
      std::array<id<MTLComputePipelineState>,4> l84_levels = {
          Pipeline(device,libs/"postprocess_l84.metallib","fg_postprocess_l84_level0"),
          Pipeline(device,libs/"postprocess_l84.metallib","fg_postprocess_l84_level1"),
          Pipeline(device,libs/"postprocess_l84.metallib","fg_postprocess_l84_level2"),
          Pipeline(device,libs/"postprocess_l84.metallib","fg_postprocess_l84_level3")};
      auto l85_pipeline = Pipeline(device, libs / "postprocess_l85.metallib",
                                   "fg_postprocess_l85_exact");
      auto l86_pipeline = Pipeline(device, libs / "postprocess_l86.metallib",
                                   "fg_postprocess_l86_mode0");
      auto l87_pipeline = Pipeline(device, libs / "compositor_l87.metallib",
                                   "fg_compositor_l87_exact");

      std::array<const std::vector<std::uint8_t>*, 17> weight_before{};
      std::array<id<MTLBuffer>, 17> weights{};
      for (std::size_t index = 0; index < kPrefixWeights.size(); ++index) {
        const auto path = model / TokenName(kPrefixWeights[index].token);
        weight_before[index] = &CachedRead(path);
        if (weight_before[index]->size() != kPrefixWeights[index].bytes ||
            CachedSha256(path) != kPrefixWeights[index].sha256)
          throw std::runtime_error("prefix model identity mismatch");
        weights[index] = Guarded(device, weight_before[index],
                                 kPrefixWeights[index].bytes, 0);
      }
      const auto prefix_input_before = PrefixInput();
      id<MTLBuffer> prefix_input =
          Guarded(device, &prefix_input_before, kInputBytes, 0);
      std::array<id<MTLBuffer>, 32> tensors{};
      for (std::size_t index = 0; index < tensors.size(); ++index)
        tensors[index] = Guarded(device, nullptr,
                                 PrefixTensorBytes(kPrefixTensors[index]),
                                 poison);
      const auto& l20 = fg_conv_3x3::FindLayer("l20");
      const auto& l22 = fg_conv_1x1::FindLayer("l22");
      const auto& l24 = fg_conv_3x3::FindLayer("l24");
      const auto& l26 = fg_conv_3x3::FindLayer("l26");
      const auto& l28 = fg_conv_3x3::FindLayer("l28");
      const auto& l30 = fg_conv_1x1::FindLayer("l30");
      const auto& l31 = fg_conv_1x1::FindLayer("l31");
      const auto& l34 = fg_conv_1x1::FindLayer("l34");
      const auto& l35 = fg_conv_1x1::FindLayer("l35");
      const auto& l38 = fg_conv_1x1::FindLayer("l38");
      const auto& l39 = fg_conv_1x1::FindLayer("l39");
      const auto& l42 = fg_conv_1x1::FindLayer("l42");
      const auto& l43 = fg_conv_1x1::FindLayer("l43");
      const auto& l46 = fg_conv_1x1::FindLayer("l46");
      const auto& l47 = fg_conv_1x1::FindLayer("l47");
      const auto& l50 = fg_conv_1x1::FindLayer("l50");
      const auto& l51 = fg_conv_1x1::FindLayer("l51");
      id<MTLCommandBuffer> prefix_command = [queue commandBuffer];
      g_prefix_encoder = [prefix_command computeCommandEncoder];
      PrefixEncode3x3(prefix_command, conv3, l20, weights[0], prefix_input,
                      tensors[0]);
      PrefixEncodePool(prefix_command, pool, tensors[0], tensors[1],160,96,32);
      PrefixEncode1x1(prefix_command,conv1,l22,weights[1],tensors[1],tensors[2]);
      PrefixEncodePool(prefix_command,pool,tensors[2],tensors[3],80,48,64);
      PrefixEncode3x3(prefix_command,conv3,l24,weights[2],tensors[3],tensors[4]);
      PrefixEncodePool(prefix_command,pool,tensors[4],tensors[5],40,24,128);
      PrefixEncode3x3(prefix_command,conv3,l26,weights[3],tensors[5],tensors[6]);
      PrefixEncodePool(prefix_command,pool,tensors[6],tensors[7],20,12,128);
      PrefixEncode3x3(prefix_command,conv3,l28,weights[4],tensors[7],tensors[8]);
      PrefixEncodePool(prefix_command,pool,tensors[8],tensors[9],10,6,256);
      PrefixEncode1x1(prefix_command,conv1,l30,weights[5],tensors[9],tensors[10]);
      PrefixEncode1x1(prefix_command,conv1,l31,weights[6],tensors[10],tensors[11]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[11],tensors[12],5,3,10,6,256);
      PrefixEncodeAdd(prefix_command,add,tensors[12],tensors[8],tensors[13],10,6,256,3,2,8);
      PrefixEncode1x1(prefix_command,conv1,l34,weights[7],tensors[13],tensors[14]);
      PrefixEncode1x1(prefix_command,conv1,l35,weights[8],tensors[14],tensors[15]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[15],tensors[16],10,6,20,12,128);
      PrefixEncodeAdd(prefix_command,add,tensors[16],tensors[6],tensors[17],20,12,128,5,3,8);
      PrefixEncode1x1(prefix_command,conv1,l38,weights[9],tensors[17],tensors[18]);
      PrefixEncode1x1(prefix_command,conv1,l39,weights[10],tensors[18],tensors[19]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[19],tensors[20],20,12,40,24,128);
      PrefixEncodeAdd(prefix_command,add,tensors[20],tensors[4],tensors[21],40,24,128,10,6,8);
      PrefixEncode1x1(prefix_command,conv1,l42,weights[11],tensors[21],tensors[22]);
      PrefixEncode1x1(prefix_command,conv1,l43,weights[12],tensors[22],tensors[23]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[23],tensors[24],40,24,80,48,64);
      PrefixEncodeAdd(prefix_command,add,tensors[24],tensors[2],tensors[25],80,48,64,20,12,8);
      PrefixEncode1x1(prefix_command,conv1,l46,weights[13],tensors[25],tensors[26]);
      PrefixEncode1x1(prefix_command,conv1,l47,weights[14],tensors[26],tensors[27]);
      PrefixEncodeUpscale(prefix_command,upscale,tensors[27],tensors[28],80,48,160,96,32);
      PrefixEncodeAdd(prefix_command,add,tensors[28],tensors[0],tensors[29],160,96,32,40,24,4);
      PrefixEncode1x1(prefix_command,conv1,l50,weights[15],tensors[29],tensors[30]);
      PrefixEncode1x1(prefix_command,conv1,l51,weights[16],tensors[30],tensors[31]);
      [g_prefix_encoder endEncoding]; g_prefix_encoder = nil;
      Wait(prefix_command);

      id<MTLBuffer> rom_buffer = Guarded(device, &rom, rom.size(), 0);
      const auto depth_before = BinaryDepthMac();
      auto depth = ConnectedTexture(device, MTLPixelFormatR8Unorm, 320, 180,
                                    MTLTextureUsageShaderRead, depth_before, 1);
      auto l52_a = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(320u * 180u * 4u, poison), 4);
      auto l52_b = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(320u * 180u * 4u, poison), 4);
      auto l52_confidence = ConnectedTexture(
          device, MTLPixelFormatR16Float, 320, 180,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(320u * 180u * 2u, poison), 2);
      const L52Params l52_params{320,180,160,96,0,0,3,3};
      id<MTLBuffer> l52_params_buffer =
          [device newBufferWithBytes:&l52_params length:sizeof(l52_params)
                             options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> l52_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> l52_encoder =
          [l52_command computeCommandEncoder];
      [l52_encoder setComputePipelineState:l52_pipeline];
      [l52_encoder setBuffer:tensors[31] offset:kGuardBytes atIndex:0];
      [l52_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:1];
      [l52_encoder setBuffer:l52_params_buffer offset:0 atIndex:2];
      [l52_encoder setTexture:depth atIndex:0];
      [l52_encoder setTexture:l52_a atIndex:1];
      [l52_encoder setTexture:l52_b atIndex:2];
      [l52_encoder setTexture:l52_confidence atIndex:3];
      [l52_encoder dispatchThreads:MTLSizeMake(320,180,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l52_encoder endEncoding];
      Wait(l52_command);

      const std::array<std::vector<std::uint8_t>, 4> l53_input_before = {
          m53::CurrToPrevFlow(), m53::PrevToCurrFlow(false),
          m53::CurrDepth(), m53::PrevDepth()};
      auto l53_flow_a = ConnectedTexture(
          device, MTLPixelFormatRG32Float, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[0], 8);
      auto l53_flow_b = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[1], 4);
      auto l53_depth_a = ConnectedTexture(
          device, MTLPixelFormatR32Uint, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[2], 4);
      auto l53_depth_b = ConnectedTexture(
          device, MTLPixelFormatR32Uint, 320, 180,
          MTLTextureUsageShaderRead, l53_input_before[3], 4);
      const auto l53_initial = m53::InitialOutput();
      id<MTLBuffer> l53_a =
          Guarded(device, &l53_initial, m53::kOutputBytes, 0);
      id<MTLBuffer> l53_b =
          Guarded(device, &l53_initial, m53::kOutputBytes, 0);
      std::vector<std::uint8_t> high_initial(m53::kPixelCount * 4u);
      for (std::size_t pixel = 0; pixel < m53::kPixelCount; ++pixel)
        m53::Store32(&high_initial, pixel * 4u, 0x7fffffffu);
      const std::vector<std::uint8_t> low_initial(m53::kPixelCount * 4u,
                                                  0xffu);
      id<MTLBuffer> high_a =
          Guarded(device, &high_initial, high_initial.size(), 0);
      id<MTLBuffer> low_a =
          Guarded(device, &low_initial, low_initial.size(), 0);
      id<MTLBuffer> high_b =
          Guarded(device, &high_initial, high_initial.size(), 0);
      id<MTLBuffer> low_b =
          Guarded(device, &low_initial, low_initial.size(), 0);
      L53Params l53_params{320,180,0.5f,12.96000003814697265625f,
                           40.0f,1.0f,0,0,0,0};
      id<MTLCommandBuffer> l53_command = [queue commandBuffer];
      for (std::uint32_t phase = 0; phase < 2u; ++phase) {
        l53_params.low_phase = phase;
        id<MTLComputeCommandEncoder> encoder =
            [l53_command computeCommandEncoder];
        [encoder setComputePipelineState:l53_candidates];
        [encoder setTexture:l53_flow_a atIndex:0];
        [encoder setTexture:l53_flow_b atIndex:1];
        [encoder setTexture:l53_depth_a atIndex:2];
        [encoder setTexture:l53_depth_b atIndex:3];
        [encoder setBuffer:high_a offset:kGuardBytes atIndex:0];
        [encoder setBuffer:low_a offset:kGuardBytes atIndex:1];
        [encoder setBuffer:high_b offset:kGuardBytes atIndex:2];
        [encoder setBuffer:low_b offset:kGuardBytes atIndex:3];
        [encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:4];
        [encoder setBytes:&l53_params length:sizeof(l53_params) atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake(20,12,1)
               threadsPerThreadgroup:MTLSizeMake(324,1,1)];
        [encoder endEncoding];
      }
      id<MTLComputeCommandEncoder> l53_pack_encoder =
          [l53_command computeCommandEncoder];
      [l53_pack_encoder setComputePipelineState:l53_pack];
      [l53_pack_encoder setBuffer:high_a offset:kGuardBytes atIndex:0];
      [l53_pack_encoder setBuffer:low_a offset:kGuardBytes atIndex:1];
      [l53_pack_encoder setBuffer:high_b offset:kGuardBytes atIndex:2];
      [l53_pack_encoder setBuffer:low_b offset:kGuardBytes atIndex:3];
      [l53_pack_encoder setBuffer:l53_a offset:kGuardBytes atIndex:4];
      [l53_pack_encoder setBuffer:l53_b offset:kGuardBytes atIndex:5];
      [l53_pack_encoder setBytes:&l53_params length:sizeof(l53_params)
                          atIndex:6];
      [l53_pack_encoder dispatchThreads:MTLSizeMake(m53::kPixelCount,1,1)
                       threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [l53_pack_encoder endEncoding];
      Wait(l53_command);

      auto l54_blend = ConnectedTexture(
          device, MTLPixelFormatRG16Float, 320, 180,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(fg_mid_l54::kOutputBytes, poison), 4);
      const L54Params l54_params{320,180};
      id<MTLCommandBuffer> l54_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> l54_encoder =
          [l54_command computeCommandEncoder];
      [l54_encoder setComputePipelineState:l54_pipeline];
      [l54_encoder setTexture:l52_a atIndex:0];
      [l54_encoder setTexture:l52_b atIndex:1];
      [l54_encoder setTexture:l54_blend atIndex:2];
      [l54_encoder setBuffer:l53_a offset:kGuardBytes atIndex:0];
      [l54_encoder setBuffer:l53_b offset:kGuardBytes atIndex:1];
      [l54_encoder setBuffer:rom_buffer offset:kGuardBytes atIndex:2];
      [l54_encoder setBytes:&l54_params length:sizeof(l54_params) atIndex:3];
      [l54_encoder dispatchThreads:MTLSizeMake(320,180,1)
          threadsPerThreadgroup:MTLSizeMake(64,1,1)];
      [l54_encoder endEncoding];
      Wait(l54_command);

      const m55::GeneratedInputs l55_input;
      auto rgba_a = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight, MTLTextureUsageShaderRead,
          l55_input.rgba_a, 8);
      auto rgba_b = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight, MTLTextureUsageShaderRead,
          l55_input.rgba_b, 8);
      auto motion = ConnectedTexture(
          device, MTLPixelFormatRG16Float, m55::kCandidateWidth,
          m55::kCandidateHeight, MTLTextureUsageShaderRead,
          l55_input.motion, 4);
      auto surface_a = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(m55::kSurfaceRgbaBytes, poison), 8);
      auto surface_b = ConnectedTexture(
          device, MTLPixelFormatRGBA16Float, m55::kActiveWidth,
          m55::kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(m55::kSurfaceRgbaBytes, poison), 8);
      auto surface_mask = ConnectedTexture(
          device, MTLPixelFormatR8Unorm, m55::kActiveWidth,
          m55::kActiveHeight,
          MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite,
          std::vector<std::uint8_t>(m55::kSurfaceMaskBytes, poison), 1);
      id<MTLBuffer> linear_a =
          Guarded(device, nullptr, m55::kLinearRgbaBytes, poison);
      id<MTLBuffer> linear_b =
          Guarded(device, nullptr, m55::kLinearRgbaBytes, poison);
      id<MTLBuffer> linear_mask =
          Guarded(device, nullptr, m55::kLinearMaskBytes, poison);
      const m55::MetalParameters l55_params{
          m55::kPhysicalWidth,m55::kPhysicalHeight,m55::kAccumulationMode,
          m55::kCandidateWidth,m55::kCandidateHeight,m55::kCandidateWidth,
          m55::kCandidateHeight,m55::kActiveWidth,m55::kActiveHeight,
          m55::kOptionalUiEnabled};
      id<MTLBuffer> l55_params_buffer =
          [device newBufferWithBytes:&l55_params length:sizeof(l55_params)
                             options:MTLResourceStorageModeShared];
      id<MTLCommandBuffer> l55_command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> l55_encoder =
          [l55_command computeCommandEncoder];
      [l55_encoder setComputePipelineState:l55_pipeline];
      const std::array<id<MTLBuffer>,7> l55_buffers = {
          l53_a,l53_b,linear_a,linear_b,linear_mask,rom_buffer,
          l55_params_buffer};
      const std::array<NSUInteger,7> l55_offsets = {
          kGuardBytes,kGuardBytes,kGuardBytes,kGuardBytes,kGuardBytes,
          kGuardBytes,0};
      for (NSUInteger index = 0; index < l55_buffers.size(); ++index)
        [l55_encoder setBuffer:l55_buffers[index] offset:l55_offsets[index]
                       atIndex:index];
      const std::array<id<MTLTexture>,7> l55_textures = {
          rgba_a,rgba_b,motion,l54_blend,surface_a,surface_b,surface_mask};
      for (NSUInteger index = 0; index < l55_textures.size(); ++index)
        [l55_encoder setTexture:l55_textures[index] atIndex:index];
      [l55_encoder dispatchThreads:MTLSizeMake(m55::kPhysicalWidth,
                                               m55::kPhysicalHeight,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l55_encoder endEncoding];
      Wait(l55_command);
      const auto l55_surface_a_before_tail=AuditOnly([&]{return ConnectedReadTexture(surface_a,8);});
      const auto l55_surface_b_before_tail=AuditOnly([&]{return ConnectedReadTexture(surface_b,8);});
      const auto l55_surface_mask_before_tail=AuditOnly([&]{return ConnectedReadTexture(surface_mask,1);});
      const auto l55_linear_a_before_tail=AuditOnly([&]{return Payload(linear_a,m55::kLinearRgbaBytes);});
      const auto l55_linear_b_before_tail=AuditOnly([&]{return Payload(linear_b,m55::kLinearRgbaBytes);});
      const auto l55_linear_mask_before_tail=AuditOnly([&]{return Payload(linear_mask,m55::kLinearMaskBytes);});
      const auto l55_join_before_tail=JoinVec({
          l55_surface_a_before_tail,l55_surface_b_before_tail,
          l55_surface_mask_before_tail,l55_linear_a_before_tail,
          l55_linear_b_before_tail,l55_linear_mask_before_tail});
      id<MTLBuffer> l84_color_a=
          Guarded(device,nullptr,m55::kLinearRgbaBytes,poison);
      id<MTLBuffer> l84_color_b=
          Guarded(device,nullptr,m55::kLinearRgbaBytes,poison);
      id<MTLCommandBuffer> l84_copy_command=[queue commandBuffer];
      id<MTLBlitCommandEncoder> l84_copy_encoder=
          [l84_copy_command blitCommandEncoder];
      const MTLSize l84_copy_size=MTLSizeMake(640,360,1);
      [l84_copy_encoder copyFromTexture:surface_a sourceSlice:0 sourceLevel:0
                            sourceOrigin:MTLOriginMake(0,0,0)
                              sourceSize:l84_copy_size toBuffer:l84_color_a
                       destinationOffset:kGuardBytes
                  destinationBytesPerRow:640u*8u
                destinationBytesPerImage:640u*360u*8u];
      [l84_copy_encoder copyFromTexture:surface_b sourceSlice:0 sourceLevel:0
                            sourceOrigin:MTLOriginMake(0,0,0)
                              sourceSize:l84_copy_size toBuffer:l84_color_b
                       destinationOffset:kGuardBytes
                  destinationBytesPerRow:640u*8u
                destinationBytesPerImage:640u*360u*8u];
      [l84_copy_encoder endEncoding];
      Wait(l84_copy_command);
      const auto l84_color_a_before_tail=AuditOnly([&]{return Payload(l84_color_a,m55::kLinearRgbaBytes);});
      const auto l84_color_b_before_tail=AuditOnly([&]{return Payload(l84_color_b,m55::kLinearRgbaBytes);});

      std::array<const std::vector<std::uint8_t>*,25> tail_weight_before{};
      std::array<const std::vector<std::uint8_t>*,25> tail_bias_before{};
      std::array<id<MTLBuffer>,25> tail_weight{};
      std::array<id<MTLBuffer>,25> tail_bias{};
      for (std::size_t index = 0; index < c::kFullModels.size(); ++index) {
        const auto& spec = c::kFullModels[index];
        const auto weight_path = model / TokenName(spec.weight_token), bias_path = model / TokenName(spec.bias_token);
        tail_weight_before[index] = &CachedRead(weight_path);
        tail_bias_before[index] = &CachedRead(bias_path);
        if (tail_weight_before[index]->size() != spec.weight_bytes ||
            CachedSha256(weight_path) != spec.weight_sha ||
            tail_bias_before[index]->size() != spec.bias_bytes ||
            CachedSha256(bias_path) != spec.bias_sha)
          throw std::runtime_error("tail model identity mismatch");
        tail_weight[index] = Guarded(device, tail_weight_before[index],
                                     spec.weight_bytes, 0);
        tail_bias[index] = Guarded(device, tail_bias_before[index],
                                   spec.bias_bytes, 0);
      }
      id<MTLBuffer> l56=Guarded(device,nullptr,c::kL56Bytes,poison);
      id<MTLBuffer> l57=Guarded(device,nullptr,c::kL57Bytes,poison);
      id<MTLBuffer> l58=Guarded(device,nullptr,c::kL58Bytes,poison);
      id<MTLBuffer> l59=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l60=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l61=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l62=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l63=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l64=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l65=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l66=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l67=Guarded(device,nullptr,c::kLow64Bytes,poison);
      id<MTLBuffer> l68a=Guarded(device,nullptr,c::kL68PlaneBytes,poison);
      id<MTLBuffer> l68b=Guarded(device,nullptr,c::kL68PlaneBytes,poison);
      id<MTLBuffer> l68c=Guarded(device,nullptr,c::kL68PlaneBytes,poison);
      id<MTLBuffer> l69a=Guarded(device,nullptr,c::kL69Out0Bytes,poison);
      id<MTLBuffer> l69b=Guarded(device,nullptr,c::kL69Out1Bytes,poison);
      id<MTLBuffer> l69c=Guarded(device,nullptr,c::kL69Out2Bytes,poison);
      id<MTLBuffer> l70a=Guarded(device,nullptr,c::kL70Out0Bytes,poison);
      id<MTLBuffer> l70b=Guarded(device,nullptr,c::kL70Out1Bytes,poison);
      id<MTLBuffer> l70c=Guarded(device,nullptr,c::kL70Out2Bytes,poison);
      id<MTLBuffer> l71=Guarded(device,nullptr,c::kL71Bytes,poison);
      id<MTLBuffer> l72=Guarded(device,nullptr,c::kL72Bytes,poison);
      id<MTLBuffer> l73=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l74=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l75=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l76=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l77=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l78=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l79=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l80=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l81=Guarded(device,nullptr,c::kBlock1Bytes,poison);
      id<MTLBuffer> l82a=Guarded(device,nullptr,c::kL82PlaneBytes,poison);
      id<MTLBuffer> l82b=Guarded(device,nullptr,c::kL82PlaneBytes,poison);
      id<MTLBuffer> l82c=Guarded(device,nullptr,c::kL82PlaneBytes,poison);
      id<MTLBuffer> l83=Guarded(device,nullptr,c::kL83Bytes,poison);
      const auto tail_mask_before = TailMask();
      id<MTLBuffer> tail_mask =
          Guarded(device,&tail_mask_before,kTailMaskBytes,0);
      id<MTLBuffer> tail_sigmoid =
          Guarded(device,&sigmoid,kSigmoidBytes,0);
      id<MTLBuffer> tail_scratch =
          Guarded(device,nullptr,kTailScratchBytes,poison);
      std::array<id<MTLBuffer>,4> pyramid = {
          Guarded(device,nullptr,kPyramidBytes[0],poison),
          Guarded(device,nullptr,kPyramidBytes[1],poison),
          Guarded(device,nullptr,kPyramidBytes[2],poison),
          Guarded(device,nullptr,kPyramidBytes[3],poison)};
      constexpr std::array<std::size_t,4> l85_sizes = {
          20u*12u*8u,10u*6u*8u,5u*3u*8u,3u*2u*8u};
      std::array<id<MTLBuffer>,4> l85_output = {
          Guarded(device,nullptr,l85_sizes[0],poison),
          Guarded(device,nullptr,l85_sizes[1],poison),
          Guarded(device,nullptr,l85_sizes[2],poison),
          Guarded(device,nullptr,l85_sizes[3],poison)};
      id<MTLTexture> tail_color=TailTexture(
          device,MTLPixelFormatRGBA8Unorm,MTLTextureUsageShaderWrite,poison,4);
      id<MTLTexture> tail_mask_output=TailTexture(
          device,MTLPixelFormatR8Unorm,
          MTLTextureUsageShaderRead|MTLTextureUsageShaderWrite,poison,1);
      id<MTLTexture> tail_native=TailTexture(
          device,MTLPixelFormatRGBA8Unorm,MTLTextureUsageShaderWrite,poison,4);
      const InitialParams initial_params{368,640,0.5f};
      const ResidualParams residual_off{0},residual_on{1};
      const UpsampleParams up0{184,320,368,640,4};
      const UpsampleParams up1{184,320,368,640,1};
      const UpsampleParams up2{184,320,368,640,3};
      const CentralParams central_params{640,368,0.5f};
      const Tail84Params tail84_params{640,360,368,1};
      const Tail85Params tail85_params{40,23};
      const Tail87Params tail87_params{10,640,360,0};
      id<MTLBuffer> initial_params_buffer=[device newBufferWithBytes:&initial_params length:sizeof(initial_params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> residual_off_buffer=[device newBufferWithBytes:&residual_off length:sizeof(residual_off) options:MTLResourceStorageModeShared];
      id<MTLBuffer> residual_on_buffer=[device newBufferWithBytes:&residual_on length:sizeof(residual_on) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up0_buffer=[device newBufferWithBytes:&up0 length:sizeof(up0) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up1_buffer=[device newBufferWithBytes:&up1 length:sizeof(up1) options:MTLResourceStorageModeShared];
      id<MTLBuffer> up2_buffer=[device newBufferWithBytes:&up2 length:sizeof(up2) options:MTLResourceStorageModeShared];
      id<MTLBuffer> central_params_buffer=[device newBufferWithBytes:&central_params length:sizeof(central_params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> tail84_params_buffer=[device newBufferWithBytes:&tail84_params length:sizeof(tail84_params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> tail85_params_buffer=[device newBufferWithBytes:&tail85_params length:sizeof(tail85_params) options:MTLResourceStorageModeShared];
      id<MTLBuffer> tail87_params_buffer=[device newBufferWithBytes:&tail87_params length:sizeof(tail87_params) options:MTLResourceStorageModeShared];
      const auto G=NSUInteger(kGuardBytes),Z=NSUInteger(0);
      id<MTLCommandBuffer> core_command=[queue commandBuffer];
      Dispatch(core_command,initial,{{l56,G},{linear_a,G},{linear_b,G},{linear_mask,G},{initial_params_buffer,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1));
      std::array<id<MTLBuffer>,8> b0_input={l59,l60,l61,l62,l63,l64,l65,l66};
      std::array<id<MTLBuffer>,8> b0_residual={l59,l59,l61,l61,l63,l63,l65,l65};
      std::array<id<MTLBuffer>,8> b0_output={l60,l61,l62,l63,l64,l65,l66,l67};
      std::array<id<MTLBuffer>,8> b1_input={l73,l74,l75,l76,l77,l78,l79,l80};
      std::array<id<MTLBuffer>,8> b1_residual={l73,l73,l75,l75,l77,l77,l79,l79};
      std::array<id<MTLBuffer>,8> b1_output={l74,l75,l76,l77,l78,l79,l80,l81};
      // Custom block, launches 57-83. Default: the guarded exact pipelines (byte-identical to the scalar
      // oracles and to the RTX 5090 checkpoints); NSFG_SCALAR_BLOCK=1 restores the scalar dispatch list.
      static const bool scalar_block = std::getenv("NSFG_SCALAR_BLOCK") != nullptr;
      if (scalar_block) {
      Dispatch(core_command,block0_pre,{{tail_weight[0],G},{tail_bias[0],G},{l56,G},{l57,G}},MTLSizeMake(320,184,32),MTLSizeMake(8,4,1));
      Dispatch(core_command,block0_conv0,{{tail_weight[1],G},{tail_bias[1],G},{l57,G},{l58,G}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1));
      Dispatch(core_command,block0_conv1,{{tail_weight[2],G},{tail_bias[2],G},{l58,G},{l59,G}},MTLSizeMake(80,46,64),MTLSizeMake(8,4,1));
      for(int index=0;index<8;++index)Dispatch(core_command,block0_conv2,{{tail_weight[3+index],G},{tail_bias[3+index],G},{b0_input[index],G},{b0_residual[index],G},{b0_output[index],G},{index&1?residual_on_buffer:residual_off_buffer,Z}},MTLSizeMake(80,46,64),MTLSizeMake(8,4,1));
      Dispatch(core_command,block0_bot0,{{tail_weight[11],G},{tail_bias[11],G},{l67,G},{l68a,G},{l68b,G},{l68c,G}},MTLSizeMake(160,92,96),MTLSizeMake(8,4,1));
      Dispatch(core_command,block0_bot1,{{tail_weight[12],G},{tail_bias[12],G},{l68a,G},{l68b,G},{l68c,G},{l69a,G},{l69b,G},{l69c,G}},MTLSizeMake(320,184,8),MTLSizeMake(8,4,1));
      Dispatch(core_command,custom_upsample,{{l69a,G},{l70a,G},{up0_buffer,Z}},MTLSizeMake(640,368,4),MTLSizeMake(8,4,1));
      Dispatch(core_command,custom_upsample,{{l69b,G},{l70b,G},{up1_buffer,Z}},MTLSizeMake(640,368,1),MTLSizeMake(8,4,1));
      Dispatch(core_command,custom_upsample,{{l69c,G},{l70c,G},{up2_buffer,Z}},MTLSizeMake(640,368,3),MTLSizeMake(8,4,1));
      Dispatch(core_command,central,{{l71,G},{linear_a,G},{linear_b,G},{linear_mask,G},{l70a,G},{l70b,G},{l70c,G},{central_params_buffer,Z}},MTLSizeMake(640,368,1),MTLSizeMake(16,16,1));
      Dispatch(core_command,block1_conv0,{{tail_weight[13],G},{tail_bias[13],G},{l71,G},{l72,G}},MTLSizeMake(320,184,16),MTLSizeMake(8,4,1));
      Dispatch(core_command,block1_conv1,{{tail_weight[14],G},{tail_bias[14],G},{l72,G},{l73,G}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1));
      for(int index=0;index<8;++index)Dispatch(core_command,block1_conv2,{{tail_weight[15+index],G},{tail_bias[15+index],G},{b1_input[index],G},{b1_residual[index],G},{b1_output[index],G},{index&1?residual_on_buffer:residual_off_buffer,Z}},MTLSizeMake(160,92,32),MTLSizeMake(8,4,1));
      Dispatch(core_command,block1_bot0,{{tail_weight[23],G},{tail_bias[23],G},{l81,G},{l82a,G},{l82b,G},{l82c,G}},MTLSizeMake(320,184,48),MTLSizeMake(8,4,1));
      Dispatch(core_command,block1_bot1,{{tail_weight[24],G},{tail_bias[24],G},{l82a,G},{l82b,G},{l82c,G},{l70a,G},{l70b,G},{l70c,G},{l83,G}},MTLSizeMake(640,368,8),MTLSizeMake(8,4,1));
      } else {
        auto& gb = nsfg_guarded::TheBlock();
        gb.Init(device, libs, tail_weight);
        gb.Encode(core_command, G, tail_weight, tail_bias, l56, l57, l58, b0_input, b0_residual, b0_output, l67,
                  l68a, l68b, l68c, l69a, l69b, l69c, l70a, l70b, l70c, l71, l72, b1_input, b1_residual, b1_output, l81,
                  l82a, l82b, l82c, l83, residual_on_buffer, residual_off_buffer, central, linear_a, linear_b, linear_mask, central_params_buffer);
      }
      Wait(core_command);
      if (const char* dump = std::getenv("NSFG_DUMP_BLOCK_DIR")) {   // debugging aid: per-frame block tensors
        static int dump_frame = 0; const std::filesystem::path dir(dump); std::filesystem::create_directories(dir);
        auto put = [&](const char* name, id<MTLBuffer> buf, std::size_t bytes) { auto v = Payload(buf, bytes); std::ofstream o(dir / (std::to_string(dump_frame) + "-" + name + ".bin"), std::ios::binary); o.write((const char*)v.data(), (std::streamsize)v.size()); };
        put("l56", l56, c::kL56Bytes); put("l57", l57, c::kL57Bytes); put("l68a", l68a, c::kL68PlaneBytes); put("l68b", l68b, c::kL68PlaneBytes); put("l68c", l68c, c::kL68PlaneBytes); put("l69a", l69a, c::kL69Out0Bytes); put("l69b", l69b, c::kL69Out1Bytes); put("l69c", l69c, c::kL69Out2Bytes); put("l58", l58, c::kL58Bytes); put("l59", l59, c::kLow64Bytes); put("l67", l67, c::kLow64Bytes); put("l71", l71, c::kL71Bytes); put("l83", l83, c::kL83Bytes);
        ++dump_frame;
      }

      const auto l83_before_tail = AuditOnly([&]{return Payload(l83,c::kL83Bytes);});
      id<MTLCommandBuffer> l84_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l84_encoder=
          [l84_command computeCommandEncoder];
      [l84_encoder setComputePipelineState:l84_pixels];
      const std::array<id<MTLBuffer>,7> l84_buffers = {
          l84_color_a,l84_color_b,tail_mask,l83,tail_sigmoid,tail_scratch,
          tail84_params_buffer};
      for(NSUInteger index=0;index<l84_buffers.size();++index)
        [l84_encoder setBuffer:l84_buffers[index]
                        offset:index==6?0:G atIndex:index];
      [l84_encoder setTexture:tail_color atIndex:0];
      [l84_encoder setTexture:tail_mask_output atIndex:1];
      [l84_encoder setTexture:tail_native atIndex:2];
      [l84_encoder dispatchThreads:MTLSizeMake(640,368,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l84_encoder endEncoding];
      const std::array<MTLSize,4> l84_grid = {
          MTLSizeMake(320,180,1),MTLSizeMake(160,90,1),
          MTLSizeMake(80,45,1),MTLSizeMake(40,23,1)};
      for(NSUInteger index=0;index<l84_levels.size();++index){
        id<MTLComputeCommandEncoder> encoder=
            [l84_command computeCommandEncoder];
        [encoder setComputePipelineState:l84_levels[index]];
        [encoder setBuffer:tail_scratch offset:G atIndex:0];
        [encoder setBuffer:pyramid[index] offset:G atIndex:1];
        [encoder setBuffer:rom_buffer offset:G atIndex:2];
        [encoder dispatchThreads:l84_grid[index]
            threadsPerThreadgroup:MTLSizeMake(16,8,1)];
        [encoder endEncoding];
      }
      Wait(l84_command);
      const auto color84=TailReadTexture(tail_color,4);
      const auto mask84=TailReadTexture(tail_mask_output,1);
      const std::array<std::vector<std::uint8_t>,4> pyramid84 = {
          Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),
          Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3])};
      const auto checkpoint84=JoinVec({color84,mask84,pyramid84[0],
                                       pyramid84[1],pyramid84[2],
                                       pyramid84[3]});

      id<MTLCommandBuffer> l85_command=[queue commandBuffer];
      Dispatch(l85_command,l85_pipeline,
               {{pyramid[3],G},{l85_output[0],G},{l85_output[1],G},
                {l85_output[2],G},{l85_output[3],G},{rom_buffer,G},
                {tail85_params_buffer,Z}},
               MTLSizeMake(20,12,1),MTLSizeMake(8,8,1));
      Wait(l85_command);
      const auto checkpoint85=Payload(l85_output[0],l85_sizes[0]);

      id<MTLCommandBuffer> l86_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l86_encoder=
          [l86_command computeCommandEncoder];
      [l86_encoder setComputePipelineState:l86_pipeline];
      for(NSUInteger index=0;index<pyramid.size();++index)
        [l86_encoder setBuffer:pyramid[index] offset:G atIndex:index];
      [l86_encoder setBuffer:l85_output[0] offset:G atIndex:4];
      [l86_encoder setBuffer:rom_buffer offset:G atIndex:5];
      [l86_encoder dispatchThreadgroups:MTLSizeMake(20,12,1)
                 threadsPerThreadgroup:MTLSizeMake(256,1,1)];
      [l86_encoder endEncoding];
      Wait(l86_command);
      const std::array<std::vector<std::uint8_t>,5> pyramid86 = {
          Payload(pyramid[0],kPyramidBytes[0]),
          Payload(pyramid[1],kPyramidBytes[1]),
          Payload(pyramid[2],kPyramidBytes[2]),
          Payload(pyramid[3],kPyramidBytes[3]),
          Payload(l85_output[0],l85_sizes[0])};
      const auto checkpoint86=JoinVec({pyramid86[0],pyramid86[1],
                                       pyramid86[2],pyramid86[3],
                                       pyramid86[4]});

      id<MTLCommandBuffer> l87_command=[queue commandBuffer];
      id<MTLComputeCommandEncoder> l87_encoder=
          [l87_command computeCommandEncoder];
      [l87_encoder setComputePipelineState:l87_pipeline];
      [l87_encoder setBuffer:pyramid[0] offset:G atIndex:0];
      [l87_encoder setBuffer:rom_buffer offset:G atIndex:1];
      [l87_encoder setBuffer:tail87_params_buffer offset:0 atIndex:2];
      [l87_encoder setTexture:tail_mask_output atIndex:0];
      [l87_encoder setTexture:tail_color atIndex:1];
      [l87_encoder dispatchThreads:MTLSizeMake(640,360,1)
          threadsPerThreadgroup:MTLSizeMake(16,16,1)];
      [l87_encoder endEncoding];
      Wait(l87_command);
      const auto final_frame=TailReadTexture(tail_color,4);

      std::array<std::string, 68> checkpoints{};
      std::size_t canaries = 0;
      for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto bytes = Payload(tensors[index],
                                   PrefixTensorBytes(kPrefixTensors[index]));
        checkpoints[index] = Sha256(bytes);
        canaries += CanaryMismatches(tensors[index],
                                     PrefixTensorBytes(kPrefixTensors[index]));
      }
      const auto l52_a_bytes = ConnectedReadTexture(l52_a, 4);
      const auto l52_b_bytes = ConnectedReadTexture(l52_b, 4);
      const auto l52_confidence_bytes = ConnectedReadTexture(l52_confidence, 2);
      const auto l52_join =
          JoinVec({l52_a_bytes, l52_b_bytes, l52_confidence_bytes});
      checkpoints[32] = Sha256(l52_join);
      const auto l53_a_bytes = Payload(l53_a, m53::kOutputBytes);
      const auto l53_b_bytes = Payload(l53_b, m53::kOutputBytes);
      const auto l53_join = JoinVec({l53_a_bytes, l53_b_bytes});
      checkpoints[33] = Sha256(l53_join);
      const auto l54_bytes = ConnectedReadTexture(l54_blend, 4);
      checkpoints[34] = Sha256(l54_bytes);
      checkpoints[35] = Sha256(l55_join_before_tail);
      checkpoints[36]=Sha256(Payload(l56,c::kL56Bytes));
      checkpoints[37]=Sha256(Payload(l57,c::kL57Bytes));
      checkpoints[38]=Sha256(Payload(l58,c::kL58Bytes));
      const std::array<id<MTLBuffer>,9> low64 = {
          l59,l60,l61,l62,l63,l64,l65,l66,l67};
      for(std::size_t index=0;index<low64.size();++index)
        checkpoints[39+index]=Sha256(Payload(low64[index],c::kLow64Bytes));
      checkpoints[48]=Sha256(Concat({{l68a,c::kL68PlaneBytes},
                                     {l68b,c::kL68PlaneBytes},
                                     {l68c,c::kL68PlaneBytes}}));
      checkpoints[49]=Sha256(Concat({{l69a,c::kL69Out0Bytes},
                                     {l69b,c::kL69Out1Bytes},
                                     {l69c,c::kL69Out2Bytes}}));
      checkpoints[50]=Sha256(Concat({{l70a,c::kL70Out0Bytes},
                                     {l70b,c::kL70Out1Bytes},
                                     {l70c,c::kL70Out2Bytes}}));
      checkpoints[51]=Sha256(Payload(l71,c::kL71Bytes));
      checkpoints[52]=Sha256(Payload(l72,c::kL72Bytes));
      const std::array<id<MTLBuffer>,9> block1 = {
          l73,l74,l75,l76,l77,l78,l79,l80,l81};
      for(std::size_t index=0;index<block1.size();++index)
        checkpoints[53+index]=Sha256(Payload(block1[index],c::kBlock1Bytes));
      checkpoints[62]=Sha256(Concat({{l82a,c::kL82PlaneBytes},
                                     {l82b,c::kL82PlaneBytes},
                                     {l82c,c::kL82PlaneBytes}}));
      checkpoints[63]=Sha256(l83_before_tail);
      checkpoints[64]=Sha256(checkpoint84);
      checkpoints[65]=Sha256(checkpoint85);
      checkpoints[66]=Sha256(checkpoint86);
      checkpoints[67]=Sha256(final_frame);
      std::size_t mismatches =
          Mismatches(prefix_input_before, Payload(prefix_input,kInputBytes)) +
          Mismatches(depth_before, ConnectedReadTexture(depth, 1)) +
          Mismatches(rom, Payload(rom_buffer, rom.size()));
      for (std::size_t index = 0; index < l53_input_before.size(); ++index) {
        const std::array<id<MTLTexture>,4> textures = {
            l53_flow_a,l53_flow_b,l53_depth_a,l53_depth_b};
        const std::array<NSUInteger,4> bytes_per_pixel = {8,4,4,4};
        mismatches += Mismatches(
            l53_input_before[index],
            ConnectedReadTexture(textures[index], bytes_per_pixel[index]));
      }
      mismatches += Mismatches(l55_input.rgba_a,
                               ConnectedReadTexture(rgba_a, 8));
      mismatches += Mismatches(l55_input.rgba_b,
                               ConnectedReadTexture(rgba_b, 8));
      mismatches += Mismatches(l55_input.motion,
                               ConnectedReadTexture(motion, 4));
      mismatches += Mismatches(l55_linear_a_before_tail,
                               Payload(linear_a,m55::kLinearRgbaBytes));
      mismatches += Mismatches(l55_linear_b_before_tail,
                               Payload(linear_b,m55::kLinearRgbaBytes));
      mismatches += Mismatches(l55_linear_mask_before_tail,
                               Payload(linear_mask,m55::kLinearMaskBytes));
      mismatches += Mismatches(l55_surface_a_before_tail,
                               std::vector<std::uint8_t>(
                                   l84_color_a_before_tail.begin(),
                                   l84_color_a_before_tail.begin()+
                                       m55::kSurfaceRgbaBytes));
      mismatches += Mismatches(l55_surface_b_before_tail,
                               std::vector<std::uint8_t>(
                                   l84_color_b_before_tail.begin(),
                                   l84_color_b_before_tail.begin()+
                                       m55::kSurfaceRgbaBytes));
      mismatches += Mismatches(l84_color_a_before_tail,
                               Payload(l84_color_a,m55::kLinearRgbaBytes));
      mismatches += Mismatches(l84_color_b_before_tail,
                               Payload(l84_color_b,m55::kLinearRgbaBytes));
      mismatches += Mismatches(tail_mask_before,
                               Payload(tail_mask,kTailMaskBytes));
      mismatches += Mismatches(sigmoid,
                               Payload(tail_sigmoid,kSigmoidBytes));
      mismatches += Mismatches(l83_before_tail,Payload(l83,c::kL83Bytes));
      std::size_t readonly_tail_mismatches=0;
      for(std::size_t index=1;index<pyramid84.size();++index)
        readonly_tail_mismatches+=Mismatches(pyramid84[index],pyramid86[index]);
      readonly_tail_mismatches+=Mismatches(checkpoint85,pyramid86[4]);
      for(std::size_t index=1;index<l85_output.size();++index)
        readonly_tail_mismatches+=Mismatches(
            std::vector<std::uint8_t>(l85_sizes[index],poison),
            Payload(l85_output[index],l85_sizes[index]));
      mismatches+=readonly_tail_mismatches;
      canaries += CanaryMismatches(prefix_input, kInputBytes) +
                  CanaryMismatches(rom_buffer, rom.size());
      canaries += CanaryMismatches(l53_a, m53::kOutputBytes) +
                  CanaryMismatches(l53_b, m53::kOutputBytes) +
                  CanaryMismatches(high_a, high_initial.size()) +
                  CanaryMismatches(low_a, low_initial.size()) +
                  CanaryMismatches(high_b, high_initial.size()) +
                  CanaryMismatches(low_b, low_initial.size()) +
                  CanaryMismatches(linear_a, m55::kLinearRgbaBytes) +
                  CanaryMismatches(linear_b, m55::kLinearRgbaBytes) +
                  CanaryMismatches(linear_mask, m55::kLinearMaskBytes) +
                  CanaryMismatches(l84_color_a,m55::kLinearRgbaBytes) +
                  CanaryMismatches(l84_color_b,m55::kLinearRgbaBytes);
      const std::array<std::pair<id<MTLBuffer>,std::size_t>,36> tail_buffers = {{
          {l56,c::kL56Bytes},{l57,c::kL57Bytes},{l58,c::kL58Bytes},
          {l59,c::kLow64Bytes},{l60,c::kLow64Bytes},{l61,c::kLow64Bytes},
          {l62,c::kLow64Bytes},{l63,c::kLow64Bytes},{l64,c::kLow64Bytes},
          {l65,c::kLow64Bytes},{l66,c::kLow64Bytes},{l67,c::kLow64Bytes},
          {l68a,c::kL68PlaneBytes},{l68b,c::kL68PlaneBytes},
          {l68c,c::kL68PlaneBytes},{l69a,c::kL69Out0Bytes},
          {l69b,c::kL69Out1Bytes},{l69c,c::kL69Out2Bytes},
          {l70a,c::kL70Out0Bytes},{l70b,c::kL70Out1Bytes},
          {l70c,c::kL70Out2Bytes},{l71,c::kL71Bytes},{l72,c::kL72Bytes},
          {l73,c::kBlock1Bytes},{l74,c::kBlock1Bytes},{l75,c::kBlock1Bytes},
          {l76,c::kBlock1Bytes},{l77,c::kBlock1Bytes},{l78,c::kBlock1Bytes},
          {l79,c::kBlock1Bytes},{l80,c::kBlock1Bytes},{l81,c::kBlock1Bytes},
          {l82a,c::kL82PlaneBytes},{l82b,c::kL82PlaneBytes},
          {l82c,c::kL82PlaneBytes},{l83,c::kL83Bytes}}};
      for(const auto& item:tail_buffers)
        canaries+=CanaryMismatches(item.first,item.second);
      canaries+=CanaryMismatches(tail_mask,kTailMaskBytes)+
                CanaryMismatches(tail_sigmoid,kSigmoidBytes)+
                CanaryMismatches(tail_scratch,kTailScratchBytes);
      for(std::size_t index=0;index<pyramid.size();++index){
        canaries+=CanaryMismatches(pyramid[index],kPyramidBytes[index]);
        canaries+=CanaryMismatches(l85_output[index],l85_sizes[index]);
      }
      for (std::size_t index = 0; index < weights.size(); ++index) {
        mismatches += Mismatches(*weight_before[index],
                                Payload(weights[index],
                                        kPrefixWeights[index].bytes));
        canaries += CanaryMismatches(weights[index],
                                     kPrefixWeights[index].bytes);
      }
      for(std::size_t index=0;index<tail_weight.size();++index){
        mismatches+=Mismatches(*tail_weight_before[index],
                               Payload(tail_weight[index],
                                       c::kFullModels[index].weight_bytes));
        mismatches+=Mismatches(*tail_bias_before[index],
                               Payload(tail_bias[index],
                                       c::kFullModels[index].bias_bytes));
        canaries+=CanaryMismatches(tail_weight[index],
                                   c::kFullModels[index].weight_bytes);
        canaries+=CanaryMismatches(tail_bias[index],
                                   c::kFullModels[index].bias_bytes);
      }
      const bool pass = mismatches == 0 && canaries == 0;
      Write(output / "l51.bin",
            Payload(tensors[31], PrefixTensorBytes(kPrefixTensors[31])));
      Write(output / "l52.bin", l52_join);
      Write(output / "l53.bin", l53_join);
      Write(output / "l54.bin", l54_bytes);
      Write(output / "l55.bin", l55_join_before_tail);
      Write(output / "l83.bin", l83_before_tail);
      Write(output / "l84.bin", checkpoint84);
      Write(output / "l85.bin", checkpoint85);
      Write(output / "l86.bin", checkpoint86);
      Write(output / "output.bin", final_frame);
      std::ofstream receipt(output / "run-receipt.json");
      receipt << "{\n  \"schema\": \"neuralstudio.fg.connected-l20-l87-run.v1\",\n"
              << "  \"backend\": \"mac-metal\",\n"
              << "  \"run_id\": \"" << argv[6] << "\",\n"
              << "  \"process_id\": " << getpid() << ",\n"
              << "  \"poison_byte\": " << unsigned(poison) << ",\n"
              << "  \"external_flow_scenario\": \"small-flow\",\n"
              << "  \"selected_module_census_root_sha256\": \"324e579b133caf212ad9db44908a73ae6baa90ffbb4978696b51ae99cb114c26\",\n"
              << "  \"allocation_provenance_root_sha256\": \"8a687084ddbc9c0ebeefd4a12dabb98162691bb8dbf62a5505e531ef64c9c75c\",\n"
              << "  \"descriptor_provenance_root_sha256\": \"d3c72b68dcec1d1bb307239c7a782a4c49e06c92b121e8e74a1f0a63ba509733\",\n"
              << "  \"launch_range\": [20, 87],\n"
              << "  \"launch_count\": 68,\n"
              << "  \"checkpoint_sha256\": {";
      for (std::size_t index = 0; index < checkpoints.size(); ++index)
        receipt << (index ? "," : "") << "\n    \"launch-"
                << std::setfill('0') << std::setw(6) << (20u + index)
                << "\": \"" << checkpoints[index] << "\"";
      receipt << "\n  },\n  \"input_model_mismatched_bytes\": "
              << mismatches << ",\n  \"canary_mismatched_bytes\": "
              << canaries << ",\n  \"readonly_tail_mismatched_bytes\": "
              << readonly_tail_mismatches << ",\n"
              << "  \"live_device_handoff_count\": 67,\n"
              << "  \"host_intermediate_ingress\": false,\n"
              << "  \"expected_or_archived_tensor_ingress\": false,\n"
              << "  \"l84_color_edge\": \"l55-surface-a-b-live-gpu-blit\",\n"
              << "  \"address_fields_recorded\": false,\n"
              << "  \"final_output_bytes\": " << final_frame.size() << ",\n"
              << "  \"final_output_sha256\": \"" << checkpoints.back()
              << "\",\n  \"result\": \"" << (pass ? "pass" : "fail")
              << "\"\n}\n";
      std::cout << "l51_sha256=" << checkpoints[31] << '\n'
                << "l52_sha256=" << checkpoints[32] << '\n'
                << "l53_sha256=" << checkpoints[33] << '\n'
                << "l54_sha256=" << checkpoints[34] << '\n'
                << "l55_sha256=" << checkpoints[35] << '\n'
                << "l83_sha256=" << checkpoints[63] << '\n'
                << "l84_sha256=" << checkpoints[64] << '\n'
                << "l85_sha256=" << checkpoints[65] << '\n'
                << "l86_sha256=" << checkpoints[66] << '\n'
                << "l87_sha256=" << checkpoints[67] << '\n'
                << "result=" << (pass ? "pass" : "fail") << '\n';
      return pass ? 0 : 3;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
