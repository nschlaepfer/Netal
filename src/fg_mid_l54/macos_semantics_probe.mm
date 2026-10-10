#define FG_CONV_L20_NO_MAIN
#include "../fg_conv/macos_probe.mm"

namespace {

constexpr std::uint32_t kDenseEx2Count = (2u << 21u) + 1u;
constexpr std::size_t kDenseEx2Bytes =
    std::size_t{kDenseEx2Count} * sizeof(std::uint32_t);

}  // namespace

int main(int argc, const char* argv[]) {
  @autoreleasepool {
    if (argc != 3) {
      std::cerr << "usage=fg-mid-l54-mac-semantics <metallib> <output>\n";
      return 2;
    }
    try {
      id<MTLDevice> device = MTLCreateSystemDefaultDevice();
      if (device == nil) throw std::runtime_error("Metal device unavailable");
      NSError* error = nil;
      id<MTLLibrary> library = [device
          newLibraryWithURL:[NSURL fileURLWithPath:
                                      [NSString stringWithUTF8String:argv[1]]]
                     error:&error];
      if (library == nil) throw std::runtime_error("metallib load failed");
      id<MTLFunction> function =
          [library newFunctionWithName:@"fg_mid_l54_dense_ex2"];
      if (function == nil) throw std::runtime_error("Metal function missing");
      id<MTLComputePipelineState> pipeline =
          [device newComputePipelineStateWithFunction:function error:&error];
      if (pipeline == nil) throw std::runtime_error("pipeline creation failed");
      id<MTLBuffer> output =
          [device newBufferWithLength:kDenseEx2Bytes
                              options:MTLResourceStorageModeShared];
      if (output == nil) throw std::runtime_error("output allocation failed");
      std::memset(output.contents, 0xA5, kDenseEx2Bytes);
      id<MTLCommandQueue> queue = [device newCommandQueue];
      id<MTLCommandBuffer> command = [queue commandBuffer];
      id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
      [encoder setComputePipelineState:pipeline];
      [encoder setBuffer:output offset:0 atIndex:0];
      [encoder dispatchThreads:MTLSizeMake(kDenseEx2Count, 1, 1)
          threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      [encoder endEncoding];
      [command commit];
      [command waitUntilCompleted];
      if (command.status != MTLCommandBufferStatusCompleted) {
        throw std::runtime_error("Metal command failed");
      }
      std::vector<std::uint8_t> bytes(kDenseEx2Bytes);
      std::memcpy(bytes.data(), output.contents, bytes.size());
      Write(argv[2], bytes);
      std::cout << "records=" << kDenseEx2Count << '\n'
                << "bytes=" << bytes.size() << '\n'
                << "sha256=" << Sha256(bytes) << '\n'
                << "result=pass\n";
      return 0;
    } catch (const std::exception& error) {
      std::cerr << "error=" << error.what() << '\n';
      return 1;
    }
  }
}
