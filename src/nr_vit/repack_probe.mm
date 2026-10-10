// Runs the two generated repack kernels on the recorded boundary tensors and compares with the RTX output.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <unistd.h>
namespace fs = std::filesystem;
constexpr size_t Guard = 256;
struct Buffer { id<MTLBuffer> metal; size_t bytes; };
struct Params { int width, height; uint32_t in_dwords, out_dwords; };
static std::vector<uint8_t> Read(const fs::path& p) {
  std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p.string());
  const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v; v.resize(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v;
}
static uint8_t* Data(Buffer& b) { return static_cast<uint8_t*>(b.metal.contents) + Guard; }
static Buffer Make(id<MTLDevice> dev, size_t bytes, uint8_t poison) {
  Buffer b{[dev newBufferWithLength:bytes + 2 * Guard options:MTLResourceStorageModeShared], bytes};
  std::memset(b.metal.contents, 0xc3, bytes + 2 * Guard); std::memset(Data(b), poison, bytes); return b;
}
static size_t Canary(Buffer& b) { auto* p = static_cast<uint8_t*>(b.metal.contents); size_t e = 0; for (size_t i = 0; i < Guard; ++i) e += (p[i] != 0xc3) + (p[Guard + b.bytes + i] != 0xc3); return e; }
int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 5) throw std::runtime_error("usage: repack_probe LIB FIXTURE_DIR POISON_HEX OUTPUT_DIR [--in58 PATH] [--in99 PATH] [--only 58|99] [--require-live]");
  fs::path lib = argv[1], fx = argv[2], out = argv[4]; const unsigned poison = unsigned(std::stoul(argv[3], nullptr, 16));
  fs::path in58, in99; int only = 0; bool require_live = false;      // live inputs for the connected cross-family run (default: the recorded fixture); --only runs one direction; --require-live forbids fixture inputs
  for (int i = 5; i < argc; ++i) { std::string a = argv[i]; if (a == "--in58" && i + 1 < argc) in58 = argv[++i]; else if (a == "--in99" && i + 1 < argc) in99 = argv[++i]; else if (a == "--only" && i + 1 < argc) only = std::stoi(argv[++i]); else if (a == "--require-live") require_live = true; else throw std::runtime_error("unknown argument " + a); }
  if (only && only != 58 && only != 99) throw std::runtime_error("--only takes 58 or 99");
  if (fs::exists(out)) throw std::runtime_error("preserve previous run"); fs::create_directories(out);
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> library = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib.c_str()]] error:&err];
  if (!library) throw std::runtime_error(err.localizedDescription.UTF8String);
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  bool ok = true; std::ofstream js(out / "run.json"); js << "{\n \"schema\": \"neuralstudio.nr.vit-repack-probe.v1\",\n \"poison\": " << poison << ",\n \"pid\": " << getpid() << ",\n \"input_sources\": {\"repack58\": \"" << (in58.empty() ? std::string("fixture") : in58.string()) << "\", \"repack99\": \"" << (in99.empty() ? std::string("fixture") : in99.string()) << "\"},\n \"kernels\": {\n";
  struct Case { const char* kernel; const char* in; const char* expect; size_t in_bytes, out_bytes; };
  const Case cases[] = {{"nr_vit_repack_2d_to_1d", "repack58_in.bin", "repack58_out.e4m3", 720896, 655360}, {"nr_vit_repack_1d_to_2d", "repack99_in.e4m3", "repack99_out.bin", 655360, 655360}};
  bool first = true;
  for (const auto& c : cases) {
    const bool is58 = std::string(c.in) == "repack58_in.bin";
    if (only && ((only == 58) != is58)) continue;                     // single-direction execution: the other kernel never runs
    const fs::path override = is58 ? in58 : in99;
    if (require_live && override.empty()) throw std::runtime_error(std::string("strict: no live input for ") + c.kernel);
    auto in = Read(override.empty() ? fx / c.in : override), expect = Read(fx / c.expect); if (in.size() != c.in_bytes || expect.size() != c.out_bytes) throw std::runtime_error("fixture extent");
    id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:c.kernel]] error:&err];
    if (!ps) throw std::runtime_error(err.localizedDescription.UTF8String);
    Buffer bin = Make(dev, in.size(), 0), bout = Make(dev, c.out_bytes, uint8_t(poison)), cov = Make(dev, c.out_bytes, 0); std::memcpy(Data(bin), in.data(), in.size());
    Params prm{20, 32, uint32_t(in.size() / 4), uint32_t(c.out_bytes / 4)};
    id<MTLCommandBuffer> cmd = [queue commandBuffer]; id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder];
    [enc setComputePipelineState:ps]; [enc setBuffer:bin.metal offset:Guard atIndex:0]; [enc setBuffer:bout.metal offset:Guard atIndex:1]; [enc setBuffer:cov.metal offset:Guard atIndex:2]; [enc setBytes:&prm length:sizeof(prm) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(640, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)]; [enc endEncoding]; [cmd commit]; [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error("command buffer failed");
    size_t mism = 0, first_m = size_t(-1); for (size_t i = 0; i < expect.size(); ++i) if (Data(bout)[i] != expect[i]) { if (!mism) first_m = i; ++mism; }
    size_t cover = 0; auto* cu = reinterpret_cast<uint32_t*>(Data(cov)); for (size_t i = 0; i < c.out_bytes / 4; ++i) cover += cu[i] != 1;
    const size_t canary = Canary(bin) + Canary(bout) + Canary(cov); const bool immutable = std::memcmp(Data(bin), in.data(), in.size()) == 0;
    ok = ok && !mism && !cover && !canary && immutable;
    std::ofstream(out / (std::string(c.kernel) + ".bin"), std::ios::binary).write(reinterpret_cast<char*>(Data(bout)), c.out_bytes);
    js << (first ? "" : ",\n") << "  \"" << c.kernel << "\": {\"mismatches\": " << mism << ", \"first_mismatch\": " << long(mism ? first_m : -1) << ", \"coverage_errors_dwords\": " << cover << ", \"canary_errors\": " << canary << ", \"input_immutable\": " << (immutable ? "true" : "false") << "}";
    first = false; std::cout << c.kernel << ": mismatches " << mism << "/" << expect.size() << " coverage_errors " << cover << " canary " << canary << "\n";
  }
  js << "\n },\n \"result\": \"" << (ok ? "pass" : "fail") << "\"\n}\n"; std::cout << "result " << (ok ? "pass" : "fail") << "\n"; return ok ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "repack_probe: " << e.what() << "\n"; return 2; } } }
