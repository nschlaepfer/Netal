// NR ViT-1D block probe: runs the Metal kernels for one recorded ViT block on the oracle's block input
// and compares every stage with the recorded RTX output. Stage inputs come from the previous stage's
// live Metal output unless --isolated is given (then every stage reads the recorded input, for triage).
// Outputs are poisoned, guarded, coverage-counted; inputs are hashed before and after.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <unistd.h>

namespace fs = std::filesystem;
constexpr size_t Guard = 256;
struct Buffer { id<MTLBuffer> metal; size_t bytes; };
struct Params { uint32_t tokens, block, stage, reserved; };

static std::vector<uint8_t> Read(const fs::path& p) {
  std::ifstream s(p, std::ios::binary | std::ios::ate);
  if (!s) throw std::runtime_error("open failed: " + p.string());
  const size_t n = static_cast<size_t>(s.tellg()); s.seekg(0); std::vector<uint8_t> v; v.resize(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n));
  if (!s) throw std::runtime_error("read failed: " + p.string()); return v;
}
static std::string Sha(const void* d, size_t n) {
  std::array<unsigned char, 32> h{}; CC_SHA256(d, CC_LONG(n), h.data());
  std::ostringstream o; o << std::hex << std::setfill('0'); for (auto b : h) o << std::setw(2) << unsigned(b); return o.str();
}
static uint8_t* Data(Buffer& b) { return static_cast<uint8_t*>(b.metal.contents) + Guard; }
static Buffer Make(id<MTLDevice> dev, size_t bytes, uint8_t poison) {
  Buffer b{[dev newBufferWithLength:bytes + 2 * Guard options:MTLResourceStorageModeShared], bytes};
  if (!b.metal) throw std::runtime_error("buffer allocation failed");
  std::memset(b.metal.contents, 0xc3, bytes + 2 * Guard); std::memset(Data(b), poison, bytes); return b;
}
static Buffer Upload(id<MTLDevice> dev, const std::vector<uint8_t>& v) { Buffer b = Make(dev, v.size(), 0); std::memcpy(Data(b), v.data(), v.size()); return b; }
static size_t Canary(Buffer& b) {
  auto* p = static_cast<uint8_t*>(b.metal.contents); size_t e = 0;
  for (size_t i = 0; i < Guard; ++i) e += (p[i] != 0xc3) + (p[Guard + b.bytes + i] != 0xc3); return e;
}
static size_t Coverage(Buffer& c, size_t n) { auto* u = reinterpret_cast<uint32_t*>(Data(c)); size_t e = 0; for (size_t i = 0; i < n; ++i) e += u[i] != 1; return e; }
static size_t Mismatch(Buffer& b, const std::vector<uint8_t>& expect, size_t& first) {
  size_t m = 0; first = size_t(-1); auto* d = Data(b);
  for (size_t i = 0; i < expect.size(); ++i) if (d[i] != expect[i]) { if (!m) first = i; ++m; }
  return m;
}

int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 6) throw std::runtime_error("usage: vit_probe LIB FIXTURE_DIR BLOCK POISON_HEX OUTPUT_DIR [--isolated] [--stages expand,contract,qkv,attention,projection]");
  fs::path lib = argv[1], fixtures = argv[2]; const int block = std::stoi(argv[3]);
  const unsigned poison = unsigned(std::stoul(argv[4], nullptr, 16)); if (poison > 255) throw std::runtime_error("poison out of range");
  fs::path output = argv[5]; if (fs::exists(output)) throw std::runtime_error("preserve previous run: " + output.string());
  bool isolated = false; std::string stages = "expand,contract,qkv,attention,projection"; fs::path tokens_override;
  for (int i = 6; i < argc; ++i) { std::string a = argv[i]; if (a == "--isolated") isolated = true; else if (a == "--stages" && i + 1 < argc) stages = argv[++i]; else if (a == "--input-tokens" && i + 1 < argc) tokens_override = argv[++i]; }
  auto want = [&](const char* s) { return stages.find(s) != std::string::npos; };
  const fs::path bd = fixtures / ("block" + std::to_string(block));
  const auto tokens = Read(tokens_override.empty() ? bd / "tokens_in.e4m3" : tokens_override), w0 = Read(bd / "weights.layer0.bin"), w1 = Read(bd / "weights.layer1.bin"), w2 = Read(bd / "weights.layer2.bin"), w4 = Read(bd / "weights.layer4.bin");
  const auto exp_e = Read(bd / "expand.e4m3"), con_e = Read(bd / "contract.e4m3"), q_e = Read(bd / "q.e4m3"), k_e = Read(bd / "k.e4m3"), v_e = Read(bd / "v.e4m3"), att_e = Read(bd / "attention.e4m3"), proj_e = Read(bd / "projection.e4m3");
  const auto rsq = Read(fixtures / "rtx-rsqrt-half.bin"), rcp = Read(fixtures / "rtx-rcp-half.bin");
  const uint32_t T = 640;
  if (tokens.size() != T * 1024 || exp_e.size() != T * 4096 || w0.size() != 4194320 || w1.size() != 4196352 || w2.size() != 3145856 || w4.size() != 1050624) throw std::runtime_error("fixture extents");

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> library = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib.c_str()]] error:&err];
  if (!library) throw std::runtime_error(err.localizedDescription.UTF8String);
  auto pso = [&](const char* name) {
    id<MTLComputePipelineState> s = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
    if (!s) throw std::runtime_error(std::string(name) + ": " + err.localizedDescription.UTF8String); return s;
  };
  id<MTLComputePipelineState> ps_expand = pso("nr_vit_expand"), ps_contract = pso("nr_vit_contract"), ps_qkv = pso("nr_vit_qkv"), ps_attention = pso("nr_vit_attention"), ps_projection = pso("nr_vit_projection");
  id<MTLCommandQueue> queue = [dev newCommandQueue];

  Buffer tok = Upload(dev, tokens), W0 = Upload(dev, w0), W1 = Upload(dev, w1), W2 = Upload(dev, w2), W4 = Upload(dev, w4), RSQ = Upload(dev, rsq), RCP = Upload(dev, rcp);
  Buffer expand = Make(dev, T * 4096, uint8_t(poison)), contract = Make(dev, T * 1024, uint8_t(poison)), q = Make(dev, T * 1024, uint8_t(poison)), k = Make(dev, T * 1024, uint8_t(poison)), v = Make(dev, T * 1024, uint8_t(poison)), attention = Make(dev, T * 1024, uint8_t(poison)), projection = Make(dev, T * 1024, uint8_t(poison));
  Buffer cov_expand = Make(dev, T * 4096 * 4, 0), cov_contract = Make(dev, T * 1024 * 4, 0), cov_q = Make(dev, T * 1024 * 4, 0), cov_k = Make(dev, T * 1024 * 4, 0), cov_v = Make(dev, T * 1024 * 4, 0), cov_att = Make(dev, T * 1024 * 4, 0), cov_proj = Make(dev, T * 1024 * 4, 0);
  // isolated mode: stage inputs are the recorded tensors
  Buffer exp_in = isolated ? Upload(dev, exp_e) : expand, con_in = isolated ? Upload(dev, con_e) : contract, q_in = isolated ? Upload(dev, q_e) : q, k_in = isolated ? Upload(dev, k_e) : k, v_in = isolated ? Upload(dev, v_e) : v, att_in = isolated ? Upload(dev, att_e) : attention;
  const auto before = std::map<std::string, std::string>{{"tokens", Sha(tokens.data(), tokens.size())}, {"w0", Sha(w0.data(), w0.size())}, {"w1", Sha(w1.data(), w1.size())}, {"w2", Sha(w2.data(), w2.size())}, {"w4", Sha(w4.data(), w4.size())}};
  Params prm{T, uint32_t(block), 0, 0};

  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  auto dispatch = [&](id<MTLComputePipelineState> ps, std::vector<Buffer*> bufs, size_t threads, uint32_t stage) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:ps];
    for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i]->metal offset:Guard atIndex:i];
    prm.stage = stage; [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()];
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
  };
  if (want("expand")) dispatch(ps_expand, {&tok, &W0, &expand, &cov_expand}, size_t(T) * 256, 1);
  if (want("contract")) dispatch(ps_contract, {&exp_in, &tok, &W1, &contract, &cov_contract}, size_t(T) * 64, 2);
  if (want("qkv")) dispatch(ps_qkv, {&con_in, &W2, &q, &k, &v, &cov_q, &cov_k, &cov_v, &RSQ}, size_t(T) * 32, 3);
  if (want("attention")) dispatch(ps_attention, {&q_in, &k_in, &v_in, &attention, &cov_att, &RCP}, size_t(T) * 32, 4);
  if (want("projection")) dispatch(ps_projection, {&att_in, &con_in, &W4, &projection, &cov_proj}, size_t(T) * 64, 5);
  const auto t0 = [NSDate date];
  [cmd commit]; [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed");
  const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0, wall_ms = -[t0 timeIntervalSinceNow] * 1000.0;

  fs::create_directories(output);
  std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.vit-block-probe.v1\",\n \"tokens_source\": \"" << (tokens_override.empty() ? std::string("fixture") : tokens_override.string()) << "\",\n \"block\": " << block << ",\n \"poison\": " << poison << ",\n \"pid\": " << getpid() << ",\n \"isolated\": " << (isolated ? "true" : "false") << ",\n \"gpu_ms\": " << gpu_ms << ",\n \"wall_ms\": " << wall_ms << ",\n \"stages\": {\n";
  bool ok = true, firstStage = true;
  auto report = [&](const char* name, Buffer& out, Buffer& cov, const std::vector<uint8_t>& expect, bool ran) {
    if (!ran) return;
    size_t first = 0; const size_t mism = Mismatch(out, expect, first), cover = Coverage(cov, expect.size()), canary = Canary(out) + Canary(cov);
    ok = ok && !mism && !cover && !canary;
    std::ofstream(output / (std::string(name) + ".bin"), std::ios::binary).write(reinterpret_cast<char*>(Data(out)), out.bytes);
    js << (firstStage ? "" : ",\n") << "  \"" << name << "\": {\"bytes\": " << expect.size() << ", \"mismatches\": " << mism << ", \"first_mismatch\": " << (mism ? long(first) : -1) << ", \"coverage_errors\": " << cover << ", \"canary_errors\": " << canary << ", \"sha256\": \"" << Sha(Data(out), out.bytes) << "\", \"expected_sha256\": \"" << Sha(expect.data(), expect.size()) << "\"}";
    firstStage = false;
    std::cout << name << ": mismatches " << mism << "/" << expect.size() << " coverage_errors " << cover << " canary " << canary << "\n";
  };
  report("expand", expand, cov_expand, exp_e, want("expand")); report("contract", contract, cov_contract, con_e, want("contract"));
  report("q", q, cov_q, q_e, want("qkv")); report("k", k, cov_k, k_e, want("qkv")); report("v", v, cov_v, v_e, want("qkv"));
  report("attention", attention, cov_att, att_e, want("attention")); report("projection", projection, cov_proj, proj_e, want("projection"));
  size_t immutable = 0;
  immutable += std::memcmp(Data(tok), tokens.data(), tokens.size()) != 0; immutable += std::memcmp(Data(W0), w0.data(), w0.size()) != 0; immutable += std::memcmp(Data(W1), w1.data(), w1.size()) != 0; immutable += std::memcmp(Data(W2), w2.data(), w2.size()) != 0; immutable += std::memcmp(Data(W4), w4.data(), w4.size()) != 0;
  immutable += Canary(tok) + Canary(W0) + Canary(W1) + Canary(W2) + Canary(W4) + Canary(RSQ) + Canary(RCP);
  ok = ok && !immutable;
  js << "\n },\n \"immutable_input_errors\": " << immutable << ",\n \"inputs_sha256\": {";
  bool f = true; for (auto& kv : before) { js << (f ? "" : ", ") << "\"" << kv.first << "\": \"" << kv.second << "\""; f = false; }
  js << "},\n \"result\": \"" << (ok ? "pass" : "fail") << "\"\n}\n";
  std::ofstream(output / "run.json") << js.str();
  std::cout << "gpu_ms " << gpu_ms << " wall_ms " << wall_ms << " result " << (ok ? "pass" : "fail") << "\n";
  return ok ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "vit_probe: " << e.what() << "\n"; return 2; } } }
