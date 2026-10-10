// NR split-Swin family probe: runs the Metal kernels for one recorded launch and compares the output with
// the recorded RTX bytes. Inputs default to the launch's fixture files and can be replaced by live outputs
// of earlier launches (--in ROLE=PATH) for the connected chain. Outputs are poisoned, guarded and
// coverage-counted; inputs are hashed before and after.
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
constexpr size_t TIN = 2160u * 512u, VIEW = 16u * 69120u, POOL = 640u * 512u, VIT = 640u * 1024u;
struct Buffer { id<MTLBuffer> metal; size_t bytes; };
struct Params { int32_t shift_x, shift_y; uint32_t flags, tokens; };
enum : uint32_t { FLAG_RES_VIEW = 1, FLAG_OUT_VIEW = 2, FLAG_EMIT_F16 = 4, FLAG_IN_VIEW = 8 };

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
// pooled/ViT grid: 8 tiles per row, row-major 4x4 slots, valid when R < 18 && C < 30
static bool VitTokenValid(size_t token) { const size_t tile = token / 16, slot = token % 16; const size_t R = 4 * (tile / 8) + slot / 4, C = 4 * (tile % 8) + slot % 4; return R < 18 && C < 30; }
static size_t ARowIndex(size_t t, size_t c, size_t channels) {   // A-identity layout (mirrors nr_a_index)
  const size_t lane = 4 * (t % 8) + ((c % 32) % 16) / 4, reg = ((t % 16) / 8) + 2 * ((c % 32) / 16);
  return (t / 16) * (16 * channels) + (c / 32) * 512 + lane * 16 + reg * 4 + c % 4;
}

int main(int argc, const char** argv) { @autoreleasepool { try {
  if (argc < 5) throw std::runtime_error("usage: split_probe LIB KERNEL POISON_HEX OUTPUT_DIR --fixture DIR [--in ROLE=PATH]... [--shift SX,SY] [--tables DIR]\n  KERNEL: ffwd | ffwd_view | conv | conv_resview | conv_outview | conv_pool | qkv_attend | final_head | dec_upsample");
  fs::path lib = argv[1]; std::string kernel = argv[2];
  const unsigned poison = unsigned(std::stoul(argv[3], nullptr, 16)); if (poison > 255) throw std::runtime_error("poison out of range");
  fs::path output = argv[4]; if (fs::exists(output)) throw std::runtime_error("preserve previous run: " + output.string());
  fs::path fixture, tables; std::map<std::string, fs::path> overrides; int sx = 0, sy = 0; bool strict_inputs = false;   // --strict-inputs: every tensor input must be a live override; only weights may come from the fixture
  for (int i = 5; i < argc; ++i) {
    std::string a = argv[i];
    if (a == "--fixture" && i + 1 < argc) fixture = argv[++i];
    else if (a == "--tables" && i + 1 < argc) tables = argv[++i];
    else if (a == "--strict-inputs") strict_inputs = true;
    else if (a == "--in" && i + 1 < argc) { std::string kv = argv[++i]; auto eq = kv.find('='); overrides[kv.substr(0, eq)] = kv.substr(eq + 1); }
    else if (a == "--shift" && i + 1 < argc) { std::string s = argv[++i]; auto c = s.find(','); sx = std::stoi(s.substr(0, c)); sy = std::stoi(s.substr(c + 1)); }
    else throw std::runtime_error("unknown argument " + a);
  }
  if (fixture.empty()) throw std::runtime_error("--fixture required"); if (tables.empty()) tables = fixture.parent_path();
  std::map<std::string, std::string> sources;
  auto input = [&](const std::string& role) {
    if (strict_inputs && role != "weights" && !overrides.count(role)) throw std::runtime_error("strict: input role " + role + " has no live producer");
    fs::path p = overrides.count(role) ? overrides[role] : fixture / (role + ".bin"); sources[role] = p.string(); return Read(p);
  };

  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NSError* err = nil;
  id<MTLLibrary> library = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:lib.c_str()]] error:&err];
  if (!library) throw std::runtime_error(err.localizedDescription.UTF8String);
  auto pso = [&](const char* name) {
    id<MTLComputePipelineState> s = [dev newComputePipelineStateWithFunction:[library newFunctionWithName:[NSString stringWithUTF8String:name]] error:&err];
    if (!s) throw std::runtime_error(std::string(name) + ": " + err.localizedDescription.UTF8String); return s;
  };
  id<MTLCommandQueue> queue = [dev newCommandQueue];
  id<MTLCommandBuffer> cmd = [queue commandBuffer];
  Params prm{sx, sy, 0, 2160};
  auto dispatch = [&](id<MTLComputePipelineState> ps, std::vector<Buffer*> bufs, size_t threads) {
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:ps];
    for (size_t i = 0; i < bufs.size(); ++i) [enc setBuffer:bufs[i]->metal offset:Guard atIndex:i];
    [enc setBytes:&prm length:sizeof(prm) atIndex:bufs.size()];
    [enc dispatchThreads:MTLSizeMake(threads, 1, 1) threadsPerThreadgroup:MTLSizeMake(64, 1, 1)]; [enc endEncoding];
  };
  // immutable inputs: (name, bytes) hashed before, compared after
  std::vector<std::pair<std::string, Buffer>> inputs; inputs.reserve(16); std::map<std::string, std::vector<uint8_t>> input_bytes;   // reserved: pointers into it are handed out
  auto in = [&](const std::string& role, size_t expect_bytes) {
    auto v = input(role); if (v.size() != expect_bytes) throw std::runtime_error(role + ": extent " + std::to_string(v.size()) + " != " + std::to_string(expect_bytes));
    inputs.emplace_back(role, Upload(dev, v)); input_bytes[role] = std::move(v); return &inputs.back().second;
  };
  // outputs: (name, buffer, coverage, expected bytes, masked)
  struct Out { std::string name; Buffer buf, cov; std::vector<uint8_t> expect; bool masked; };
  std::vector<Out> outs; outs.reserve(8); std::vector<Buffer> scratch; scratch.reserve(16);
  auto out = [&](const std::string& role, size_t bytes, bool masked) {
    auto e = Read(fixture / (role + ".bin")); if (e.size() != bytes) throw std::runtime_error(role + " expected extent");
    outs.push_back({role, Make(dev, bytes, uint8_t(poison)), Make(dev, bytes * 4, 0), std::move(e), masked}); return outs.size() - 1;
  };
  std::vector<Buffer*> ordered; // built per kernel
  Buffer weights = Upload(dev, input("weights")); const auto wbytes = Read(sources["weights"]);
  if (kernel == "ffwd" || kernel == "ffwd_view") {
    const bool view = kernel == "ffwd_view"; prm.flags = view ? FLAG_IN_VIEW : 0;
    Buffer* x = in(view ? "x_view" : "x", view ? VIEW : TIN); size_t o = out("out", TIN, false);
    dispatch(pso("nr_split_ffwd"), {x, &weights, &outs[o].buf, &outs[o].cov}, 2160u * 8u);
  } else if (kernel == "conv" || kernel == "conv_resview" || kernel == "conv_outview" || kernel == "conv_pool") {
    const bool rv = kernel == "conv_resview", ov = kernel == "conv_outview", pool = kernel == "conv_pool";
    prm.flags = (rv ? FLAG_RES_VIEW : 0) | (ov ? FLAG_OUT_VIEW : 0) | (pool ? FLAG_EMIT_F16 : 0);
    Buffer* x = in("x", TIN); Buffer* res = in(rv ? "res_view" : "res", rv ? VIEW : TIN);
    size_t o = out(ov ? "out_view" : "out", ov ? VIEW : TIN, false);
    scratch.push_back(Make(dev, pool ? TIN * 2 : 64, 0)); Buffer* y16 = &scratch.back();
    dispatch(pso("nr_split_conv"), {x, res, &weights, &outs[o].buf, y16, &outs[o].cov}, 2160u * 32u);
    if (pool) { size_t po = out("pooled", POOL, false); dispatch(pso("nr_split_pool"), {y16, &outs[po].buf, &outs[po].cov}, 640u * 32u); }
  } else if (kernel == "qkv_attend") {
    Buffer* x = in("x", TIN); size_t o = out("out", TIN, false);
    Buffer rsq = Upload(dev, Read(tables / "rtx-rsqrt-half.bin")), rcp = Upload(dev, Read(tables / "rtx-rcp-half.bin"));
    for (int i = 0; i < 3; ++i) scratch.push_back(Make(dev, TIN, uint8_t(poison)));
    Buffer *q = &scratch[scratch.size() - 3], *k = &scratch[scratch.size() - 2], *v = &scratch[scratch.size() - 1];
    dispatch(pso("nr_split_qkv"), {x, &weights, q, k, v, &rsq}, 2160u * 16u);
    dispatch(pso("nr_split_attend"), {q, k, v, &weights, &outs[o].buf, &outs[o].cov, &rcp}, 40u * 16u * 64u);
    inputs.emplace_back("rsqrt_table", rsq); inputs.emplace_back("rcp_table", rcp); input_bytes["rsqrt_table"] = Read(tables / "rtx-rsqrt-half.bin"); input_bytes["rcp_table"] = Read(tables / "rtx-rcp-half.bin");
  } else if (kernel == "final_head") {
    Buffer* x = in("x_pooled", POOL); size_t o = out("out_vit", VIT, true);
    dispatch(pso("nr_split_final_head"), {x, &weights, &outs[o].buf, &outs[o].cov}, 640u * 64u);
  } else if (kernel == "dec_upsample") {
    Buffer* x = in("x_vit", VIT); Buffer* res = in("res", TIN); size_t o = out("out", TIN, false);
    dispatch(pso("nr_split_dec_upsample"), {x, res, &weights, &outs[o].buf, &outs[o].cov}, 2160u * 32u);
  } else throw std::runtime_error("unknown kernel " + kernel);
  std::map<std::string, std::string> before; for (auto& kv : input_bytes) before[kv.first] = Sha(kv.second.data(), kv.second.size()); before["weights"] = Sha(wbytes.data(), wbytes.size());

  const auto t0 = [NSDate date];
  [cmd commit]; [cmd waitUntilCompleted];
  if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(cmd.error ? cmd.error.localizedDescription.UTF8String : "command buffer failed");
  const double gpu_ms = (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0, wall_ms = -[t0 timeIntervalSinceNow] * 1000.0;

  fs::create_directories(output);
  bool ok = true; std::ostringstream js;
  js << "{\n \"schema\": \"neuralstudio.nr.split-probe.v1\",\n \"kernel\": \"" << kernel << "\",\n \"poison\": " << poison << ",\n \"pid\": " << getpid() << ",\n \"shift\": [" << sx << ", " << sy << "],\n \"gpu_ms\": " << gpu_ms << ",\n \"wall_ms\": " << wall_ms << ",\n \"outputs\": {\n";
  bool firstOut = true;
  for (auto& o : outs) {
    auto* d = Data(o.buf); auto* cov = reinterpret_cast<uint32_t*>(Data(o.cov)); size_t mism = 0, first = size_t(-1), coverr = 0, expected_bytes = 0;
    if (!o.masked) {
      for (size_t i = 0; i < o.expect.size(); ++i) { if (d[i] != o.expect[i]) { if (!mism) first = i; ++mism; } coverr += cov[i] != 1; }
      expected_bytes = o.expect.size();
    } else {
      // ViT-input layout: 16384-byte tiles of 16 tokens x 1024 channels; only valid pooled tokens are written
      std::vector<uint8_t> mask(o.expect.size(), 0);
      for (size_t token = 0; token < 640; ++token) if (VitTokenValid(token)) for (size_t c = 0; c < 1024; ++c) mask[ARowIndex(token, c, 1024)] = 1;
      for (size_t i = 0; i < o.expect.size(); ++i) {
        if (mask[i]) { ++expected_bytes; if (d[i] != o.expect[i]) { if (!mism) first = i; ++mism; } coverr += cov[i] != 1; }
        else { coverr += cov[i] != 0; if (d[i] != uint8_t(poison)) { if (!mism) first = i; ++mism; } }
      }
    }
    const size_t canary = Canary(o.buf) + Canary(o.cov); ok = ok && !mism && !coverr && !canary;
    std::ofstream(output / (o.name + ".bin"), std::ios::binary).write(reinterpret_cast<char*>(d), o.buf.bytes);
    js << (firstOut ? "" : ",\n") << "  \"" << o.name << "\": {\"bytes\": " << o.expect.size() << ", \"compared_bytes\": " << expected_bytes << ", \"mismatches\": " << mism << ", \"first_mismatch\": " << (mism ? long(first) : -1) << ", \"coverage_errors\": " << coverr << ", \"canary_errors\": " << canary << ", \"sha256\": \"" << Sha(d, o.buf.bytes) << "\", \"expected_sha256\": \"" << Sha(o.expect.data(), o.expect.size()) << "\"}";
    firstOut = false;
    std::cout << o.name << ": mismatches " << mism << "/" << expected_bytes << " coverage_errors " << coverr << " canary " << canary << "\n";
  }
  size_t immutable = Canary(weights) + (std::memcmp(Data(weights), wbytes.data(), wbytes.size()) != 0);
  for (auto& kv : inputs) { auto& bytes = input_bytes[kv.first]; immutable += Canary(kv.second) + (std::memcmp(Data(kv.second), bytes.data(), bytes.size()) != 0); }
  for (auto& s : scratch) immutable += Canary(s);
  ok = ok && !immutable;
  js << "\n },\n \"immutable_input_errors\": " << immutable << ",\n \"inputs_sha256\": {"; bool f = true;
  for (auto& kv : before) { js << (f ? "" : ", ") << "\"" << kv.first << "\": \"" << kv.second << "\""; f = false; }
  js << "},\n \"input_sources\": {"; f = true;
  for (auto& kv : sources) { js << (f ? "" : ", ") << "\"" << kv.first << "\": \"" << kv.second << "\""; f = false; }
  js << "},\n \"result\": \"" << (ok ? "pass" : "fail") << "\"\n}\n";
  std::ofstream(output / "run.json") << js.str();
  std::cout << "gpu_ms " << gpu_ms << " wall_ms " << wall_ms << " result " << (ok ? "pass" : "fail") << "\n";
  return ok ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "split_probe: " << e.what() << "\n"; return 2; } } }
