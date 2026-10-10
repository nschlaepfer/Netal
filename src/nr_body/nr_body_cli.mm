// nr_body_cli.mm — offline harness adapter for the v1 runtime: loads the chain descriptor and the live-input FILES named in it into Metal buffers /
// a texture (this is the only place files become tensors), runs nr_body_run_frame, prints the same "done <code> <ms>" lines as the probe in --serve
// mode. Comparisons with references stay in the Python drivers. Command line identical to middle_chain_probe.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
#include <cstdlib>
#include <mach-o/dyld.h>
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <string>
#include <vector>
static std::vector<uint8_t> Read(const std::string& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
static std::string ShaB(const void* p, size_t n) { unsigned char d[32]; CC_SHA256(p, (CC_LONG)n, d); char h[65]; for (int i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", d[i]); return std::string(h, 64); }
static std::string SelfSha() { char buf[4096]; uint32_t n = sizeof(buf); if (_NSGetExecutablePath(buf, &n) != 0) return ""; auto v = Read(buf); return ShaB(v.data(), v.size()); }
static std::string JQ(const std::string& s) { std::string o = "\""; for (char ch : s) { if (ch == '"' || ch == '\\') o += '\\'; o += ch; } return o + "\""; }
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static long JInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.longValue : 0; }
static int run_one(NrBodyContext& ctx, const char* chain_path, const char* out_path) {
  NSError* err = nil; NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:chain_path]] options:0 error:&err]; if (!chain) { std::cerr << "chain json\n"; return 2; }
  auto up = [&](const std::vector<uint8_t>& v) { id<MTLBuffer> b = [ctx.dev newBufferWithLength:v.size() + 16 options:MTLResourceStorageModeShared]; std::memcpy(b.contents, v.data(), v.size()); return b; };
  NrBodyFrameInputs in; in.poison = uint8_t(std::stoul(JStr(chain, "poison"), nullptr, 16) & 0xffu);
  try {
    { auto e = Read(JStr(chain, "entry_file")); in.entry = {up(e), 0, e.size(), ShaB(e.data(), e.size()), "file:" + JStr(chain, "entry_file")}; }
    if (NSDictionary* D = chain[@"decoder"]) {
      in.production = JStr(D, "checks") == "final"; NSArray* patches = D[@"patches"];
      for (unsigned i = 0; i < patches.count && i < 5; ++i) { NSDictionary* P = patches[i]; auto v = Read(JStr(P, "file")); NrBodyRegion r{up(v), 0, v.size(), ShaB(v.data(), v.size()), "file:" + JStr(P, "file")}; if (i < 4) in.skips[i] = r; else in.block0_skip = r; }
      const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); auto texin = Read(JStr(D, "tex_in"));
      MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
      in.input_color = [ctx.dev newTextureWithDescriptor:d]; [in.input_color replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:texin.data() bytesPerRow:tw * 8];
    }
  } catch (const std::exception& e) { std::cerr << "nr_body_cli: " << e.what() << "\n"; return 2; }
  if (const char* f = getenv("NR_BODY_FAULT")) in.test_fault = f;   // fault-injection tests (test builds only; production runtime rejects it)
  NrBodyResult res; const int rc = nr_body_run_frame(ctx, chain, in, (out_path && std::string(out_path) != "-") ? out_path : nullptr, res);   // OUTPUT_DIR "-" = no diagnostics
  if (rc) { std::cerr << "nr_body_cli: code " << rc << ": " << res.error << "\n"; for (auto& f : res.check_failures) std::cerr << "  check failed: " << f << "\n"; }
  else std::cout << "frame ok: middle " << res.middle_gpu_ms << " ms, decoder " << res.decoder_gpu_ms << " ms, final " << res.final_surface_sha256.substr(0, 16) << " pixels_not_written " << res.final_pixels_not_written << "\n";
  // execution-time provenance manifest (v3): executable identity, descriptor identity, live-input identities, every library/constant hashed at load, post-frame re-hash verdict
  { std::ostringstream m; auto cf = Read(chain_path); m << "{\n \"schema\": \"neuralstudio.nr.body-provenance-manifest.v1\",\n \"executable_sha256\": " << JQ(SelfSha()) << ",\n \"descriptor\": " << JQ(chain_path) << ",\n \"descriptor_sha256\": " << JQ(ShaB(cf.data(), cf.size()))
      << ",\n \"code\": " << rc << ",\n \"checks_enforced\": " << res.checks_enforced << ",\n \"constants_rehashed_after_frame\": " << res.constants_rehashed << ",\n \"constants_unchanged_after_frame\": " << (res.constants_unchanged_after_frame ? "true" : "false")
      << ",\n \"live_inputs\": {\"entry\": " << JQ(in.entry.provenance_sha256) << ", \"skips\": [" << JQ(in.skips[0].provenance_sha256) << ", " << JQ(in.skips[1].provenance_sha256) << ", " << JQ(in.skips[2].provenance_sha256) << ", " << JQ(in.skips[3].provenance_sha256) << "], \"block0_skip\": " << JQ(in.block0_skip.provenance_sha256) << ", \"input_texture\": " << JQ(res.input_texture_sha256_before) << "}"
      << ",\n \"final_surface_sha256\": " << JQ(res.final_surface_sha256) << ",\n \"loaded\": {"; bool f1 = true; for (auto& kv : res.provenance) { m << (f1 ? "\n  " : ",\n  ") << JQ(kv.first) << ": " << JQ(kv.second); f1 = false; } m << "\n }\n}\n";
    if (out_path && std::string(out_path) != "-") std::ofstream(std::string(out_path) + "/manifest.json") << m.str(); else if (const char* mp = getenv("NR_BODY_MANIFEST")) std::ofstream(mp) << m.str(); }
  return rc;
}
int main(int argc, const char** argv) {
  if (argc < 2) { std::cerr << "usage: nr_body_cli CHAIN.json OUTPUT_DIR | nr_body_cli --serve\n"; return 2; }
  NrBodyContext ctx; ctx.dev = MTLCreateSystemDefaultDevice(); ctx.queue = [ctx.dev newCommandQueue];
  if (std::string(argv[1]) == "--serve") { std::string line; while (std::getline(std::cin, line)) { if (line.empty()) continue; const size_t sp = line.find(' '); if (sp == std::string::npos) { std::cout << "done 2 0\n" << std::flush; continue; }
      const auto t0 = [NSDate date]; const int rc = run_one(ctx, line.substr(0, sp).c_str(), line.substr(sp + 1).c_str()); std::cout << "done " << rc << " " << (-[t0 timeIntervalSinceNow] * 1000.0) << "\n" << std::flush; } return 0; }
  if (argc < 3) { std::cerr << "usage: nr_body_cli CHAIN.json OUTPUT_DIR\n"; return 2; }
  return run_one(ctx, argv[1], argv[2]);
}
