// nr_post_replay.mm — ISOLATED diagnostic replay of launches 155 (post block, temporal) and 156 (copy) on an ARCHIVED second-frame pool state
// captured on RTX (Codex packet). Explicitly archived-component diagnostics: never a live whole-model producer, never acceptance evidence.
// The launch-155 record is CONSTRUCTED from the accepted frame-0 record by the verified rule and checked word-for-word (outside pointer slots)
// against the captured record; captured pointers are never used. Textures: colour (input), history, motion (RG16F), output (poisoned), history_after.
// Usage: nr_post_replay --pool F --params-base F --params-captured F --weights155 F --params156 F --weights156 F --tables DIR --post-lib F --copy-lib F
//        --color F --history F --motion F --expected F --out DIR
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_temporal/nr_temporal.h"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
static std::string Sha(const void* p, size_t n) { unsigned char d[32]; CC_SHA256(p, (CC_LONG)n, d); char h[65]; for (int i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", d[i]); return std::string(h, 64); }
static std::vector<uint8_t> Read(const std::string& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
int main(int argc, const char** argv) { @autoreleasepool { try {
  std::map<std::string, std::string> a; for (int i = 1; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
  for (const char* k : {"--pool", "--params-base", "--params-captured", "--weights155", "--params156", "--weights156", "--tables", "--post-lib", "--copy-lib", "--color", "--history", "--motion", "--expected", "--out"}) if (!a.count(k)) throw std::runtime_error(std::string("missing ") + k);
  const unsigned W = 1920, H = 1080; const uint8_t poison = 0xa5; id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> q = [dev newCommandQueue]; NSError* err = nil;
  auto up = [&](const std::vector<uint8_t>& v, size_t pad) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + pad options:MTLResourceStorageModeShared]; std::memset(b.contents, 0, v.size() + pad); std::memcpy(b.contents, v.data(), v.size()); return b; };
  auto lib = [&](const std::string& p) { id<MTLLibrary> l = [dev newLibraryWithURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:p.c_str()]] error:&err]; if (!l) throw std::runtime_error("lib " + p); return l; };
  auto pso = [&](id<MTLLibrary> l, const char* n) { id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:[l newFunctionWithName:[NSString stringWithUTF8String:n]] error:&err]; if (!p) throw std::runtime_error(std::string("pso ") + n); return p; };
  auto texf = [&](MTLPixelFormat f, const std::vector<uint8_t>* bytes, unsigned bpp) { MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:f width:W height:H mipmapped:NO]; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared; id<MTLTexture> t = [dev newTextureWithDescriptor:d];
    std::vector<uint8_t> pz(size_t(W) * H * bpp, poison); const std::vector<uint8_t>& src = bytes ? *bytes : pz; if (src.size() != size_t(W) * H * bpp) throw std::runtime_error("texture size"); [t replaceRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0 withBytes:src.data() bytesPerRow:W * bpp]; return t; };
  auto texbytes = [&](id<MTLTexture> t, unsigned bpp) { std::vector<uint8_t> o(size_t(W) * H * bpp); [t getBytes:o.data() bytesPerRow:W * bpp fromRegion:MTLRegionMake2D(0, 0, W, H) mipmapLevel:0]; return o; };
  // inputs
  auto pool_bytes = Read(a["--pool"]); if (pool_bytes.size() != 212519936u) throw std::runtime_error("pool size"); const std::string pool_sha = Sha(pool_bytes.data(), pool_bytes.size());
  auto base = Read(a["--params-base"]); auto captured = Read(a["--params-captured"]); if (base.size() != 184 || captured.size() != 184) throw std::runtime_error("155 records must be 184 bytes");
  std::vector<uint8_t> rec(184); if (!nr_temporal_build_post_record(base.data(), 184, W, H, 1u, 6u, rec.data())) throw std::runtime_error("record construction");
  const size_t ptr_words[] = {0, 4, 8, 12, 20, 24, 28, 60, 104, 108}; std::vector<size_t> diff;
  for (size_t off = 0; off < 184; off += 4) { bool skip = false; for (size_t pw : ptr_words) skip |= (pw == off); if (!skip && std::memcmp(rec.data() + off, captured.data() + off, 4) != 0) diff.push_back(off); }
  if (!diff.empty()) { std::ostringstream e; e << "constructed record differs from the captured second-frame record at offsets"; for (size_t d : diff) e << " " << d; throw std::runtime_error(e.str()); }
  auto w155 = Read(a["--weights155"]), w156 = Read(a["--weights156"]), p156 = Read(a["--params156"]); auto color = Read(a["--color"]), hist = Read(a["--history"]), motion = Read(a["--motion"]), expected = Read(a["--expected"]);
  const std::string tables = a["--tables"]; auto tbl = [&](const char* n) { return up(Read(tables + "/" + n), 16); };
  id<MTLBuffer> bex2h = nil; std::string ex2h_sha; if (a.count("--ex2-half-table")) { auto t = Read(a["--ex2-half-table"]); ex2h_sha = Sha(t.data(), t.size()); if (t.size() != 524288 || ex2h_sha != "f9ae941e60868e72489db582fe725bdb401e033fce8694ecd35dd0c3329ba3cf") throw std::runtime_error("ex2 half-domain table: size/sha mismatch against the pinned hash"); bex2h = up(t, 16); }
  id<MTLBuffer> bfault = [dev newBufferWithLength:64 options:MTLResourceStorageModeShared]; std::memset(bfault.contents, 0, 64);
  id<MTLBuffer> bpool = up(pool_bytes, 4096), bw = up(w155, 4096), bprm = up(rec, 64), brsq = tbl("rtx-rsqrt-half.bin"), brcp = tbl("rtx-rcp-half.bin"), brom = tbl("sm120-rcp-correction-rom.bin"), bex2u = tbl("unit-corrections.bin"), bex2f = tbl("fraction-corrections.bin");
  std::vector<uint8_t> dbgz(64 * 32 * 4, 0); id<MTLBuffer> bdbg = up(dbgz, 0); id<MTLBuffer> bcov = [dev newBufferWithLength:pool_bytes.size() + 4096 options:MTLResourceStorageModeShared]; id<MTLBuffer> btexcov = [dev newBufferWithLength:size_t(W) * H * 4 + 16 options:MTLResourceStorageModeShared];
  id<MTLTexture> tcolor = texf(MTLPixelFormatRGBA16Float, &color, 8), thist = texf(MTLPixelFormatRGBA16Float, &hist, 8), tmotion = texf(MTLPixelFormatRG16Float, &motion, 4), tout = texf(MTLPixelFormatRGBA16Float, nullptr, 8), thist_after = texf(MTLPixelFormatRGBA16Float, nullptr, 8);
  const std::string color_sha = Sha(color.data(), color.size()), hist_sha = Sha(hist.data(), hist.size()), motion_sha = Sha(motion.data(), motion.size()), pool_pre_sha = pool_sha;
  auto run = [&](id<MTLLibrary> l, const char* kernel, MTLSize grid, MTLSize block, id<MTLBuffer> wb, id<MTLBuffer> pb, id<MTLTexture> t0, id<MTLTexture> t1, id<MTLTexture> t2, id<MTLTexture> t3) {
    id<MTLCommandBuffer> cmd = [q commandBuffer]; { id<MTLBlitCommandEncoder> bl = [cmd blitCommandEncoder]; [bl fillBuffer:bcov range:NSMakeRange(0, pool_bytes.size() + 4096) value:0]; [bl fillBuffer:btexcov range:NSMakeRange(0, size_t(W) * H * 4 + 16) value:0]; [bl endEncoding]; }
    id<MTLComputeCommandEncoder> enc = [cmd computeCommandEncoder]; [enc setComputePipelineState:pso(l, kernel)];
    [enc setBuffer:bpool offset:0 atIndex:0]; [enc setBuffer:wb offset:0 atIndex:1]; [enc setBuffer:pb offset:0 atIndex:2]; [enc setBuffer:brsq offset:0 atIndex:3]; [enc setBuffer:brcp offset:0 atIndex:4]; [enc setBuffer:bdbg offset:0 atIndex:5]; [enc setBuffer:bcov offset:0 atIndex:6]; [enc setBuffer:brom offset:0 atIndex:7]; [enc setBuffer:bex2u offset:0 atIndex:8]; [enc setBuffer:bex2f offset:0 atIndex:9]; [enc setBuffer:btexcov offset:0 atIndex:10]; if (bex2h) { [enc setBuffer:bex2h offset:0 atIndex:11]; [enc setBuffer:bfault offset:0 atIndex:12]; }
    [enc setTexture:t0 atIndex:0]; [enc setTexture:t1 atIndex:1]; if (t2) [enc setTexture:t2 atIndex:2]; if (t3) [enc setTexture:t3 atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:block]; [enc endEncoding]; const auto t0s = [NSDate date]; [cmd commit]; [cmd waitUntilCompleted]; if (cmd.status != MTLCommandBufferStatusCompleted) throw std::runtime_error(std::string("command failed: ") + kernel);
    const uint32_t* tc = static_cast<const uint32_t*>(btexcov.contents); size_t once = 0, multi = 0; for (size_t i = 0; i < size_t(W) * H; ++i) { if (tc[i] == 1) ++once; else if (tc[i] > 1) ++multi; }
    return std::make_tuple(-[t0s timeIntervalSinceNow] * 1000.0, once, multi); };
  // launch 155 (temporal): textures 0 colour, 1 OUT (caller output), 2 motion, 3 history
  auto [ms155, once155, multi155] = run(lib(a["--post-lib"]), "ptx_post_block_swin_1h_32", MTLSizeMake(241, 145, 1), MTLSizeMake(32, 1, 1), bw, bprm, tcolor, tout, tmotion, thist);
  auto outb = texbytes(tout, 8); const std::string out_sha = Sha(outb.data(), outb.size()); const std::string exp_sha = Sha(expected.data(), expected.size());
  size_t mism = 0, first_shown = 0; std::ostringstream firsts; for (size_t i = 0; i + 1 < outb.size(); i += 2) if (outb[i] != expected[i] || outb[i + 1] != expected[i + 1]) { ++mism; if (first_shown < 8) { const size_t v = i / 2, px = v / 4; firsts << (first_shown ? ", " : "") << "{\"pixel\": [" << px % W << ", " << px / W << "], \"channel\": " << v % 4 << ", \"got\": \"" << std::hex << (unsigned(outb[i]) | (unsigned(outb[i + 1]) << 8)) << "\", \"expected\": \"" << (unsigned(expected[i]) | (unsigned(expected[i + 1]) << 8)) << std::dec << "\"}"; ++first_shown; } }
  const bool color_ok = Sha(texbytes(tcolor, 8).data(), color.size()) == color_sha, hist_ok = Sha(texbytes(thist, 8).data(), hist.size()) == hist_sha, motion_ok = Sha(texbytes(tmotion, 4).data(), motion.size()) == motion_sha, w_ok = Sha(bw.contents, w155.size()) == Sha(w155.data(), w155.size()), p_ok = Sha(bprm.contents, 184) == Sha(rec.data(), 184);
  const std::string pool_post_sha = Sha(bpool.contents, pool_bytes.size());
  // launch 156 (copy): textures 0 = caller output, 1 = history_after
  auto [ms156, once156, multi156] = run(lib(a["--copy-lib"]), "ptx_cg2r_copy", MTLSizeMake(120, 68, 1), MTLSizeMake(16, 16, 1), up(w156, 4096), up(p156, 64), tout, thist_after, nil, nil);
  auto hab = texbytes(thist_after, 8); const std::string hist_after_sha = Sha(hab.data(), hab.size());
  const uint32_t* fl = static_cast<const uint32_t*>(bfault.contents); const uint32_t ex2_arg_faults = fl[0], ex2_domain_faults = fl[1];   // half-domain EX2 lookups (0 unless --ex2-half-table)
  const bool pass = mism == 0 && out_sha == exp_sha && once155 == size_t(W) * H && multi155 == 0 && color_ok && hist_ok && motion_ok && w_ok && p_ok && hist_after_sha == out_sha && once156 == size_t(W) * H && multi156 == 0 && ex2_arg_faults == 0 && ex2_domain_faults == 0;
  std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.post155-isolated-replay.v1\",\n \"label\": \"ISOLATED archived-component diagnostic replay of launches 155+156 on the RTX-captured second-frame pool; not a live producer, not whole-model acceptance\",\n"
     << " \"post_lib\": \"" << a["--post-lib"] << "\", \"copy_lib\": \"" << a["--copy-lib"] << "\",\n \"inputs\": {\"pool_sha256\": \"" << pool_sha << "\", \"color_sha256\": \"" << color_sha << "\", \"history_sha256\": \"" << hist_sha << "\", \"motion_sha256\": \"" << motion_sha << "\", \"weights155_sha256\": \"" << Sha(w155.data(), w155.size()) << "\", \"constructed_record_sha256\": \"" << Sha(rec.data(), 184) << "\", \"captured_record_sha256\": \"" << Sha(captured.data(), 184) << "\", \"record_equal_outside_pointer_slots\": true},\n"
     << " \"launch155\": {\"gpu_ms\": " << ms155 << ", \"pixels_once\": " << once155 << ", \"pixels_multi\": " << multi155 << ", \"output_sha256\": \"" << out_sha << "\", \"expected_sha256\": \"" << exp_sha << "\", \"mismatching_values\": " << mism << ", \"first_mismatches\": [" << firsts.str() << "], \"color_unchanged\": " << (color_ok ? "true" : "false") << ", \"history_unchanged\": " << (hist_ok ? "true" : "false") << ", \"motion_unchanged\": " << (motion_ok ? "true" : "false") << ", \"weights_unchanged\": " << (w_ok ? "true" : "false") << ", \"record_unchanged\": " << (p_ok ? "true" : "false") << ", \"pool_pre_sha256\": \"" << pool_pre_sha << "\", \"pool_post_sha256\": \"" << pool_post_sha << "\"},\n"
     << " \"ex2_half_table\": {\"file\": \"" << (bex2h ? a["--ex2-half-table"] : std::string()) << "\", \"sha256\": \"" << ex2h_sha << "\", \"argument_faults\": " << ex2_arg_faults << ", \"domain_faults\": " << ex2_domain_faults << "},\n"
     << " \"launch156\": {\"gpu_ms\": " << ms156 << ", \"pixels_once\": " << once156 << ", \"pixels_multi\": " << multi156 << ", \"history_after_sha256\": \"" << hist_after_sha << "\", \"history_after_equals_output\": " << (hist_after_sha == out_sha ? "true" : "false") << "},\n \"result\": \"" << (pass ? "pass" : "fail") << "\"\n}\n";
  std::string od = a["--out"]; system(("mkdir -p '" + od + "'").c_str()); std::ofstream(od + "/replay.json") << js.str(); std::ofstream(od + "/output.rgba16le", std::ios::binary).write(reinterpret_cast<const char*>(outb.data()), std::streamsize(outb.size())); std::cout << js.str(); return pass ? 0 : 1;
} catch (const std::exception& e) { std::cerr << "nr_post_replay: " << e.what() << "\n"; return 2; } } }
