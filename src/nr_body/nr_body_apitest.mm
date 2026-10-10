// nr_body_apitest.mm — gate-2 misuse/refusal tests of the public resident API (nr_body_encode / commit / complete / pending_reset) on the archived
// Mac encoder outputs named by a chain descriptor (same loading as nr_body_cli). Every misuse must be refused with code 2 and a "misuse:" error and
// must leave the good frame unaffected; legitimate reuse must work. Fault-test builds add: a failed pending (code 3) refuses reuse until reset, and the
// damaged context refuses every later frame. Usage: nr_body_apitest CHAIN.json OUT.json
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
#include <CommonCrypto/CommonDigest.h>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <type_traits>
#include <utility>
static_assert(!std::is_copy_constructible<NrBodyPending>::value && !std::is_copy_assignable<NrBodyPending>::value, "NrBodyPending must not be copyable (Codex ownership review)");
static_assert(!std::is_copy_constructible<NrBodyContext>::value && !std::is_copy_assignable<NrBodyContext>::value && !std::is_move_constructible<NrBodyContext>::value && !std::is_move_assignable<NrBodyContext>::value, "NrBodyContext must be unique and address-stable (v17)");
static std::vector<uint8_t> Read(const std::string& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
static std::string ShaB(const void* p, size_t n) { unsigned char d[32]; CC_SHA256(p, (CC_LONG)n, d); char h[65]; for (int i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", d[i]); return std::string(h, 64); }
static std::string JQ(const std::string& s) { std::string o = "\""; for (char ch : s) { if (ch == '"' || ch == '\\') o += '\\'; o += ch; } return o + "\""; }
static std::string JStr(NSDictionary* d, const char* k) { NSString* v = d[[NSString stringWithUTF8String:k]]; return v ? std::string(v.UTF8String) : std::string(); }
static long JInt(NSDictionary* d, const char* k) { NSNumber* v = d[[NSString stringWithUTF8String:k]]; return v ? v.longValue : 0; }
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc != 3) { std::cerr << "usage: nr_body_apitest CHAIN.json OUT.json\n"; return 2; }
  NSError* err = nil; NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:&err]; if (!chain) { std::cerr << "chain json\n"; return 2; }
  id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); NrBodyContext ctx; ctx.dev = dev; ctx.queue = [dev newCommandQueue];
  auto up = [&](const std::vector<uint8_t>& v) { id<MTLBuffer> b = [dev newBufferWithLength:v.size() + 16 options:MTLResourceStorageModeShared]; std::memcpy(b.contents, v.data(), v.size()); return b; };
  NrBodyFrameInputs in; in.poison = uint8_t(std::stoul(JStr(chain, "poison"), nullptr, 16) & 0xffu);
  { auto e = Read(JStr(chain, "entry_file")); in.entry = {up(e), 0, e.size(), ShaB(e.data(), e.size()), "file"}; NSDictionary* D = chain[@"decoder"]; in.production = JStr(D, "checks") == "final"; NSArray* patches = D[@"patches"];
    for (unsigned i = 0; i < patches.count && i < 5; ++i) { NSDictionary* P = patches[i]; auto v = Read(JStr(P, "file")); NrBodyRegion r{up(v), 0, v.size(), ShaB(v.data(), v.size()), "file"}; if (i < 4) in.skips[i] = r; else in.block0_skip = r; }
    const unsigned tw = unsigned(JInt(D, "tex_w")), th = unsigned(JInt(D, "tex_h")); auto texin = Read(JStr(D, "tex_in")); MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:tw height:th mipmapped:NO]; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
    in.input_color = [dev newTextureWithDescriptor:d]; [in.input_color replaceRegion:MTLRegionMake2D(0, 0, tw, th) mipmapLevel:0 withBytes:texin.data() bytesPerRow:tw * 8]; }
  std::vector<std::string> out; bool all_ok = true; std::string final0;
  auto rec = [&](const char* name, bool ok, int rc, const std::string& err_text, const std::string& extra = "") { all_ok = all_ok && ok; std::ostringstream o; o << "{\"case\": " << JQ(name) << ", \"ok\": " << (ok ? "true" : "false") << ", \"rc\": " << rc << ", \"error\": " << JQ(err_text) << (extra.empty() ? "" : ", " + extra) << "}"; out.push_back(o.str()); std::cout << (ok ? "PASS " : "FAIL ") << name << " :: rc " << rc << " " << err_text.substr(0, 120) << "\n"; };
  auto is_misuse = [](const NrBodyResult& r) { return r.code == 2 && r.error.rfind("misuse:", 0) == 0; };
  // A. encode twice on one pending -> second refused; the first frame still completes exactly
  { NrBodyPending p; NrBodyResult r1, r2; int a = nr_body_encode(ctx, chain, in, nullptr, p); int b = nr_body_encode(ctx, chain, in, nullptr, p); r2 = p.res; rec("A1-encode-then-encode-refused", a == 0 && b == 2 && is_misuse(r2) && p.state == NrBodyPending::Encoded, b, r2.error);
    int c = nr_body_commit(p); int d = nr_body_complete(p, r1); final0 = r1.final_surface_sha256; rec("A2-first-frame-completes-after-refused-encode", c == 0 && d == 0 && r1.code == 0 && p.state == NrBodyPending::Completed && r1.inventory_command_buffers == 1 && r1.inventory_waits == 1, d, r1.error, "\"final_surface_sha256\": " + JQ(r1.final_surface_sha256) + ", \"validators_run\": " + std::to_string(r1.validators_run) + ", \"digests_complete\": " + std::to_string(r1.digests_complete)); }
  // B. complete before commit -> refused; then commit + complete works
  { NrBodyPending p; NrBodyResult r; int a = nr_body_encode(ctx, chain, in, nullptr, p); int b = nr_body_complete(p, r); rec("B1-complete-before-commit-refused", a == 0 && b == 2 && is_misuse(r) && p.state == NrBodyPending::Encoded && ctx.in_flight, b, r.error);
    int c = nr_body_commit(p); int d = nr_body_complete(p, r); rec("B2-then-commit-complete-exact", c == 0 && d == 0 && r.final_surface_sha256 == final0 && !ctx.in_flight, d, r.error); }
  // C. commit twice -> second refused; complete twice -> second refused
  { NrBodyPending p; NrBodyResult r; nr_body_encode(ctx, chain, in, nullptr, p); int a = nr_body_commit(p); int b = nr_body_commit(p); rec("C1-double-commit-refused", a == 0 && b == 2 && is_misuse(p.res) && p.state == NrBodyPending::Committed, b, p.res.error);
    int c = nr_body_complete(p, r); NrBodyResult r2; int d = nr_body_complete(p, r2); rec("C2-double-complete-refused-first-exact", c == 0 && r.final_surface_sha256 == final0 && d == 2 && is_misuse(r2), d, r2.error); }
  // D. two pendings on one context: the second encode is refused while the first is in flight; allowed after its completion
  { NrBodyPending p1, p2; NrBodyResult r1, r2; int a = nr_body_encode(ctx, chain, in, nullptr, p1); int b = nr_body_encode(ctx, chain, in, nullptr, p2); rec("D1-second-pending-refused-while-in-flight", a == 0 && b == 2 && is_misuse(p2.res) && p2.state == NrBodyPending::Idle, b, p2.res.error);
    nr_body_commit(p1); int c = nr_body_complete(p1, r1); int d = nr_body_encode(ctx, chain, in, nullptr, p2); int e = nr_body_commit(p2); int f = nr_body_complete(p2, r2); rec("D2-second-pending-runs-after-first-completed", c == 0 && d == 0 && e == 0 && f == 0 && r1.final_surface_sha256 == final0 && r2.final_surface_sha256 == final0, f, r2.error); }
  // E. reuse of a completed pending without reset is allowed; commit/complete on an idle pending are refused
  { NrBodyPending p; NrBodyResult r; nr_body_encode(ctx, chain, in, nullptr, p); nr_body_commit(p); nr_body_complete(p, r); int a = nr_body_encode(ctx, chain, in, nullptr, p); int b = nr_body_commit(p); int c = nr_body_complete(p, r); rec("E1-completed-pending-reused", a == 0 && b == 0 && c == 0 && r.final_surface_sha256 == final0, c, r.error);
    NrBodyPending q; NrBodyResult rq; int d = nr_body_commit(q); int e = nr_body_complete(q, rq); rec("E2-commit-and-complete-on-idle-refused", d == 2 && e == 2 && is_misuse(rq) && q.state == NrBodyPending::Idle, e, rq.error); }
  // F. run_frame (the synchronous wrapper) equals the three-step path
  { NrBodyResult r; int a = nr_body_run_frame(ctx, chain, in, nullptr, r); rec("F1-run-frame-wrapper-exact", a == 0 && r.final_surface_sha256 == final0 && r.inventory_command_buffers == 1 && r.inventory_waits == 1, a, r.error); }
  // H. ownership (v16): moved-from handle refused, moved-to handle completes; abandonment of an Encoded handle releases the reservation without damage;
  //    reset while in flight is a no-op; a forged Committed state on a handle that does not own the in-flight frame is refused; abandonment of a
  //    Committed handle damages the context (on a throw-away context)
  { NrBodyPending p1; int a = nr_body_encode(ctx, chain, in, nullptr, p1); NrBodyPending p2 = std::move(p1); int b = nr_body_commit(p1); NrBodyResult rb; int c = nr_body_complete(p1, rb); rec("H1-moved-from-handle-refused", a == 0 && p1.state == NrBodyPending::Moved && b == 2 && is_misuse(p1.res) && c == 2 && is_misuse(rb), c, rb.error);
    NrBodyResult r2; int d = nr_body_commit(p2); int e = nr_body_complete(p2, r2); rec("H2-moved-to-handle-completes-exact", d == 0 && e == 0 && r2.final_surface_sha256 == final0 && !ctx.in_flight && ctx.in_flight_id == 0, e, r2.error); }
  { { NrBodyPending p; int a = nr_body_encode(ctx, chain, in, nullptr, p); rec("H3a-encoded-handle-in-flight", a == 0 && ctx.in_flight, a, p.res.error); } rec("H3b-abandoned-encoded-handle-releases-reservation-no-damage", !ctx.in_flight && ctx.in_flight_id == 0 && ctx.damaged.empty(), 0, ctx.damaged);
    NrBodyResult r; int b = nr_body_run_frame(ctx, chain, in, nullptr, r); rec("H3c-next-frame-exact-after-abandoned-encode", b == 0 && r.final_surface_sha256 == final0, b, r.error); }
  { NrBodyPending p; NrBodyResult r; nr_body_encode(ctx, chain, in, nullptr, p); nr_body_pending_reset(p); rec("H4a-reset-while-in-flight-is-noop", p.state == NrBodyPending::Encoded && ctx.in_flight, 0, ""); int b = nr_body_commit(p); int c = nr_body_complete(p, r); rec("H4b-frame-completes-exact-after-noop-reset", b == 0 && c == 0 && r.final_surface_sha256 == final0, c, r.error); }
  { NrBodyPending p1; NrBodyResult r; int a = nr_body_encode(ctx, chain, in, nullptr, p1); int b, c;
    { NrBodyPending forged; forged.state = NrBodyPending::Committed; forged.ctx = &ctx; forged.ctx_alive = ctx.alive; forged.frame_id = 0; b = nr_body_complete(forged, r); forged.frame_id = p1.frame_id + 7; c = nr_body_complete(forged, r); }   // v17: the forged handle is destroyed AS IS (no field reset): its abandonment must not damage the context
    rec("H5-forged-committed-handle-without-ownership-refused", a == 0 && b == 2 && c == 2 && is_misuse(r) && r.error.find("stale") != std::string::npos && ctx.in_flight, c, r.error);
    rec("H5a-destroyed-forged-handle-does-not-damage-context", ctx.damaged.empty() && ctx.in_flight && ctx.in_flight_id == p1.frame_id, 0, ctx.damaged);
    int d = nr_body_commit(p1); int e = nr_body_complete(p1, r); rec("H5b-owner-still-completes-exact", d == 0 && e == 0 && r.final_surface_sha256 == final0, e, r.error); }
  // I. context lifetime (v17): a context destroyed while a frame is Encoded -> commit refused, pending destructor safe; destroyed while Committed -> complete refused, buffer waited for, no crash
  { NrBodyPending p; int a; { NrBodyContext ctx4; ctx4.dev = dev; ctx4.queue = [dev newCommandQueue]; a = nr_body_encode(ctx4, chain, in, nullptr, p); } int b = nr_body_commit(p); rec("I1-context-destroyed-while-encoded-commit-refused", a == 0 && b == 2 && is_misuse(p.res) && p.res.error.find("context destroyed") != std::string::npos && p.state == NrBodyPending::Failed, b, p.res.error); }
  { NrBodyPending p; NrBodyResult r; int a, b; { NrBodyContext ctx5; ctx5.dev = dev; ctx5.queue = [dev newCommandQueue]; a = nr_body_encode(ctx5, chain, in, nullptr, p); b = nr_body_commit(p); } int c = nr_body_complete(p, r); rec("I2-context-destroyed-while-committed-complete-refused", a == 0 && b == 0 && c == 2 && is_misuse(r) && r.error.find("context destroyed") != std::string::npos && p.state == NrBodyPending::Failed, c, r.error); }
  { NrBodyResult r; int a = nr_body_run_frame(ctx, chain, in, nullptr, r); rec("I3-main-context-still-exact-after-lifetime-cases", a == 0 && r.final_surface_sha256 == final0 && ctx.damaged.empty(), a, r.error); }
  { NrBodyContext ctx3; ctx3.dev = dev; ctx3.queue = [dev newCommandQueue]; { NrBodyPending p; nr_body_encode(ctx3, chain, in, nullptr, p); nr_body_commit(p); } rec("H6a-abandoned-committed-handle-damages-context", !ctx3.damaged.empty() && !ctx3.in_flight && ctx3.in_flight_id == 0, 0, ctx3.damaged);
    NrBodyResult r; int b = nr_body_run_frame(ctx3, chain, in, nullptr, r); rec("H6b-damaged-context-refuses-next-frame", b == 2 && r.error.find("context damaged") != std::string::npos, b, r.error); }
#ifdef NR_BODY_FAULT_TESTS
  // G. a failed frame (code 3 via an in-buffer fault) leaves the pending Failed: encode on it is refused until reset; the context is damaged, so even after reset the next encode is refused; a fresh context recovers
  { NrBodyFrameInputs fin = in; fin.test_fault = "middle-coverage"; NrBodyPending p; NrBodyResult r; int a = nr_body_encode(ctx, chain, fin, nullptr, p); int b = nr_body_commit(p); int c = nr_body_complete(p, r); rec("G1-faulted-frame-code-3", a == 0 && b == 0 && c == 3 && p.state == NrBodyPending::Failed && !ctx.damaged.empty(), c, r.error);
    int d = nr_body_encode(ctx, chain, in, nullptr, p); rec("G2-failed-pending-refuses-reuse", d == 2 && is_misuse(p.res), d, p.res.error);
    nr_body_pending_reset(p); int e = nr_body_encode(ctx, chain, in, nullptr, p); rec("G3-after-reset-damaged-context-refuses", e == 2 && p.res.error.find("context damaged") != std::string::npos && p.state == NrBodyPending::Failed, e, p.res.error);
    NrBodyContext ctx2; ctx2.dev = dev; ctx2.queue = [dev newCommandQueue]; NrBodyPending q; NrBodyResult rq; int f = nr_body_encode(ctx2, chain, in, nullptr, q); int g = nr_body_commit(q); int h = nr_body_complete(q, rq); rec("G4-fresh-context-recovers-exact", f == 0 && g == 0 && h == 0 && rq.final_surface_sha256 == final0, h, rq.error); }
#endif
  std::ofstream f(argv[2]); f << "{\"schema\": \"neuralstudio.nr.body-api-misuse.v3\", \"descriptor\": " << JQ(argv[1]) << ", \"cases\": ["; for (size_t i = 0; i < out.size(); ++i) f << (i ? ",\n" : "\n") << out[i]; f << "\n], \"all_ok\": " << (all_ok ? "true" : "false") << "}\n";
  std::cout << "all_ok " << (all_ok ? "true" : "false") << "\n"; return all_ok ? 0 : 1;
} }
