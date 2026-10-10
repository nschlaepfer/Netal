// Experimental NR runtime: implementation of netal_nr_experimental (src/nr_experimental/nr_experimental_api.h).
// Owner: Opus. EXPERIMENTAL LANE - original weights and kernels, research validation removed. Never an
// exact-parity claim, and this file must never make one.
//
// Contract honoured here: encode-only onto the caller's command buffer; no commit, no wait, no CPU read of
// any current-frame GPU value; a rejection happens BEFORE anything is encoded; an admitted frame yields a
// ticket that must be finalised from the same command buffer's completion callback.
//
// Admission is deliberately conservative: while any stage of the model is not yet ported to the encode-only
// path, encode() REJECTS BEFORE ENCODING and says exactly which stage is missing. It never encodes a partial
// chain, never returns an identity or stub image, and never reports model_encoded for work it did not do.
#include "../nr_experimental_api.h"
#include "nr_experimental_assets.h"
#include "nr_experimental_block0.h"
#include "nr_experimental_encoder.h"
#include "nr_body_runtime.h"
#include "nr_experimental_gpu_profile.h"
#include "nr_experimental_encoder_scope_adoption.h"
#include "nr_experimental_boundary_diag.h"
#include "nr_experimental_context.h"

#include <CommonCrypto/CommonDigest.h>

namespace nr_exp {
static BoundaryHashes g_boundary;
bool boundary_hashes_requested() {
  static const bool on = [] { const char* v = std::getenv("NR_EXP_BOUNDARY_HASHES"); return v && v[0] == '1'; }();
  return on;
}
const BoundaryHashes& last_boundary_hashes() { return g_boundary; }
static BodyChunk g_body_chunk;
const BodyChunk& body_chunk() { return g_body_chunk; }
void set_body_chunk(const BodyChunk& b) { g_body_chunk = b; }
static ChainBytes g_chain_bytes;
const ChainBytes& last_chain_bytes() { return g_chain_bytes; }
void set_last_chain_bytes(const ChainBytes& b) { g_chain_bytes = b; }
void set_last_boundary_hashes(const BoundaryHashes& b) { g_boundary = b; }
static std::string diag_sha(const void* p, size_t n) {
  unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(p, CC_LONG(n), d);
  static const char* H = "0123456789abcdef"; std::string o; o.reserve(64);
  for (unsigned char c : d) { o.push_back(H[c >> 4]); o.push_back(H[c & 15]); }
  return o;
}
}  // namespace nr_exp

#include "nr_temporal.h"
#import <Foundation/Foundation.h>
#include <atomic>
#include <cstdlib>
#include <chrono>
#include <mutex>

namespace netal_nr_experimental {

// Stages of the pinned model, in execution order, and whether each has an encode-only implementation yet.
struct Stage { const char* name; bool encode_only_ready; };
static const Stage kStages[] = {
  {"block0 (features + 34,560-window Swin-32 + export)", true},
  {"encoder blocks 1-4 (C32)", true},
  {"encoder blocks 5-8 (wide 64)", true},
  {"encoder blocks 9-14 (C128)", true},
  {"encoder blocks 15-22 (C256) and the native skip/entry exports", true},
  {"body: middle, ViT, split, decoder, post block, output copy", true},
};
static const char* first_missing_stage() {
  for (const auto& s : kStages) if (!s.encode_only_ready) return s.name;
  return nullptr;
}


struct Ticket {
  std::weak_ptr<Context> context;
  NrBodyPending pending;                   // owns the body's per-frame state until complete() runs
  id<MTLCommandBuffer> command = nil;      // retained so the ticket cannot outlive an unknown buffer
  id<MTLTexture> output = nil;
  uint64_t stream_id = 0, frame_id = 0;
  EncodeStatus status = EncodeStatus::RejectedBeforeEncoding;
  bool finalized = false;
  uint64_t continuity_generation = 0;      // captured at encode; a later invalidate() bumps the context's
  uint32_t dispatches = 0;
};

// Every discard-only path must give the caller the SAME instruction. The command buffer carries the game's own
// rendering and MetalFX; it is always committable and must be committed. Only the NR output is discarded.
// Used ONLY where this runtime can account for every encoder it opened, so the committable claim is
// backed by close_all() having actually closed them. An adoption fault cannot use it: see below.
static const char* kIndeterminateEncoderGuidance =
    " An encoder on this command buffer could NOT be accounted for, so this runtime does NOT claim the "
    "buffer is in a committable state and will not end an encoder it does not own - it may be another "
    "holder's live encoder or already ended. Do NOT publish the NR output. NR history is invalidated; the "
    "next admitted frame must be a reset. Treat the command buffer's encoder state as indeterminate and "
    "check it before committing.";

static const char* kDiscardOnlyGuidance =
    " Commit and complete the command buffer as normal - it is in a committable state and the game's own work "
    "must finish - but do NOT publish the NR output for this frame. NR history is invalidated; the next "
    "admitted frame must be a reset.";

static const char* arithmetic_label(Arithmetic a) {
  return a == Arithmetic::ApproximateNative ? "experimental-approximate-native (unchanged weights, native GPU arithmetic)"
                                            : "experimental-original-kernels (unchanged weights, research validation removed)";
}

bool prepare(id<MTLDevice> device, const PrepareOptions& options, ContextPtr& out, Report& report) {
  const auto t0 = std::chrono::steady_clock::now();
  report = Report{};
  report.arithmetic_label = arithmetic_label(options.arithmetic);
  report.experimental = true;
  if (!device) { report.error = "no Metal device"; return false; }
  if (options.width != 1920 || options.height != 1080) {
    report.error = "this lane is pinned to 1920x1080; got " + std::to_string(options.width) + "x" + std::to_string(options.height);
    return false;
  }
  if (options.runtime_bundle.empty()) { report.error = "runtime_bundle (pinned asset root) required"; return false; }
  auto ctx = std::make_shared<Context>();
  ctx->device = device; ctx->options = options;
  std::string error;
  ctx->bundle = options.runtime_bundle;
  // ApproximateNative loads a SEPARATE library set built from the same sources with the native fp32 FMA
  // accumulation instead of the exact fixed-point QMMA emulation. Same weights, same graph, same dispatches.
  // Its libraries differ from the pinned ones by construction, so the load is explicitly marked unpinned.
  const bool approximate = options.arithmetic == Arithmetic::ApproximateNative;
  const std::string encoder_root = options.runtime_bundle + (approximate ? "/encoder-approx" : "/encoder");
  // The approximate mode also swaps the body's decoder and post libraries, which carry their own QMMA
  // emulation. Both halves must move together, or the mode would be approximate in one half only.
  const std::string body_dir = options.runtime_bundle + (approximate ? "/body-approx" : "/body");
  const std::string descriptor_path = body_dir + "/chain.json";
  if (!nr_exp::load_assets(device, encoder_root, ctx->assets, error, approximate)) {
    report.error = approximate ? ("approximate library set: " + error + " (expected at " + encoder_root + ")") : error;
    return false;
  }
  // A substituted kernel must never be presentable as the pinned configuration, even when every
  // library in the bundle matched its pin. The per-kernel override replaces exactly one function, so
  // say so - naming the function, the version and the tiers - rather than reporting "original kernels".
  if (ctx->assets.qkv_override_active) {
    std::string tiers;
    for (const auto& kv : ctx->assets.qkv_override_sha256)
      tiers += (tiers.empty() ? "" : ", ") + kv.first + "=" + kv.second.substr(0, 12);
    // The FULL digest of the exact manifest bytes that were parsed, not a prefix and not just a version
    // string: two manifests can share a version and differ in what they declare, so the version alone is
    // not a run identity. Emitted in full so an execution receipt identifies one manifest exactly.
    std::string fns;
    for (const auto& f : ctx->assets.qkv_override_functions) fns += (fns.empty() ? "" : ",") + f;
    report.arithmetic_label += " + EXPERIMENTAL per-kernel override of {" + fns + "}" +
                               " v" + ctx->assets.qkv_override_version +
                               " manifest-sha256=" + ctx->assets.qkv_override_manifest_sha256 +
                               " [" + tiers + "]";
  }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  if (ctx->assets.encdense_active) {
    std::string libs;
    for (const auto& kv : ctx->assets.encdense_sha256) libs += (libs.empty() ? "" : ", ") + kv.first + "=" + kv.second.substr(0, 12);
    char mask[16]; snprintf(mask, sizeof mask, "0x%x", ctx->assets.encdense_tier_mask);
    report.arithmetic_label += " + EXPERIMENTAL encoder dense stages {expand,mid,mix,qkv,project} v" + ctx->assets.encdense_version +
                               " manifest-sha256=" + ctx->assets.encdense_manifest_sha256 + " tiers=" + mask + " [" + libs + "]";
  }
#endif
  // EXPERIMENTAL body override, loaded fail-closed before any frame is encoded. Absent env: nothing is
  // read, nothing is loaded, and the body is bit-for-bit its accepted configuration.
  if (const char* broot = std::getenv("NR_EXP_BODY_OVERRIDE")) {
    if (broot[0]) {
      const std::string dir = broot, mpath = dir + "/body-override.json";
      NSData* md = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:mpath.c_str()]];
      if (!md) { report.error = "body override: manifest unreadable at " + mpath; return false; }
      unsigned char d[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(md.bytes, (CC_LONG)md.length, d);
      char hx[65]; for (int i = 0; i < 32; ++i) snprintf(hx + 2 * i, 3, "%02x", d[i]);
      ctx->body.override.manifest_sha256 = hx;
      NSDictionary* m = [NSJSONSerialization JSONObjectWithData:md options:0 error:nil];
      if (![m isKindOfClass:[NSDictionary class]]) { report.error = "body override: manifest is not a JSON object"; return false; }
      NSString* schema = m[@"schema"];
      if (![schema isKindOfClass:[NSString class]] ||
          strcmp([schema UTF8String], "neuralstudio.nr.body-override.v1") != 0) {
        report.error = "body override: unsupported manifest schema"; return false; }
      NSString* ver = m[@"version"];
      if (![ver isKindOfClass:[NSString class]] || [ver length] == 0) { report.error = "body override: manifest has no version"; return false; }
      ctx->body.override.version = [ver UTF8String];
      NSDictionary* libs = m[@"libraries"];
      if (![libs isKindOfClass:[NSDictionary class]] || libs.count == 0) { report.error = "body override: manifest declares no libraries"; return false; }
      for (NSString* rel in libs) {
        const std::string relc = [rel UTF8String];
        NSDictionary* e = libs[rel];
        if (![e isKindOfClass:[NSDictionary class]]) { report.error = "body override: bad entry for " + relc; return false; }
        NSString* want = e[@"sha256"]; NSArray* fns = e[@"functions"];
        if (![want isKindOfClass:[NSString class]] || ![fns isKindOfClass:[NSArray class]] || fns.count == 0) {
          report.error = "body override: " + relc + " needs sha256 and a non-empty functions array"; return false; }
        // The package file defaults to the library's basename. Two descriptor libraries can share a basename
        // (the ViT and split three-product engines are both tp.metallib), so an entry may name its file explicitly;
        // the name must be a plain file inside the package, never a path.
        std::string file = dir + "/" + relc.substr(relc.find_last_of('/') == std::string::npos ? 0 : relc.find_last_of('/') + 1);
        if (NSString* ff = e[@"file"]) {
          if (![ff isKindOfClass:[NSString class]] || [ff length] == 0 || [ff rangeOfString:@"/"].location != NSNotFound || [ff hasPrefix:@"."]) {
            report.error = "body override: " + relc + " has an invalid file field (a plain file name inside the package is required)"; return false; }
          file = dir + "/" + std::string([ff UTF8String]);
        }
        NSData* lb = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:file.c_str()]];
        if (!lb) { report.error = "body override: cannot read " + file; return false; }
        unsigned char ld[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(lb.bytes, (CC_LONG)lb.length, ld);
        char lhx[65]; for (int i = 0; i < 32; ++i) snprintf(lhx + 2 * i, 3, "%02x", ld[i]);
        if (std::string(lhx) != std::string([want UTF8String])) {
          report.error = "body override: " + relc + " has " + std::string(lhx).substr(0, 16) +
                         ", manifest declares " + std::string([want UTF8String]).substr(0, 16); return false; }
        NSError* lerr = nil;
        dispatch_data_t dd = dispatch_data_create(lb.bytes, lb.length, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib = [device newLibraryWithData:dd error:&lerr];
        if (!lib) { report.error = "body override: library load failed for " + relc; return false; }
        for (id v in fns) {
          if (![v isKindOfClass:[NSString class]]) { report.error = "body override: bad function name for " + relc; return false; }
          const std::string fn = [v UTF8String];
          if (![lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]]) {
            report.error = "body override: " + relc + " does not define " + fn; return false; }
          ctx->body.override.functions[relc].insert(fn);
        }
        ctx->body.override.libraries[relc] = lib;
        ctx->body.override.sha256[relc] = lhx;
      }
#ifdef NR_EXPERIMENTAL_NATIVE_RANGE
      // CANDIDATE ONLY: optional native_ranges. Every declaration is verified here, before any frame: a known kind, a
      // consistent launch range, both libraries as plain files inside the package with matching sha256, and every
      // function the kind encodes. Anything else refuses preparation (no silent fallback to the PTX dispatch).
      if (id nr = m[@"native_ranges"]) {
        if (![nr isKindOfClass:[NSArray class]]) { report.error = "body override: native_ranges must be an array"; return false; }
        // __strong spelled out: ARC infers __autoreleasing for an unqualified id& parameter, which cannot bind the struct's strong members.
        auto load_plain = [&](NSDictionary* e, const char* file_key, const char* sha_key, __strong id<MTLLibrary>& out, std::string& sha_out) -> bool {
          NSString* ff = e[[NSString stringWithUTF8String:file_key]]; NSString* want = e[[NSString stringWithUTF8String:sha_key]];
          if (![ff isKindOfClass:[NSString class]] || [ff length] == 0 || [ff rangeOfString:@"/"].location != NSNotFound || [ff hasPrefix:@"."] || ![want isKindOfClass:[NSString class]]) {
            report.error = std::string("body override: native range needs a plain ") + file_key + " and " + sha_key; return false; }
          const std::string path = dir + "/" + std::string([ff UTF8String]);
          NSData* lb = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:path.c_str()]];
          if (!lb) { report.error = "body override: cannot read " + path; return false; }
          unsigned char ld[CC_SHA256_DIGEST_LENGTH]; CC_SHA256(lb.bytes, (CC_LONG)lb.length, ld);
          char lhx[65]; for (int i = 0; i < 32; ++i) snprintf(lhx + 2 * i, 3, "%02x", ld[i]);
          if (std::string(lhx) != std::string([want UTF8String])) { report.error = "body override: native range library " + path + " hash mismatch"; return false; }
          dispatch_data_t dd = dispatch_data_create(lb.bytes, lb.length, dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
          NSError* lerr = nil; out = [device newLibraryWithData:dd error:&lerr];
          if (!out) { report.error = "body override: native range library load failed for " + path; return false; }
          sha_out = lhx; return true; };
        for (id v in (NSArray*)nr) {
          if (![v isKindOfClass:[NSDictionary class]]) { report.error = "body override: bad native range entry"; return false; }
          NSDictionary* e = v; NrBodyOverride::NativeRange r;
          NSString* kind = e[@"kind"]; NSNumber* first = e[@"first_launch"]; NSNumber* last = e[@"last_launch"];
          if (![kind isKindOfClass:[NSString class]] || ![first isKindOfClass:[NSNumber class]] || ![last isKindOfClass:[NSNumber class]]) {
            report.error = "body override: native range needs kind, first_launch and last_launch"; return false; }
          r.kind = [kind UTF8String]; r.first = first.unsignedIntValue; r.last = last.unsignedIntValue;
          // v2 = v1 + the recovered launch-133 upsample_tilesync pre-block stage (build/nr-accept-v1/experiments/NR-RANGE-133-140-v1.json);
          // its prepack kind 5 is pinned by the library hash, its two new kernels are checked by name below.
          // wide128-chained-exact-dense-v2 = the same v2 path for the 4h C128 tier (experiments/NR-RANGE-133-146-v1.json)
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
          // DECODER-EXACT-MLP-FUSION-v1 (experiments/DECODER-EXACT-MLP-FUSION-v1.json): a v3 kind is its v2 kind plus the fused exact MLP head. It is
          // normalized to the v2 kind here, so every tier formula below and in the body stays the verified one, and the range is marked fused. A
          // runtime without this flag refuses the v3 names as unknown kinds, so a fused package never runs unfused.
          if (r.kind == "wide256-chained-exact-dense-v3" || r.kind == "wide128-chained-exact-dense-v3" || r.kind == "wide64-chained-exact-dense-v3") {
            r.kind = r.kind.substr(0, r.kind.size() - 2) + "v2"; r.fused = true; }
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
          // DECODER-EXACT-MLP-FUSION-v2 (experiments/DECODER-EXACT-MLP-FUSION-v2.json): a v4 kind is its v3 kind plus the second fusion set (mid+quant-split,
          // values+quant-split, project+pack, normalize+split, probabilities+perm-split). It is normalized to the v2 kind and marked fused and fused2; a
          // runtime without this flag refuses the v4 names as unknown kinds.
          if (r.kind == "wide256-chained-exact-dense-v4" || r.kind == "wide128-chained-exact-dense-v4" || r.kind == "wide64-chained-exact-dense-v4") {
            r.kind = r.kind.substr(0, r.kind.size() - 2) + "v2"; r.fused = true; r.fused2 = true; }
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
          // C32-NATIVE-FUSED-RANGE-v1 (experiments/C32-NATIVE-FUSED-RANGE-v1.json): a wide32 v3 kind is the wide32 v2 kind plus the six fused C32 pairs of
          // EXACT-C32-FUSION-GATE-v1. Normalized to the v2 kind here and marked fused32; a runtime without this flag refuses the v3 name as an unknown kind.
          if (r.kind == "wide32-chained-exact-dense-v3") { r.kind = "wide32-chained-exact-dense-v2"; r.fused32 = true; }
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_POST
          // POST-BLOCK-NATIVE-EXACT-v1 (experiments/POST-BLOCK-NATIVE-EXACT-v1.json): the native post block for launch 155 (reset frames). The library carries the
          // C32 exact dense kernels with the gated fused pairs plus the post entry, fp16 head linear and head kernels; the engine library is declared for the
          // schema but no engine kernel is encoded by this kind. A runtime without this flag refuses the kind as unknown, so the package never runs elsewhere.
          if (r.kind == "post-block-native-exact-v1") {
            if (r.first != 155u || r.last != 155u) { report.error = "body override: post-block-native-exact-v1 must declare first_launch = last_launch = 155"; return false; }
            for (const auto& other : ctx->body.override.native_ranges)
              if (!(r.last < other.first || other.last < r.first)) { report.error = "body override: overlapping native ranges"; return false; }
            if (!load_plain(e, "library", "library_sha256", r.library, r.library_sha256)) return false;
            if (!load_plain(e, "engine_library", "engine_library_sha256", r.engine, r.engine_sha256)) return false;
            static const char* const post_fns[] = {"nrp_entry", "nrp_head_linear", "nrp_head", "nrd32_prepack_exact", "nrd_tile_class", "nrd32_seed", "nrd_split_bytes", "nrd_linear_exact_ws4",
                                                   "nrd32_scores_exact_ws4", "nrd32_perm_values", "nrd32_expand_activate_split_exact_ws4", "nrd_linear_quant_split_keep_exact_ws4",
                                                   "nrd32_normalize_split", "nrd_probabilities_perm_split", "nrd32_values_quant_split_exact_ws4"};
            for (const char* fn : post_fns) if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: native post library lacks ") + fn; return false; }
            r.post = true; ctx->body.override.native_ranges.push_back(r); continue;
          }
#endif
          if (r.kind != "wide256-chained-exact-dense-v1" && r.kind != "wide256-chained-exact-dense-v2" && r.kind != "wide128-chained-exact-dense-v2" && r.kind != "wide64-chained-exact-dense-v2" && r.kind != "wide32-chained-exact-dense-v2") { report.error = "body override: unknown native range kind " + r.kind; return false; }
          if (r.first > r.last || r.last - r.first > 16) { report.error = "body override: native range launch bounds"; return false; }
          for (const auto& other : ctx->body.override.native_ranges)
            if (!(r.last < other.first || other.last < r.first)) { report.error = "body override: overlapping native ranges"; return false; }
          if (!load_plain(e, "library", "library_sha256", r.library, r.library_sha256)) return false;
          if (!load_plain(e, "engine_library", "engine_library_sha256", r.engine, r.engine_sha256)) return false;
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE
          if (r.fused && ![r.library newFunctionWithName:@"nrd_expand_activate_split_exact_ws4"]) {
            report.error = "body override: fused native range library lacks nrd_expand_activate_split_exact_ws4"; return false; }
#endif
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE2
          if (r.fused2) {
            for (const char* fn : {"nrd_linear_quant_split_exact_ws4", "nrd_values_quant_split_exact_ws4", "nrd_linear_pack_exact_ws4", "nrd_normalize_split", "nrd_probabilities_perm_split"})
              if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: fused2 native range library lacks ") + fn; return false; }
          }
#endif
          static const char* const dense_fns[] = {"nrd_import_tensor", "nrd_prepack_exact", "nrd_tile_class", "nrd_split_bytes", "nrd_split_quant_perm", "nrd_seed",
                                                  "nrd_linear_exact_ws4", "nrd_scores_exact_ws4", "nrd_perm_probs", "nrd_perm_values", "nrd_values_exact_ws4"};
          static const char* const engine_fns[] = {"nr_wide_rewindow", "nr_wide_activate", "nr_wide_normalize", "nr_swin32_probabilities", "nr_wide_pack", "nr_encoder_native_export"};
          static const char* const upsample_fns[] = {"nrd_up_import", "nrd_up_combine"};
          for (const char* fn : dense_fns) if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: native range library lacks ") + fn; return false; }
          if (r.kind != "wide256-chained-exact-dense-v1")   // both v2 kinds (wide256, wide128) carry the upsample pre-stage
            for (const char* fn : upsample_fns) if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: native range library lacks ") + fn; return false; }
          static const char* const c32_fns[] = {"nrd32_prepack_exact", "nrd32_seed", "nrd32_scores_exact_ws4", "nrd32_perm_values", "nrd32_values_exact_ws4", "nrd32_up_combine_v"};
          static const char* const c32_engine_fns[] = {"nr_swin32_rewindow", "nr_swin32_activate", "nr_swin32_normalize", "nr_swin32_pack"};
          if (r.kind == "wide32-chained-exact-dense-v2") {   // the C32 walk tier (experiments/NR-RANGE-133-154-v1.json)
            for (const char* fn : c32_fns) if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: native range library lacks ") + fn; return false; }
            for (const char* fn : c32_engine_fns) if (![r.engine newFunctionWithName:@(fn)]) { report.error = std::string("body override: native range engine library lacks ") + fn; return false; }
#ifdef NR_EXPERIMENTAL_NATIVE_FUSE32
            if (r.fused32)
              for (const char* fn : {"nrd32_expand_activate_split_exact_ws4", "nrd_linear_quant_split_keep_exact_ws4", "nrd32_normalize_split", "nrd_probabilities_perm_split", "nrd32_values_quant_split_exact_ws4", "nrd_linear_pack_exact_ws4"})
                if (![r.library newFunctionWithName:@(fn)]) { report.error = std::string("body override: fused32 native range library lacks ") + fn; return false; }
#endif
          }
          for (const char* fn : engine_fns) if (![r.engine newFunctionWithName:@(fn)]) { report.error = std::string("body override: native range engine library lacks ") + fn; return false; }
          ctx->body.override.native_ranges.push_back(r);
        }
      }
#endif
      ctx->body.override.root = dir;
      ctx->body.override.active = true;
      report.arithmetic_label += " + EXPERIMENTAL body override v" + ctx->body.override.version +
                                 " manifest-sha256=" + ctx->body.override.manifest_sha256;
    }
  }
  // ASSERT that every stage NOT declared for override resolved to the PINNED library. Without this
  // the isolation claim rests on reading the manifest; with it, a stage that quietly picked up
  // rebuilt code fails preparation instead of showing up as an unexplained slowdown later.
  if (ctx->assets.qkv_override_active) {
    for (const auto& kv : ctx->assets.resolved_pipeline_source) {
      const std::string& key = kv.first, &src = kv.second;
      const size_t colon = key.rfind(':');
      if (colon == std::string::npos) continue;
      const std::string lib = key.substr(0, colon), fn = key.substr(colon + 1);
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
      // Dense-stage pipelines (ENCODER-DENSE-WS8) are their own, explicitly bound source kind and never a qkv-override
      // declaration: accepted only for a library the dense loader loaded, carrying that library's manifest hash, and only
      // for the five dense kernels. Every other source falls through to the unchanged checks below.
      if (src.rfind("override:encdense:", 0) == 0) {
        const auto dsha = ctx->assets.encdense_sha256.find(lib);
        const bool c32lib = lib.rfind("enc_dense32", 0) == 0;   // enc_dense32 (walk) and enc_dense32b0 (Block 0): the C32 hybrid kernels
        const bool dense_kernel = c32lib ? (fn == "nrd32w_prepack" || fn == "nrd32w_decode" || fn == "nrd32w_quant_perm" || fn == "nrd32w_linear_ws8")
                                         : (fn == "nrd_prepack" || fn == "nrd_decode" || fn == "nrd_quant_perm" || fn == "nrd_seed" || fn == "nrd_linear_ws8");
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
        // DENSE-MLP-FUSION-v2 (experiments/DENSE-MLP-FUSION-v2.json): a tier whose manifest declared "fused": ["expand+activate"] also resolves its
        // fused kernel from the same loaded library. Admitted only under that library's own fuse bit, so an unfused tier never resolves one.
        const uint32_t fuse_bit = lib == "enc_dense32b0.metallib" ? 0x20u : lib == "enc_dense32.metallib" ? 1u : lib == "enc_dense64.metallib" ? 2u :
                                  lib == "enc_dense128.metallib" ? 4u : lib == "enc_dense256.metallib" ? 8u : 0u;
        const bool fused_kernel = fuse_bit != 0u && (ctx->assets.encdense_fuse_mask & fuse_bit) != 0u &&
                                  fn == (c32lib ? "nrd32w_expand_activate_ws8" : "nrd_expand_activate_ws8");
        if (!ctx->assets.encdense_active || dsha == ctx->assets.encdense_sha256.end() || !(dense_kernel || fused_kernel) ||
            src != "override:encdense:" + lib + "@" + dsha->second) {
#else
        if (!ctx->assets.encdense_active || dsha == ctx->assets.encdense_sha256.end() || !dense_kernel ||
            src != "override:encdense:" + lib + "@" + dsha->second) {
#endif
          report.error = "encoder dense: " + key + " resolved to " + src + ", which is not a loaded dense-stage library and kernel; refusing";
          return false;
        }
        continue;
      }
#endif
      auto byl = ctx->assets.qkv_override_by_library.find(lib);
      const bool declared = byl != ctx->assets.qkv_override_by_library.end() && byl->second.count(fn);
      if (!declared && src.rfind("override:", 0) == 0) {
        report.error = "qkv override: " + key + " resolved to " + src +
                       " but this candidate does not declare that function for that library; refusing";
        return false;
      }
      if (declared && src.rfind("pinned:", 0) == 0) {
        report.error = "qkv override: " + key + " is declared for override but resolved to " + src +
                       "; refusing rather than reporting a candidate that did not take effect";
        return false;
      }
    }
  }
  // Refuse an unsupported candidate configuration during PREPARATION, so it never reaches encode. The
  // encode-time admission check in block_wide stays as the backstop, but by then entry layout,
  // rewindowing and earlier blocks may already have encoded work into the caller's buffer.
  if (ctx->assets.qkv_override_active && nr_exp::wide_threadgroup_width() != 64) {
    report.error = "qkv override: the per-kernel nr_wide_qkv override is admitted only at the accepted "
                   "64-thread geometry; NR_EXP_WIDE_TG requests " +
                   std::to_string(nr_exp::wide_threadgroup_width()) +
                   ". Refusing during preparation, before anything is encoded.";
    return false;
  }
  if (approximate && !ctx->assets.unpinned_libraries) {
    report.error = "ApproximateNative was requested but every library matched the exact pins, so the approximate set is not installed at " + encoder_root;
    return false;   // fail closed rather than silently run the exact kernels under an approximate label
  }
  if (!nr_exp::allocate_block0(device, ctx->assets, ctx->block0, error)) { report.error = error; return false; }
  if (!nr_exp::allocate_encoder(device, ctx->assets, ctx->block0, ctx->encoder, error)) { report.error = error; return false; }
  {   // body descriptor and context: every library, pipeline and resident buffer is built here, never in encode()
    NSData* d = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:descriptor_path.c_str()]];
    if (!d) { report.error = "cannot read the body descriptor at " + descriptor_path; return false; }
    NSError* jerr = nil;
    id parsed = [NSJSONSerialization JSONObjectWithData:d options:NSJSONReadingMutableContainers | NSJSONReadingMutableLeaves error:&jerr];
    // The bundle is relocatable, so its descriptor stores paths relative to <bundle>/body. The body runtime
    // opens them with the process working directory, so make them absolute here, once, at prepare time.
    if ([parsed isKindOfClass:[NSDictionary class]]) {
      NSString* base = [NSString stringWithUTF8String:(body_dir + "/").c_str()];
      // Plain recursive function rather than a self-capturing block: no retain cycle, and easier to read.
      struct Fix {
        NSString* base;
        void walk(id node) const {
          if ([node isKindOfClass:[NSMutableDictionary class]]) {
            NSMutableDictionary* m = (NSMutableDictionary*)node;
            for (NSString* k in [m allKeys]) m[k] = one(m[k]);
          } else if ([node isKindOfClass:[NSMutableArray class]]) {
            NSMutableArray* a = (NSMutableArray*)node;
            for (NSUInteger i = 0; i < a.count; ++i) a[i] = one(a[i]);
          }
        }
        id one(id v) const {
          if ([v isKindOfClass:[NSString class]]) {
            NSString* str = (NSString*)v;
            // Anything the packager relocated is rooted at "assets/", including DIRECTORY values such as the two
          // table directories. Keying on file extensions missed those and left them relative to the process
          // working directory, which is what made the self-contained bundle fail to open its own tables.
          if ([str hasPrefix:@"assets/"] ||
              (![str hasPrefix:@"/"] && ([str hasSuffix:@".metallib"] || [str hasSuffix:@".bin"] || [str hasSuffix:@".e4m3"] || [str hasSuffix:@".f16"])))
              return [base stringByAppendingString:str];
            return v;
          }
          walk(v); return v;
        }
      };
      Fix{base}.walk(parsed);
    }
    ctx->descriptor = parsed;
    if (![ctx->descriptor isKindOfClass:[NSDictionary class]]) {
      report.error = "body descriptor is not a JSON object: " + std::string(jerr ? jerr.localizedDescription.UTF8String : "unknown");
      return false;
    }
    ctx->body.dev = device;
    ctx->body.queue = [device newCommandQueue];   // never used to submit in this lane; the caller owns the buffer
    if (!ctx->body.queue) { report.error = "command queue creation failed"; return false; }
    // Optional GPU timestamp profiling; inert unless NR_EXP_GPU_TIMESTAMPS=1.
    std::string perr;
    if (!nr_exp::gpu_profiler().prepare(device, 2048, perr)) { report.error = "gpu profiler: " + perr; return false; }
    // Our own output surface, created here so output_texture() is callable the moment encode() returns and the
    // bridge can append its export into the SAME command buffer, before it commits.
    MTLTextureDescriptor* od = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                                                  width:options.width height:options.height mipmapped:NO];
    od.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; od.storageMode = MTLStorageModeShared;
    ctx->output = [device newTextureWithDescriptor:od];
    if (!ctx->output) { report.error = "output surface creation failed"; return false; }
    // Our own history texture. It is ours, not the game's, and it is what makes a temporal frame real.
    if (!nr_temporal_state_create(ctx->temporal, device, options.width, options.height, 0xa5)) {
      report.error = "temporal history creation failed"; return false;
    }
    ctx->frames_since_reset = 0;
    // Pre-create EVERY library and pipeline the body can reach, including the temporal (lib_temporal) set that
    // only a non-reset frame touches. Without this the first temporal frame paid 2.9 s of pipeline creation
    // inside encode(), on the render thread - the same defect as the encoder's wide tiers had.
    {
      NSDictionary* dec = ctx->descriptor[@"decoder"];
      NSArray* launches = dec ? dec[@"launches"] : nil;
      NSMutableSet* pairs = [NSMutableSet set];
      for (NSDictionary* L in launches) {
        NSString* kernel = L[@"kernel"];
        if (!kernel) continue;
        for (NSString* key : @[@"lib", @"lib_temporal"]) {
          NSString* lib = L[key];
          if ([lib isKindOfClass:[NSString class]] && lib.length) [pairs addObject:[lib stringByAppendingFormat:@"\n%@", kernel]];
        }
      }
      for (NSString* extra : @[@"split_lib", @"vit_lib", @"split_tp_lib", @"vit_tp_lib", @"resident_lib"]) {
        NSString* lib = ctx->descriptor[extra];
        if ([lib isKindOfClass:[NSString class]] && lib.length) [pairs addObject:[lib stringByAppendingString:@"\n*"]];
      }
      NSError* lerr = nil;
      size_t made_libs = 0, made_pipes = 0;
      for (NSString* pair in pairs) {
        NSArray* parts = [pair componentsSeparatedByString:@"\n"];
        NSString* lib_path = (NSString*)parts[0];
        const std::string lp = lib_path.UTF8String;
        auto it = ctx->body.libs.find(lp);
        id<MTLLibrary> lib = nil;
        if (it != ctx->body.libs.end()) lib = it->second;
        else {
          lib = [device newLibraryWithURL:[NSURL fileURLWithPath:lib_path] error:&lerr];
          if (!lib) { report.error = "body library " + lp + ": " + std::string(lerr ? lerr.localizedDescription.UTF8String : "load failed"); return false; }
          ctx->body.libs[lp] = lib; ++made_libs;
        }
        NSString* fn = (NSString*)parts[1];
        if ([fn isEqualToString:@"*"]) continue;             // whole-library preload only; its entry points vary
        const std::string pkey = std::to_string(reinterpret_cast<uintptr_t>((__bridge void*)lib)) + "|" + fn.UTF8String;
        if (ctx->body.pipes.count(pkey)) continue;
        id<MTLFunction> f = [lib newFunctionWithName:fn];
        if (!f) continue;                                     // a kernel this descriptor names but this library lacks
        id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:f error:&lerr];
        if (!ps) { report.error = "body pipeline " + std::string(fn.UTF8String) + ": " + std::string(lerr ? lerr.localizedDescription.UTF8String : "failed"); return false; }
        ctx->body.pipes[pkey] = ps; ++made_pipes;
      }
      report.model_identity += " libs:" + std::to_string(made_libs) + " pipes:" + std::to_string(made_pipes);
    }
  }
  ctx->prepared = true;
  ctx->history_valid = false;
  ctx->arithmetic = options.arithmetic;
  ctx->invalidate_reason = "prepared; no frame admitted yet";
  report.model_identity = "nr-pinned-parent-root:" + ctx->assets.parent_root;
  report.weights_identity = (ctx->assets.unpinned_libraries ? "pinned-weights-verified, kernel libraries substituted:" : "pinned-assets-verified:") + ctx->assets.root;
  if (ctx->assets.qkv_override_active)
    report.weights_identity += ", nr_wide_qkv substituted from manifest " + ctx->assets.qkv_override_manifest_sha256;
  // Emit the resolved wide-pipeline sources, AFTER weights_identity is assigned - appending before the
  // assignment simply lost them. The isolation is then readable from a receipt, not inferred.
  if (ctx->assets.qkv_override_active) {
    std::string res;
    for (const auto& kv : ctx->assets.resolved_pipeline_source) {
      if (kv.first.find("nr_wide_") == std::string::npos) continue;
      res += (res.empty() ? "" : " ") + kv.first + "=" + kv.second;
    }
    if (!res.empty()) report.weights_identity += " | resolved: " + res;
  }
  ctx->prepare_cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  report.prepare_cpu_ms = ctx->prepare_cpu_ms; report.prepare_cpu_time_valid = true;
  out = ctx;
  return true;
}

EncodeStatus encode(const ContextPtr& ctx, const FrameInputs& in,
                    id<MTLCommandBuffer> caller_command, TicketPtr& ticket, Report& report) {
  // Reset the opt-in C64 capture ONCE PER MODEL FRAME. It must not live in encode_encoder: the reset
  // worker calls that repeatedly per frame, one call per unit chunk, so a reset there would clear the
  // capture mid-frame and let a later segment overwrite it. (Codex, NR_C64_MATRIX_REVIEW.)
  if (ctx) { ctx->encoder.qkv_capture_bytes = 0; ctx->encoder.qkv_capture_done = false; }
  const auto encode_t0 = std::chrono::steady_clock::now();
  struct CpuTimer {   // encode() lands on the game's render thread; always report what it cost there
    const std::chrono::steady_clock::time_point t0; Report& r;
    ~CpuTimer() { r.encode_cpu_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); r.encode_cpu_time_valid = true; }
  };
  report = Report{};
  report.experimental = true;
  report.stream_id = in.stream_id; report.frame_id = in.frame_id;
  CpuTimer cpu_timer{encode_t0, report};
  ticket.reset();
  if (!ctx || !ctx->prepared) { report.error = "context not prepared"; return EncodeStatus::RejectedBeforeEncoding; }
  report.arithmetic_label = arithmetic_label(ctx->options.arithmetic);
  if (ctx->assets.qkv_override_active)
  {
    std::string fns;
    for (const auto& f : ctx->assets.qkv_override_functions) fns += (fns.empty() ? "" : ",") + f;
    report.arithmetic_label += " + EXPERIMENTAL per-kernel override of {" + fns + "}" +
                               " v" + ctx->assets.qkv_override_version +
                               " manifest-sha256=" + ctx->assets.qkv_override_manifest_sha256;
  }
  report.model_identity = "nr-pinned-parent-root:" + ctx->assets.parent_root;
  report.weights_identity = (ctx->assets.unpinned_libraries ? "pinned-weights-verified, kernel libraries substituted:" : "pinned-assets-verified:") + ctx->assets.root;
  std::lock_guard<std::mutex> guard(ctx->lock);
  if (!caller_command) { report.error = "no caller command buffer"; return EncodeStatus::RejectedBeforeEncoding; }
  if (caller_command.device != ctx->device) { report.error = "caller command buffer belongs to another device"; return EncodeStatus::RejectedBeforeEncoding; }
  if (caller_command.status != MTLCommandBufferStatusNotEnqueued) { report.error = "caller command buffer is already enqueued or committed"; return EncodeStatus::RejectedBeforeEncoding; }
  if (ctx->frame_in_flight) { report.error = "busy: one admitted frame per context"; return EncodeStatus::RejectedBeforeEncoding; }
  if (!in.reset && !ctx->history_valid) { report.error = "temporal frame requested but NR history is invalid (" + ctx->invalidate_reason + "); a reset frame is required"; return EncodeStatus::RejectedBeforeEncoding; }
  // Motion convention is checked from the TYPE, never assumed. A reset frame may omit motion entirely.
  if (!in.reset) {
    if (!in.motion) { report.error = "temporal frame requires a motion texture"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.motion_units != MotionUnits::PixelsFullRes) { report.error = "this runtime accepts only MotionUnits::PixelsFullRes; convert in the bridge"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.motion_direction != MotionDirection::CurrentToPrevious) { report.error = "this runtime accepts only MotionDirection::CurrentToPrevious"; return EncodeStatus::RejectedBeforeEncoding; }
    if (!in.motion_y_down) { report.error = "this runtime accepts only y-down motion"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.motion_scale_x != 1.0f || in.motion_scale_y != 1.0f) { report.error = "this runtime accepts only motion scales of 1; resample in the bridge"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.motion.width != ctx->options.width || in.motion.height != ctx->options.height) { report.error = "motion texture is not full resolution"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.motion.pixelFormat != MTLPixelFormatRG16Float) { report.error = "motion texture is not RG16Float"; return EncodeStatus::RejectedBeforeEncoding; }
    // Stream and frame continuity are ENFORCED, not merely recorded: a temporal frame must continue the same
    // stream and must advance the frame id by exactly one. A gap means a missed frame, and a missed frame
    // invalidates history rather than being silently accepted.
    if (!ctx->have_last_frame) { report.error = "temporal frame with no completed predecessor in this context"; return EncodeStatus::RejectedBeforeEncoding; }
    if (in.stream_id != ctx->last_stream_id) {
      report.error = "temporal frame changes stream (" + std::to_string(ctx->last_stream_id) + " -> " + std::to_string(in.stream_id) + "); a stream change requires a reset";
      return EncodeStatus::RejectedBeforeEncoding;
    }
    if (in.frame_id != ctx->last_frame_id + 1) {
      report.error = "temporal frame id " + std::to_string(in.frame_id) + " does not follow " + std::to_string(ctx->last_frame_id) + "; a gap is a missed frame and requires a reset";
      return EncodeStatus::RejectedBeforeEncoding;
    }
  }

  // Reject before encoding while any stage lacks an encode-only implementation. A partial chain must never
  // reach the caller's buffer and must never be reported as NR.
  if (const char* missing = first_missing_stage()) {
    report.error = std::string("declined: no encode-only implementation yet for ") + missing +
                   ". This runtime never encodes a partial chain or a stub image.";
    return EncodeStatus::RejectedBeforeEncoding;
  }
  if (!in.color) { report.error = "no colour texture"; return EncodeStatus::RejectedBeforeEncoding; }
  if (in.color.width != ctx->options.width || in.color.height != ctx->options.height) {
    report.error = "colour texture is not " + std::to_string(ctx->options.width) + "x" + std::to_string(ctx->options.height);
    return EncodeStatus::RejectedBeforeEncoding;
  }
  if (in.color.pixelFormat != MTLPixelFormatRGBA16Float) { report.error = "colour texture is not RGBA16Float"; return EncodeStatus::RejectedBeforeEncoding; }

  // ---- everything below encodes; from here a failure is EncodedDiscardOnly, never a silent rollback ----
  auto ticket_out = std::make_shared<Ticket>();
  ticket_out->context = ctx; ticket_out->command = caller_command;
  ticket_out->stream_id = in.stream_id; ticket_out->frame_id = in.frame_id;
  ticket_out->continuity_generation = ctx->continuity_generation;

  std::string error;
  // One scope holder per encode INVOCATION, never global. It owns every encoder this runtime opens - block 0,
  // the encoder blocks and the body - and closes any that are still open however this function leaves.
  nr_exp::ScopeHolder scopes;
  struct HolderInstall {
    HolderInstall(nr_exp::ScopeHolder* h) { nr_exp::set_current_scope_holder(h); }
    ~HolderInstall() { nr_exp::set_current_scope_holder(nullptr); }
  } holder_install{&scopes};
  nr_exp::gpu_profiler().begin_frame();
  // Block 0's shape carries the RESET-LOCAL noise step, not the caller's global frame id. A reset frame is
  // step 0; a temporal frame is the number of frames since the last reset, which the body's temporal state owns.
  const uint32_t noise_step = in.reset ? 0u : ctx->temporal.frames_since_reset;
  nr_exp::Block0Encoded b0{};
  // Anything that throws after work is on the caller's buffer must still produce a RETAINED ticket, because
  // the encoded commands and every buffer they reference have to stay alive until that buffer completes or
  // is dropped. The ticket exists before any encoding, so no path can fail without one.
  try {
  if (!nr_exp::encode_block0(caller_command, ctx->assets, ctx->block0, in.color, noise_step, b0, error,
                             in.reset ? nil : ctx->temporal.history, in.reset ? nil : in.motion)) {
    // encode_block0 validates before it encodes, but it can still fail PART WAY THROUGH (an encoder the driver
    // refuses to hand out). Anything already encoded cannot be rolled back, so that case must yield a ticket.
    if (b0.compute_encoders > 0) {
      ctx->frame_in_flight = true; ticket_out->status = EncodeStatus::EncodedDiscardOnly; ticket = ticket_out;
      ctx->history_valid = false; ctx->invalidate_reason = "block 0 failed after encoding " + std::to_string(b0.compute_encoders) + " encoder(s)";
      report.error = "block 0 failed after encoding work that cannot be rolled back: " + error + kDiscardOnlyGuidance;
      return EncodeStatus::EncodedDiscardOnly;
    }
    report.error = "block 0 rejected before encoding: " + error;
    return EncodeStatus::RejectedBeforeEncoding;
  }
  // Test-only failure injection, off unless NR_EXP_FAIL_AFTER_BLOCK0=1. It throws at exactly the point the
  // review cares about: after NR scratch work is already on the caller's buffer. Never set in normal use.
  { static const bool inject = [] { const char* v = std::getenv("NR_EXP_FAIL_AFTER_BLOCK0"); return v && v[0] == '1'; }();
    if (inject) throw std::runtime_error("injected failure after block 0 (NR_EXP_FAIL_AFTER_BLOCK0)"); }
  nr_exp::EncoderEncoded e22{};
  if (!nr_exp::encode_encoder(caller_command, ctx->assets, ctx->block0, ctx->encoder, e22, error)) {
    ctx->frame_in_flight = true; ticket_out->status = EncodeStatus::EncodedDiscardOnly; ticket = ticket_out;
    ctx->history_valid = false; ctx->invalidate_reason = "encoder blocks failed after block 0 was encoded";
    report.error = "encoder blocks failed after block 0 was encoded, so the work cannot be rolled back: " + error + kDiscardOnlyGuidance;
    return EncodeStatus::EncodedDiscardOnly;
  }
  nr_exp::set_last_chain_bytes({e22.bound_bytes, e22.bound_bytes_wide, e22.bound_bytes_coverage});

  // ---- body: the encoder's live outputs are handed over as regions; nothing is read from disk ----
  NrBodyFrameInputs bin;
  auto region = [](const nr_exp::Buf& b) {
    NrBodyRegion r; r.buffer = b.metal; r.offset = nr_exp::Buf::kGuard; r.bytes = b.bytes; r.producer = "experimental-encoder"; return r;
  };
  bin.entry = region(ctx->encoder.entry);
  for (size_t i = 0; i < 4 && i < ctx->encoder.skips.size(); ++i) bin.skips[i] = region(ctx->encoder.skips[i]);
  bin.block0_skip = region(ctx->block0.skip);
  bin.input_color = in.color;
  bin.output_surface = ctx->output;    // OUR surface, already visible to the caller through output_texture()
  bin.production = true;               // final-only body checks; the research validators are compiled out as well
  // DIAGNOSTIC ONLY, opt-in, never on in a timing run. NR_EXP_BODY_DIAG_DIR turns on the body's EXISTING
  // per-launch pool digests and coverage validators (production == false) and writes chain-run.json there.
  // This is how a changed decoder producer is checked IMMEDIATELY AFTER ITS OWN LAUNCH, before any later
  // launch reads or overwrites the pool - pool_final and the final surface are both downstream of every
  // remaining decoder launch and cannot separate "this producer is unchanged" from "the difference did
  // not survive". It costs digest dispatches per launch, so any cost number taken with this set is void.
  static const char* const body_diag_dir = std::getenv("NR_EXP_BODY_DIAG_DIR");
  if (body_diag_dir && *body_diag_dir) bin.production = false;
  bin.reset = in.reset;
  bin.motion = in.motion;
  bin.temporal = &ctx->temporal;       // real history: 155 writes the caller output, 156 copies it into our history
  // Test-only: make the BODY reject mid-encode so it takes its internal catch and returns 2 with encoders
  // possibly still open. Off unless NR_EXP_FAIL_BODY_RC2=1.
  { static const bool inject = [] { const char* v = std::getenv("NR_EXP_FAIL_BODY_RC2"); return v && v[0] == '1'; }();
    if (inject) bin.entry.bytes = 1;   // the body validates this extent and throws inside its own encode
  }
  ctx->body.external_command = caller_command;
  const int rc = nr_body_encode(ctx->body, ctx->descriptor, bin, (body_diag_dir && *body_diag_dir) ? body_diag_dir : nullptr, ticket_out->pending);
  ctx->body.external_command = nil;
  // The body catches internally and returns 2, so an encoder can still be open on a nonzero return. Close
  // whatever is open before doing anything else; the caller's buffer must stay committable.
  if (rc != 0) scopes.close_all();
  // An ADOPTION fault - an encoder ended that this holder never adopted, or ended twice - means an encoder
  // was not handled the way the contract requires. It is not a Metal failure and rc may still be 0, so it
  // has to be checked separately or it stays invisible exactly as it did before. The buffer is left
  // committable (the unadopted encoder was ended, and close_all ran), and the frame is discard-only.
  const bool adoption_fault = scopes.adoption_failed();
  if (adoption_fault) {
    scopes.close_all();                    // closes only what THIS holder adopted; that is the whole point
    ctx->frame_in_flight = true; ticket_out->status = EncodeStatus::EncodedDiscardOnly; ticket = ticket_out;
    ctx->history_valid = false; ++ctx->continuity_generation;
    ctx->invalidate_reason = "encoder adoption fault during encode";
    const std::string counts = std::to_string(scopes.unadopted_ends()) + " end(s) of an encoder no holder "
        "adopted, " + std::to_string(scopes.foreign_ends()) + " end(s) of an encoder owned by ANOTHER holder, " +
        std::to_string(scopes.double_ends()) + " double end(s)";
    // NO non-live end keeps the normal guidance. A "double end" only means we saw no live scope and a
    // closed token of our own; because Metal reactivates encoder objects, that is indistinguishable from a
    // new encoding period whose adoption was missed, and the encoder may be open right now. This runtime
    // refuses to end an encoder it does not currently own, so it cannot assert the buffer is committable.
    report.error = "encoder adoption fault: " + counts + ". An encoder creation site is missing its scope "
                   "adoption, or an encoder crossed holders." +
                   (scopes.encoder_state_indeterminate() ? kIndeterminateEncoderGuidance : kDiscardOnlyGuidance);
    return EncodeStatus::EncodedDiscardOnly;
  }
  if (rc != 0) {
    ctx->frame_in_flight = true; ticket_out->status = EncodeStatus::EncodedDiscardOnly; ticket = ticket_out;
    ctx->history_valid = false; ++ctx->continuity_generation;
    ctx->invalidate_reason = "body encode failed after the encoder was encoded";
    report.error = "body encode failed after the encoder was already encoded: " + ticket_out->pending.res.error + kDiscardOnlyGuidance;
    return EncodeStatus::EncodedDiscardOnly;
  }
  nr_body_commit(ticket_out->pending);   // caller-command mode: advances the state machine, submits nothing
  ticket_out->dispatches = b0.dispatches + e22.dispatches;
  ticket_out->status = EncodeStatus::Encoded;
  ticket_out->output = ctx->output;      // callable NOW, before commit, so the bridge can export in this buffer
  ctx->frame_in_flight = true;
  ctx->admitted_stream = in.stream_id; ctx->admitted_frame = in.frame_id;
  if (in.reset) { ctx->last_stream_id = in.stream_id; ctx->last_frame_id = in.frame_id; }
  ticket = ticket_out;
  report.model_encoded = true;
  report.model_dispatches = ticket_out->dispatches;
  return EncodeStatus::Encoded;
  } catch (...) {
    // Both C++ and, under -fobjc-exceptions with ARC, Objective-C unwinding reach here. The message is derived
    // where possible, but the handling never depends on the exception type.
    std::string what = "unknown exception";
    try { throw; }
    catch (const std::exception& e) { what = e.what(); }
    catch (id ex) { what = std::string("Objective-C exception: ") + [[ex description] UTF8String]; }
    catch (...) { }
    // EncodedDiscardOnly means DISCARD THE NR OUTPUT - never the caller's command buffer. That buffer carries
    // the game's own rendering and the original MetalFX upscale; the bridge cannot abandon it and the game will
    // commit it regardless. So: close every encoder this runtime opened, leave the buffer in a committable
    // state, hand back a RETAINED ticket so nothing is released early, and invalidate NR history. The caller
    // commits and finalizes exactly as it would on success; it simply must not publish our output.
    scopes.close_all();
    const bool anything_encoded = scopes.started() > 0 || b0.compute_encoders > 0;   // counted at creation, not at close
    ctx->history_valid = false;
    ++ctx->continuity_generation;
    ctx->invalidate_reason = "exception during encode: " + what;
    if (!anything_encoded) { report.error = "rejected before encoding: " + what; return EncodeStatus::RejectedBeforeEncoding; }
    ctx->frame_in_flight = true;
    ticket_out->status = EncodeStatus::EncodedDiscardOnly;
    ticket = ticket_out;
    report.error = std::string("exception after ") + std::to_string(b0.compute_encoders) +
                   " encoder(s) were already on the caller buffer: " + what + kDiscardOnlyGuidance;
    return EncodeStatus::EncodedDiscardOnly;
  }
}

id<MTLTexture> output_texture(const TicketPtr& t) {
  return (t && t->status == EncodeStatus::Encoded) ? t->output : nil;
}

void complete(const TicketPtr& t, id<MTLCommandBuffer> completed, Report& report) {
  report = Report{};
  report.experimental = true;
  if (!t) { report.error = "no ticket"; return; }
  report.stream_id = t->stream_id; report.frame_id = t->frame_id;
  auto ctx = t->context.lock();
  if (!ctx) { report.error = "context destroyed before completion"; return; }
  std::lock_guard<std::mutex> guard(ctx->lock);
  if (t->finalized) { report.error = "ticket already finalized"; return; }

  // ---- VALIDATE FIRST, FINALIZE SECOND ----
  // Codex's callback audit (build/nr-callback-audit-v1) showed the old order consuming the ticket: a call with
  // the wrong buffer, or a premature call on the genuine but unsubmitted buffer, was rejected but had ALREADY
  // set finalized and cleared frame_in_flight, so the genuine completion afterwards failed with "ticket already
  // finalized" even though its GPU work had succeeded. A rejected misuse must change nothing: no finalization,
  // no release of the in-flight reservation, and no touching of history. The pending frame stays pending and
  // its resources stay retained, because the real command may still be running.
  if (completed != t->command) {
    report.error = "completion callback for a different command buffer; the pending frame is untouched and still awaits its own completion";
    return;                                            // NOT finalized, NOT released
  }
  if (completed.status != MTLCommandBufferStatusCompleted && completed.status != MTLCommandBufferStatusError) {
    // EVERY nonterminal completion is rejected, discard-only included. A Scheduled or Committed buffer may
    // still be reading our scratch, so releasing anything here would free memory the GPU is using. A
    // discard-only frame is still committed and completed by the caller like any other; only its OUTPUT is
    // discarded, so it reaches a terminal state exactly the same way.
    report.error = "completion called before the command buffer reached a terminal state (status " +
                   std::to_string(int(completed.status)) + "); the pending frame is untouched and still owns its resources";
    return;                                            // NOT finalized, NOT released
  }

  // ---- from here the call is genuine: finalize exactly once and release the reservation ----
  t->finalized = true;
  ctx->frame_in_flight = false;
  // A ticket may only advance history if no invalidation happened while it was in flight. The generation is
  // captured at encode; invalidate() bumps it, so a late completion can never undo a newer invalidation.
  const bool continuity_intact = (t->continuity_generation == ctx->continuity_generation);

  if (completed.status == MTLCommandBufferStatusError) {
    ctx->history_valid = false;
    ctx->invalidate_reason = "command buffer failed: " + std::string(completed.error ? completed.error.localizedDescription.UTF8String : "unknown");
    report.error = ctx->invalidate_reason;
    if (t->status == EncodeStatus::Encoded) { NrBodyResult br; nr_body_complete(t->pending, br); }   // still finalize the body's own state
    return;
  }
  if (t->status != EncodeStatus::Encoded) {
    ctx->history_valid = false; ctx->invalidate_reason = "frame was encoded discard-only";
    report.error = ctx->invalidate_reason;
    NrBodyResult br; nr_body_complete(t->pending, br);   // the body still owns encoded work; let it finish its bookkeeping
    return;
  }
  NrBodyResult br;
  const int rc = nr_body_complete(t->pending, br);
  if (rc != 0) {
    ctx->history_valid = false;
    ctx->invalidate_reason = "body completion returned " + std::to_string(rc) + ": " + br.error;
    report.error = ctx->invalidate_reason; return;
  }
  report.model_encoded = true;
  report.model_dispatches = t->dispatches;
  if (nr_exp::boundary_hashes_requested()) {
    // Post-completion, output-only: the buffer is terminal and the body has finished its bookkeeping.
    nr_exp::BoundaryHashes bh;
    bh.captured = true;
    // The receipt's six regions are six separate buffers, each hashed from its 256-byte guard offset:
    // block 0's full-resolution skip, the four native tier exports (blocks 4, 8, 14, 22 in body order,
    // C256..C32) and the block22-middle entry export.
    if (ctx->block0.skip.metal) bh.block0_skip = nr_exp::diag_sha(ctx->block0.skip.data(), ctx->block0.skip.bytes);
    if (ctx->encoder.entry.metal) bh.entry = nr_exp::diag_sha(ctx->encoder.entry.data(), ctx->encoder.entry.bytes);
    std::string* const tier[4] = {&bh.skip_c256, &bh.skip_c128, &bh.skip_c64, &bh.skip_c32};
    for (size_t i = 0; i < 4 && i < ctx->encoder.skips.size(); ++i)
      if (ctx->encoder.skips[i].metal) *tier[i] = nr_exp::diag_sha(ctx->encoder.skips[i].data(), ctx->encoder.skips[i].bytes);
    if (ctx->block0.encoder.metal) bh.encoder_buffer = nr_exp::diag_sha(ctx->block0.encoder.data(), ctx->block0.encoder.bytes);
    bh.middle_view = br.middle_view_sha256; bh.decoder_pre_image = br.decoder_pre_image_sha256;
    bh.pool_final = br.pool_final_sha256;   bh.final_surface = br.final_surface_sha256;
    bh.decoder_pre_image_source = br.decoder_pre_image_source;
    nr_exp::set_last_boundary_hashes(bh);
  }
  if (!continuity_intact) {
    // The frame executed correctly, but continuity was invalidated while it was in flight (a missed frame, a
    // resize, a mode change, a fallback). Its history must NOT be adopted.
    ctx->history_valid = false;
    report.history_advanced = false;
    report.error = "frame completed but continuity was invalidated while it was in flight (" + ctx->invalidate_reason + "); history not adopted";
    return;
  }
  // History advances only when the body actually advanced its temporal state - that is established temporal
  // execution (155 wrote the output, 156 copied it into our history, and the two matched), not a boolean of ours.
  ctx->history_valid = br.temporal_state_advanced;
  report.history_advanced = br.temporal_state_advanced;
  if (br.temporal_state_advanced) {
    ctx->frames_since_reset = ctx->temporal.frames_since_reset;
    ctx->last_stream_id = t->stream_id; ctx->last_frame_id = t->frame_id; ctx->have_last_frame = true;
  } else {
    ctx->invalidate_reason = br.damage.empty() ? "body did not advance the temporal state" : br.damage;
  }
}

void invalidate(const ContextPtr& ctx, const char* reason) {
  if (!ctx) return;
  std::lock_guard<std::mutex> guard(ctx->lock);
  ctx->history_valid = false;                       // never touches in-flight GPU resources
  ctx->have_last_frame = false;
  ++ctx->continuity_generation;                     // an in-flight ticket cannot undo this when it completes
  ctx->invalidate_reason = reason ? reason : "unspecified";
}
} // namespace netal_nr_experimental
