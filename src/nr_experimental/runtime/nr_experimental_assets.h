#pragma once
// Experimental NR runtime: pinned asset loading and a persistent pipeline cache.
// Owner: Opus (src/nr_experimental/runtime/). Experimental lane only - never an exact-parity claim.
//
// Everything here runs during prepare(), OUTSIDE the game's encode hook: file reads, SHA-256 of the pinned
// immutable assets, library and pipeline creation, and the one-time buffer arena. Nothing in this header is
// allowed on the encode path.
#import <Metal/Metal.h>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace nr_exp {

// One pinned asset: path under the asset root and its required SHA-256. Loading fails closed on any mismatch.
struct AssetPin { const char* path; const char* sha256; };
const AssetPin* pinned_assets(size_t& count);

struct Assets {
  id<MTLDevice> device = nil;
  std::map<std::string, id<MTLLibrary>> libraries;          // key: asset path, e.g. "pre.metallib"
  std::map<std::string, std::vector<uint8_t>> constants;    // key: asset path, e.g. "block0.weights"
  std::map<std::string, id<MTLComputePipelineState>> pipelines;  // key: "<library>:<function>"
  std::string root, parent_root;
  bool loaded = false;
  // Set when a .metallib differed from its pin because an experiment deliberately replaced it. Weights and
  // tables are NEVER allowed to differ: only compiled kernels can, and only when explicitly asked for.
  bool unpinned_libraries = false;
  std::map<std::string, std::string> library_sha256;   // actual hash of every library actually loaded

  // ---- EXPERIMENTAL per-kernel override: nr_wide_qkv ONLY, on the three wide tiers ----
  // Strictly opt-in and OFF unless NR_EXP_QKV_OVERRIDE names a directory. When it is off, not one byte
  // of behaviour changes: no file is read, no library is loaded, and every pipeline comes from the
  // pinned library exactly as before. When it is on, ONLY the "nr_wide_qkv" function is taken from the
  // override library for its tier; every other pipeline - including every other function in the SAME
  // library - still comes from the pinned one. That is the point: replacing a whole library would drag
  // in unrelated codegen drift and it would be attributed to the kernel under test.
  // The override never relaxes a pin. Weights, tables and every non-QKV kernel stay exactly as pinned,
  // and a hash mismatch in the override manifest is a hard failure before any GPU work.
  std::map<std::string, id<MTLLibrary>> qkv_override_libraries;   // key: pinned library path it replaces for
  std::map<std::string, std::string> qkv_override_sha256;         //      nr_wide_qkv only
  std::string qkv_override_root, qkv_override_version, qkv_override_build;
  std::string qkv_override_manifest_sha256;   // digest of the exact manifest bytes that were parsed
  // Per-tier dispatch geometry DECLARED by the override manifest. Absent means the accepted geometry.
  // The only other accepted value is "mma8x8-rows8-v1", which a candidate must declare explicitly; the
  // global 64-thread contract is not relaxed, and an undeclared library keeps the accepted dispatch.
  std::map<std::string, std::string> qkv_override_dispatch;
  // The set of functions this override replaces. Historically exactly nr_wide_qkv; a manifest may now
  // declare several, so one candidate can carry native implementations of more than one wide stage and
  // their combined effect can be measured. Every name must be one this build recognises as overridable.
  std::set<std::string> qkv_override_functions;   // union, for labels and back-compat only
  // The real selector: which functions are overridden FOR WHICH LIBRARY. A global function set made
  // an unrelated tier pull rebuilt code for stages it never meant to change - C64 and C128 were
  // taking expand/project/mix from the rebuilt library instead of the pinned one purely because a
  // C256 candidate named those functions.
  std::map<std::string, std::set<std::string>> qkv_override_by_library;
  // Where every encode pipeline actually came from, and that library's hash: "<lib>:<fn>" ->
  // "<resolved library>@<sha256>". Emitted so an unchanged stage can be ASSERTED to resolve to the
  // pinned library rather than assumed to.
  std::map<std::string, std::string> resolved_pipeline_source;
  // Tiers whose declared override library is byte-identical to the pinned one: declared so a partial
  // override cannot masquerade as a whole one, but not actually a substitution, so no duplicate
  // MTLLibrary is created for them.
  std::set<std::string> qkv_override_unchanged_tiers;
  bool qkv_override_active = false;
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // ---- EXPERIMENTAL encoder dense stages (ENCODER-DENSE-WS8): OFF unless NR_EXP_ENCODER_DENSE names a package ----
  // The five linear stages of a wide block (expand, mid, mix, qkv, project) run on prepacked weight tables and
  // weight-shared x8 kernels whose fast semantics are identical, by construction, to the front tile engine. The
  // manifest binds every dense library to the exact override engine library it was proven against, and loading
  // refuses unless that engine is the one actually selected for those five stages. Tier bits follow the walk's
  // tier numbering: bit 1 C64, bit 2 C128, bit 3 C256.
  std::map<std::string, id<MTLLibrary>> encdense_libraries;   // key: enc_dense64.metallib, enc_dense128.metallib, enc_dense256.metallib
  std::map<std::string, std::string> encdense_sha256;
  std::string encdense_root, encdense_version, encdense_manifest_sha256;
  uint32_t encdense_tier_mask = 0;
  bool encdense_active = false;
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
  // DENSE-MLP-FUSION-v1: tiers (encdense_tier_mask bits) whose manifest declares "fused": ["expand+activate"]. Their stage 0 writes the
  // activation bytes with the gated fused kernel and activate is skipped.
  uint32_t encdense_fuse_mask = 0;
#endif
#endif

  const std::vector<uint8_t>& constant(const std::string& key) const;
  // Creates on first use during prepare() and never during encode(); returns nil and sets error on failure.
  id<MTLComputePipelineState> pipeline(const std::string& library, const std::string& function, std::string& error);
};

// Reads and verifies every pinned asset, builds every library, and pre-creates the pipelines the encode path
// needs so that encode() never compiles anything. Returns false with a reason on any failure.
// `allow_unpinned_libraries` is an EXPERIMENT switch: it permits a .metallib whose bytes differ from the pin
// (for example a kernel variant built from the same source), records the actual hash, and marks the Assets so
// that no caller can present the result as the pinned configuration. Every non-library asset - all weights and
// tables - must still match its pin exactly, so no weight can drift under any setting.
bool load_assets(id<MTLDevice> device, const std::string& asset_root, Assets& out, std::string& error,
                 bool allow_unpinned_libraries = false);

// The single function the experimental per-kernel override may replace. Anything else is refused.
extern const char* const kQkvOverrideFunction;   // "nr_wide_qkv" - the default when a manifest names none
// Every wide-stage function a candidate is allowed to replace. Anything outside this list is refused.
bool is_overridable_wide_function(const std::string& fn);

// Pipelines the encode path uses, by library and function. Pre-created by load_assets().
struct PipelineName { const char* library; const char* function; };
const PipelineName* encode_pipelines(size_t& count);

} // namespace nr_exp
