#include "nr_experimental_assets.h"
#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#include <array>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace nr_exp {
const char* parent_root();

static std::string sha256_hex(const void* data, size_t bytes) {
  std::array<unsigned char, CC_SHA256_DIGEST_LENGTH> h{};
  CC_SHA256(data, static_cast<CC_LONG>(bytes), h.data());
  std::ostringstream o; o << std::hex << std::setfill('0');
  for (auto v : h) o << std::setw(2) << unsigned(v);
  return o.str();
}
static bool read_file(const std::string& path, std::vector<uint8_t>& out, std::string& error) {
  std::ifstream s(path, std::ios::binary | std::ios::ate);
  if (!s) { error = "cannot open " + path; return false; }
  const auto n = s.tellg(); s.seekg(0);
  out.resize(size_t(n));
  if (n && !s.read(reinterpret_cast<char*>(out.data()), n)) { error = "cannot read " + path; return false; }
  return true;
}

const std::vector<uint8_t>& Assets::constant(const std::string& key) const { return constants.at(key); }

const char* const kQkvOverrideFunction = "nr_wide_qkv";

// The wide stages a candidate may replace. All of them are the same logical shape - an output column n
// accumulating chunk-wise 32-product dot products with a half result between chunks - which is why one
// native matrix implementation can serve them. Anything not on this list is refused, so a manifest
// cannot quietly replace a stage nobody has reasoned about.
bool is_overridable_wide_function(const std::string& fn) {
  static const std::set<std::string> allowed = {
    "nr_wide_qkv", "nr_wide_expand", "nr_wide_mid", "nr_wide_mix",
    "nr_wide_scores", "nr_wide_values", "nr_wide_project",
    // C32 tile engine, widened one function at a time and ONLY after that function's own contract has
    // been read and an adapter written for it. Shared verbs are not shared contracts: nr_swin32_expand
    // is a single 32-term chunk seeded at zero with input stride 32 and weight base (col/32)*1024, which
    // is NOT what nr_wide_expand does. Selection stays per (library, function), so widening the allowlist
    // does not by itself change any stage - a candidate must still declare the exact library and function,
    // and the runtime still refuses if an undeclared stage resolved to an override or a declared one
    // resolved to pinned.
    "nr_swin32_expand",
    // ---- DIAGNOSTIC-ONLY admissions, added for the per-stage INTERCHANGEABILITY test ----
    // The rule above - read the contract and write an adapter before widening - exists to stop a NATIVE
    // candidate reaching a stage nobody has adapted. It does not apply to declaring the UNMODIFIED,
    // ORIGINAL-ARITHMETIC rebuild of a stage, because that is the same kernel: there is no adapter to
    // write. Without these four, Block 0's activate, normalize, probabilities and project could not be
    // tested AT ALL, and the refusal was being recorded as if it were evidence they do not transfer.
    // Admission stays necessary-but-not-sufficient: selection is per (library, function), the runtime
    // still refuses if an undeclared stage resolves to an override or a declared one resolves to pinned,
    // and every manifest hash check is unchanged. A NATIVE implementation of any of these four still
    // needs its own contract read and adapter written first, exactly as the five above did.
    "nr_swin32_activate", "nr_swin32_normalize", "nr_swin32_probabilities", "nr_swin32_project",
    // Second engine operation. Its contract is RICHER than expand's: a residual seed through
    // nr_channel_from_dense with a per-channel scale, four chunk-boundary half packs, and FOUR required
    // partial outputs (one per chunk) with a coverage increment each.
    // IMPORTANT, established by isolation experiment: the same native adapter is BIT-IDENTICAL in the
    // encoder (prefix.metallib) and DIVERGES in Block 0 (pre.metallib) - 6,220,799 of 8,294,400 components,
    // max abs 0.99. So pre.metallib's nr_swin32_contract is NOT the same kernel as prefix.metallib's. The
    // allowlist is a per-FUNCTION gate, not a per-library one, so admission here is necessary but not
    // sufficient: a candidate must still declare the exact (library, function), and the shipped engine
    // declares nr_swin32_contract for prefix.metallib ONLY. Evidence:
    // build/nr-c32-engine-v1/REJECTED-c32-contract.json and the DIAG-c32-contract-* packages.
    "nr_swin32_contract",
    // Third engine operation. Contract read before widening: source is the LAST contraction partial
    // (chunk 3) with a channel permutation on k, each source half is round-tripped through the E4M3
    // storage boundary before it multiplies, single 32-term chunk seeded at zero, output width 96.
    "nr_swin32_qkv",
    // Fourth and fifth engine operations, contracts read before widening:
    //  values - two chunks with TWO required partial outputs, a k->key permutation, the E4M3 storage
    //           round-trip on the V operand, and a window shift (row/64)*64 on the V rows;
    //  scores - a per-element BIAS SEED from weights at nr_bias_offset(query,key) so the accumulator does
    //           NOT start at zero, both operands from `normalized` at different offsets, one pack.
    "nr_swin32_values", "nr_swin32_scores"
    // The four non-matrix Block 0 stages (activate, normalize, probabilities, project) were admitted for one
    // diagnostic and removed again. The attempt is preserved: the dispatch identity check REFUSED
    // nr_swin32_probabilities - "4320 groups and 0 tiles" - because its output width is not a multiple of 8,
    // so it cannot use the tile8x8 mapping. The dispatch declaration is per LIBRARY, so tiled and non-tiled
    // stages cannot be declared together; a self-consistent all-stage Block 0 substitution would need a
    // second dispatch kind. The refusal is the guard working, not a defect.
  };
  return allowed.count(fn) != 0;
}

id<MTLComputePipelineState> Assets::pipeline(const std::string& library, const std::string& function, std::string& error) {
  const std::string key = library + ":" + function;
  auto it = pipelines.find(key);
  if (it != pipelines.end()) return it->second;
  // The override is consulted for EXACTLY ONE function name, and only when it is active and carries a
  // library for this tier. Every other function - including the others in this same library - falls
  // through to the pinned library below, unchanged.
  id<MTLLibrary> source_lib = nil;
  bool from_override = false;
  if (qkv_override_active) {
    auto byl = qkv_override_by_library.find(library);
    if (byl != qkv_override_by_library.end() && byl->second.count(function)) {
      auto ov = qkv_override_libraries.find(library);
      if (ov != qkv_override_libraries.end()) { source_lib = ov->second; from_override = true; }
    }
  }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  // Dense-stage libraries have their own names (enc_dense*.metallib), so they can never shadow a pinned library.
  bool from_encdense = false;
  if (source_lib == nil && encdense_active) {
    auto dl = encdense_libraries.find(library);
    if (dl != encdense_libraries.end()) { source_lib = dl->second; from_encdense = true; }
  }
#endif
  if (source_lib == nil) {
    auto lib = libraries.find(library);
    if (lib == libraries.end()) { error = "library not loaded: " + library; return nil; }
    source_lib = lib->second;
  }
  {   // record where this pipeline actually resolved, with the hash of the library it came from
    auto h = from_override ? qkv_override_sha256.find(library) : library_sha256.find(library);
    const std::string sha = (from_override ? (h == qkv_override_sha256.end() ? std::string("?") : h->second)
                                           : (h == library_sha256.end() ? std::string("?") : h->second));
    resolved_pipeline_source[key] = (from_override ? "override:" : "pinned:") + library + "@" + sha;
  }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
  if (from_encdense) resolved_pipeline_source[key] = "override:encdense:" + library + "@" + encdense_sha256[library];
#endif
  id<MTLFunction> fn = [source_lib newFunctionWithName:[NSString stringWithUTF8String:function.c_str()]];
  if (!fn) { error = "function not found: " + key; return nil; }
  NSError* err = nil;
  id<MTLComputePipelineState> ps = [device newComputePipelineStateWithFunction:fn error:&err];
  if (!ps) { error = "pipeline failed: " + key + ": " + std::string(err.localizedDescription.UTF8String); return nil; }
  pipelines[key] = ps;
  return ps;
}

// Every pipeline the encode path can reach. Created during prepare() so encode() never compiles.
static const PipelineName kEncodePipelines[] = {
  {"entry.metallib", "nr_pre_entry_layout"},
  {"pre.metallib", "nr_pre_features"}, {"pre.metallib", "nr_pre_adapter"},
  {"pre.metallib", "nr_swin32_expand"}, {"pre.metallib", "nr_swin32_activate"}, {"pre.metallib", "nr_swin32_contract"},
  {"pre.metallib", "nr_swin32_qkv"}, {"pre.metallib", "nr_swin32_normalize"}, {"pre.metallib", "nr_swin32_scores"},
  {"pre.metallib", "nr_swin32_probabilities"}, {"pre.metallib", "nr_swin32_values"}, {"pre.metallib", "nr_swin32_project"},
  {"pre.metallib", "nr_pre_export"},
  {"temporal-hardware.metallib", "nr_pre_temporal_features"},
  // Encoder blocks 1-4 (C32) and the block-4 downsample.
  {"prefix.metallib", "nr_swin32_expand"}, {"prefix.metallib", "nr_swin32_activate"}, {"prefix.metallib", "nr_swin32_contract"},
  {"prefix.metallib", "nr_swin32_qkv"}, {"prefix.metallib", "nr_swin32_normalize"}, {"prefix.metallib", "nr_swin32_scores"},
  {"prefix.metallib", "nr_swin32_probabilities"}, {"prefix.metallib", "nr_swin32_values"}, {"prefix.metallib", "nr_swin32_project"},
  {"prefix.metallib", "nr_swin32_pack"}, {"prefix.metallib", "nr_swin32_rewindow"},
  {"prefix.metallib", "nr_swin32_downsample_pool"}, {"prefix.metallib", "nr_swin32_downsample_project"},
};
// The wide tiers use the same eleven function names in three libraries; expanded below so that every pipeline
// the encode path can reach exists before the game hook is ever entered.
static const char* kWideLibraries[] = {"prefix.metallib", "wide128.metallib", "wide256.metallib"};
static const char* kWideFunctions[] = {"nr_wide_expand", "nr_wide_activate", "nr_wide_mid", "nr_wide_mix", "nr_wide_qkv",
                                       "nr_wide_normalize", "nr_wide_scores", "nr_swin32_probabilities", "nr_wide_values",
                                       "nr_wide_project", "nr_wide_pack", "nr_wide_rewindow", "nr_wide_pool", "nr_wide_down_project"};
const PipelineName* encode_pipelines(size_t& count) { count = sizeof(kEncodePipelines) / sizeof(kEncodePipelines[0]); return kEncodePipelines; }

bool load_assets(id<MTLDevice> device, const std::string& asset_root, Assets& out, std::string& error,
                 bool allow_unpinned_libraries) {
  @autoreleasepool {
    if (!device) { error = "no Metal device"; return false; }
    if (asset_root.empty()) { error = "asset root required"; return false; }
    out.device = device; out.root = asset_root; out.parent_root = parent_root();
    size_t n = 0; const AssetPin* pins = pinned_assets(n);
    for (size_t i = 0; i < n; ++i) {
      const std::string rel = pins[i].path, path = asset_root + "/" + rel;
      std::vector<uint8_t> bytes;
      if (!read_file(path, bytes, error)) return false;
      const std::string got = sha256_hex(bytes.data(), bytes.size());
      const bool is_library = rel.size() > 9 && rel.compare(rel.size() - 9, 9, ".metallib") == 0;
      if (got != pins[i].sha256) {   // fail closed, except for a deliberately substituted kernel library
        if (!(is_library && allow_unpinned_libraries)) {
          error = "pinned asset mismatch: " + rel + " has " + got.substr(0, 16) + ", expected " + std::string(pins[i].sha256).substr(0, 16);
          return false;
        }
        out.unpinned_libraries = true;
      }
      if (is_library) {
        out.library_sha256[rel] = got;
        NSError* err = nil;
        dispatch_data_t d = dispatch_data_create(bytes.data(), bytes.size(), dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
        id<MTLLibrary> lib = [device newLibraryWithData:d error:&err];
        if (!lib) { error = "library load failed: " + rel + ": " + std::string(err.localizedDescription.UTF8String); return false; }
        out.libraries[rel] = lib;
      } else {
        out.constants[rel] = std::move(bytes);
      }
    }
    // ---- EXPERIMENTAL nr_wide_qkv override, loaded BEFORE any pipeline is created ----
    // Absent env: nothing below runs and the configuration is bit-for-bit the pinned one.
    if (const char* root = std::getenv("NR_EXP_QKV_OVERRIDE")) {
      if (root[0]) {
        // (1) The whole point of a per-kernel override is that EVERY other pipeline is the pinned one.
        // The broad unpinned path permits any encoder or body library to differ too, which would make
        // the isolation claim false while still looking like a QKV-only experiment. Refuse the
        // combination outright rather than producing a result nobody can interpret.
        if (allow_unpinned_libraries) {
          error = "qkv override: refusing to combine a per-kernel override with the broad unpinned "
                  "(ApproximateNative) path - that path lets non-QKV libraries differ too, so the "
                  "isolation this override exists to provide would not hold. Use one or the other.";
          return false;
        }
        const std::string dir = root, manifest_path = dir + "/qkv-override.json";
        std::vector<uint8_t> mbytes;
        if (!read_file(manifest_path, mbytes, error)) {
          error = "qkv override: manifest unreadable at " + manifest_path + ": " + error; return false;
        }
        // (3) Bind the manifest itself, not just the values read out of it: record the digest of the
        // exact bytes that were parsed, so the recorded evidence identifies one manifest and not a
        // family of manifests that happen to share a version string.
        out.qkv_override_manifest_sha256 = sha256_hex(mbytes.data(), mbytes.size());
        NSError* jerr = nil;
        NSData* jd = [NSData dataWithBytes:mbytes.data() length:mbytes.size()];
        NSDictionary* m = [NSJSONSerialization JSONObjectWithData:jd options:0 error:&jerr];
        if (![m isKindOfClass:[NSDictionary class]]) { error = "qkv override: manifest is not a JSON object"; return false; }
        auto str = [&](const char* k) -> std::string {
          id v = m[[NSString stringWithUTF8String:k]];
          return [v isKindOfClass:[NSString class]] ? std::string([v UTF8String]) : std::string();
        };
        if (str("schema") != "neuralstudio.nr.qkv-override.v1") {
          error = "qkv override: unsupported manifest schema '" + str("schema") + "'"; return false;
        }
        // "overrides" (per library) is the precise form and supersedes both legacy keys; "functions"
        // (array) and "function" (string) remain valid and mean "on every declared library".
        out.qkv_override_functions.clear();
        const bool has_overrides = [m[@"overrides"] isKindOfClass:[NSDictionary class]];
        if (has_overrides) {
          // function names are validated per library in the overrides block below
        } else if (NSArray* fa = m[@"functions"]) {
          if (![fa isKindOfClass:[NSArray class]] || fa.count == 0) {
            error = "qkv override: 'functions' must be a non-empty array"; return false; }
          for (id v in fa) {
            if (![v isKindOfClass:[NSString class]]) { error = "qkv override: bad entry in 'functions'"; return false; }
            const std::string fn = [v UTF8String];
            if (!is_overridable_wide_function(fn)) {
              error = "qkv override: '" + fn + "' is not a wide stage this build allows a candidate to replace";
              return false;
            }
            out.qkv_override_functions.insert(fn);
          }
        } else {
          const std::string fn = str("function");
          if (!is_overridable_wide_function(fn)) {
            error = "qkv override: '" + fn + "' is not a wide stage this build allows a candidate to replace";
            return false;
          }
          out.qkv_override_functions.insert(fn);
        }
        out.qkv_override_version = str("version");
        out.qkv_override_build = str("build");
        if (out.qkv_override_version.empty()) { error = "qkv override: manifest has no version"; return false; }
        NSDictionary* libs = m[@"libraries"];
        if (![libs isKindOfClass:[NSDictionary class]] || libs.count == 0) {
          error = "qkv override: manifest has no libraries map"; return false;
        }
        for (NSString* rel in libs) {
          const std::string relc = [rel UTF8String];
          // Only the three wide tiers may be overridden, and only libraries the pinned set already has.
          bool is_wide = false;
          for (const char* w : kWideLibraries) if (relc == w) is_wide = true;
          // Block 0's library. It is admitted for the SAME reason the wide tiers are - it is one of this
          // build's own C32/wide kernel libraries - and admitting it changes nothing on its own: selection
          // is still per (library, function), only a function on the allowlist may be declared, and the
          // fail-closed both-directions assertion still refuses an undeclared stage that resolved to an
          // override or a declared stage that resolved to pinned. Block 0 and the encoder keep SEPARATE
          // libraries and separate declarations; a kernel is never shared implicitly between them.
          if (relc == "pre.metallib") is_wide = true;
          if (!is_wide) { error = "qkv override: '" + relc + "' is not one of the libraries this build allows a candidate to replace (wide tiers or pre.metallib)"; return false; }
          if (out.libraries.find(relc) == out.libraries.end()) {
            error = "qkv override: '" + relc + "' is not part of the pinned set"; return false;
          }
          id want = libs[rel];
          if (![want isKindOfClass:[NSString class]]) { error = "qkv override: bad hash entry for " + relc; return false; }
          const std::string expect = [want UTF8String];
          std::vector<uint8_t> lb;
          if (!read_file(dir + "/" + relc, lb, error)) { error = "qkv override: " + relc + ": " + error; return false; }
          const std::string got = sha256_hex(lb.data(), lb.size());
          if (got != expect) {   // fail closed, before any GPU work
            error = "qkv override: " + relc + " has " + got.substr(0, 16) + ", manifest declares " +
                    expect.substr(0, 16); return false;
          }
          // If this tier's override library is BYTE-IDENTICAL to the pinned one, it is not an override
          // at all - the manifest names it only because a candidate must declare all three tiers, so a
          // partial override cannot masquerade as a whole one. Creating a second MTLLibrary for the
          // same bytes measurably costs time in the tiers that did NOT change: with duplicates loaded,
          // C64 ran +77.5 ms and C128 +131.9 ms slower on byte-identical code (both resolved, block
          // design, SE < 1 ms). Reuse the pinned library instead; the declaration and its hash check
          // are unaffected.
          auto pinned_it = out.library_sha256.find(relc);
          if (pinned_it != out.library_sha256.end() && pinned_it->second == got) {
            out.qkv_override_libraries[relc] = out.libraries[relc];
            out.qkv_override_sha256[relc] = got;
            out.qkv_override_unchanged_tiers.insert(relc);
            continue;
          }
          NSError* lerr = nil;
          dispatch_data_t d = dispatch_data_create(lb.data(), lb.size(), dispatch_get_main_queue(),
                                                   DISPATCH_DATA_DESTRUCTOR_DEFAULT);
          id<MTLLibrary> lib = [device newLibraryWithData:d error:&lerr];
          if (!lib) { error = "qkv override: library load failed for " + relc + ": " +
                              std::string(lerr.localizedDescription.UTF8String); return false; }
          for (const auto& fn : out.qkv_override_functions) {
            if (![lib newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]]) {
              error = "qkv override: " + relc + " does not define " + fn; return false;
            }
          }
          out.qkv_override_libraries[relc] = lib;
          out.qkv_override_sha256[relc] = got;
        }
        // Optional per-tier dispatch geometry. Only a name we recognise is accepted, and only for a
        // tier the manifest already declares a library for.
        if (NSDictionary* dsp = m[@"dispatch"]) {
          if (![dsp isKindOfClass:[NSDictionary class]]) { error = "qkv override: 'dispatch' must be an object"; return false; }
          for (NSString* rel in dsp) {
            const std::string relc = [rel UTF8String];
            id v = dsp[rel];
            if (![v isKindOfClass:[NSString class]]) { error = "qkv override: bad dispatch entry for " + relc; return false; }
            const std::string geom = [v UTF8String];
            // c32-tile32-v1: one 32x32 OUTPUT TILE per 64-thread group, so the stage issues
            // (rows/32)*(out_width/32) groups instead of rows*out_width/64 - one sixteenth of them.
            // Added because a null kernel that returns immediately in every threadgroup costs 42.51 ms
            // on nr_swin32_expand against the real kernel's 50.52 ms: 84% of that stage is threadgroup
            // ISSUE, and no existing kind can issue fewer groups. The existing kinds are UNCHANGED; a
            // candidate must declare this one explicitly, per (library, function), and the encode path
            // still refuses before any GPU work if the geometry identity does not hold.
            if (geom != "accepted-64" && geom != "tile8x8-in-accepted-dispatch-v1" &&
                geom != "mma8x8-rows8-v1" && geom != "c32-tile32-v1") {
              error = "qkv override: unknown dispatch geometry '" + geom + "' for " + relc +
                      "; this build admits only tile8x8-in-accepted-dispatch-v1, mma8x8-rows8-v1 or c32-tile32-v1"; return false;
            }
            if (out.qkv_override_libraries.find(relc) == out.qkv_override_libraries.end()) {
              error = "qkv override: dispatch declared for " + relc + ", which this manifest does not override";
              return false;
            }
            out.qkv_override_dispatch[relc] = geom;
          }
        }
        // (3) The benchmarked contract is all THREE tiers. A manifest missing one would silently
        // become a partial override - some tiers hoisted, some pinned - and the number it produced
        // would not be the number anyone thinks they measured.
        for (const char* w : kWideLibraries) {
          if (out.qkv_override_libraries.find(w) == out.qkv_override_libraries.end()) {
            error = "qkv override: manifest is missing tier " + std::string(w) +
                    "; this mechanism requires all three wide tiers, because a partial override would "
                    "silently mix hoisted and pinned tiers in one measurement";
            return false;
          }
        }
        // Per-library selection. "overrides" is the precise form; the legacy "functions"/"function"
        // means "these functions on every declared library" and is kept so existing evidence keeps
        // its meaning.
        out.qkv_override_by_library.clear();
        if (NSDictionary* ov = m[@"overrides"]) {
          if (![ov isKindOfClass:[NSDictionary class]] || ov.count == 0) {
            error = "qkv override: 'overrides' must be a non-empty object"; return false; }
          for (NSString* rel in ov) {
            const std::string relc = [rel UTF8String];
            if (out.qkv_override_libraries.find(relc) == out.qkv_override_libraries.end()) {
              error = "qkv override: 'overrides' names " + relc + ", which this manifest does not declare";
              return false;
            }
            id list = ov[rel];
            if (![list isKindOfClass:[NSArray class]] || [list count] == 0) {
              error = "qkv override: 'overrides' entry for " + relc + " must be a non-empty array"; return false; }
            for (id v in list) {
              if (![v isKindOfClass:[NSString class]]) { error = "qkv override: bad function name for " + relc; return false; }
              const std::string fn = [v UTF8String];
              if (!is_overridable_wide_function(fn)) {
                error = "qkv override: '" + fn + "' is not a wide stage this build allows a candidate to replace";
                return false;
              }
              id<MTLLibrary> l = out.qkv_override_libraries[relc];
              if (![l newFunctionWithName:[NSString stringWithUTF8String:fn.c_str()]]) {
                error = "qkv override: " + relc + " does not define " + fn; return false; }
              out.qkv_override_by_library[relc].insert(fn);
              out.qkv_override_functions.insert(fn);
            }
          }
        } else {
          for (const auto& kv : out.qkv_override_libraries)
            out.qkv_override_by_library[kv.first] = out.qkv_override_functions;
        }
        out.qkv_override_root = dir;
        out.qkv_override_active = true;
      }
    }
#ifdef NR_EXPERIMENTAL_ENCODER_DENSE
    // ---- EXPERIMENTAL encoder dense stages (ENCODER-DENSE-WS8), loaded fail-closed BEFORE any pipeline is created ----
    // Absent env: nothing below runs, and no library, pipeline or buffer differs from the build without this flag.
    if (const char* droot = std::getenv("NR_EXP_ENCODER_DENSE")) {
      if (droot[0]) {
        if (allow_unpinned_libraries) {
          error = "encoder dense: refusing to combine the dense stages with the broad unpinned (ApproximateNative) path"; return false; }
        if (!out.qkv_override_active) {
          error = "encoder dense: the dense stages are identical only to the front tile engine, which NR_EXP_QKV_OVERRIDE loads; "
                  "without it they would replace a different engine. Refusing."; return false; }
        const std::string dir = droot, mpath = dir + "/encoder-dense.json";
        std::vector<uint8_t> mb;
        if (!read_file(mpath, mb, error)) { error = "encoder dense: manifest unreadable at " + mpath + ": " + error; return false; }
        out.encdense_manifest_sha256 = sha256_hex(mb.data(), mb.size());
        NSDictionary* m = [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:mb.data() length:mb.size()] options:0 error:nil];
        if (![m isKindOfClass:[NSDictionary class]]) { error = "encoder dense: manifest is not a JSON object"; return false; }
        id schema = m[@"schema"], version = m[@"version"];
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
        // DENSE-MLP-FUSION-v1: schema v2 may declare "fused": ["expand+activate"] per tier. A runtime without this flag refuses v2, so a
        // fused package can never run unfused by accident.
        const bool schema_v2 = [schema isKindOfClass:[NSString class]] && std::string([schema UTF8String]) == "neuralstudio.nr.encoder-dense.v2";
        if (!schema_v2 && (![schema isKindOfClass:[NSString class]] || std::string([schema UTF8String]) != "neuralstudio.nr.encoder-dense.v1")) {
#else
        if (![schema isKindOfClass:[NSString class]] || std::string([schema UTF8String]) != "neuralstudio.nr.encoder-dense.v1") {
#endif
          error = "encoder dense: unsupported manifest schema"; return false; }
        if (![version isKindOfClass:[NSString class]] || [version length] == 0) { error = "encoder dense: manifest has no version"; return false; }
        out.encdense_version = [version UTF8String];
        NSDictionary* tiers = m[@"tiers"];
        if (![tiers isKindOfClass:[NSDictionary class]] || tiers.count == 0) { error = "encoder dense: manifest declares no tiers"; return false; }
        // Optional subset for bisection and per-tier measurement: NR_EXP_ENCODER_DENSE_TIERS="64,128,256,32,b0".
        std::set<std::string> allowed = {"64", "128", "256", "32", "b0"};
        if (const char* tv = std::getenv("NR_EXP_ENCODER_DENSE_TIERS")) {
          allowed.clear();
          const std::string s = tv; size_t pos = 0;
          while (pos <= s.size()) {
            size_t comma = s.find(',', pos); if (comma == std::string::npos) comma = s.size();
            const std::string tok = s.substr(pos, comma - pos);
            if (tok != "64" && tok != "128" && tok != "256" && tok != "32" && tok != "b0") {
              error = "encoder dense: NR_EXP_ENCODER_DENSE_TIERS entry '" + tok + "' is not 64, 128, 256, 32 or b0"; return false; }
            allowed.insert(tok);
            pos = comma + 1;
          }
        }
        // Wide tiers replace five stages of block_wide. The C32 tiers - the walk ("32", blocks 1-4) and Block 0 ("b0") - are the
        // HYBRID of ENCODER-DENSE32-HYBRID-GATE-v1: only expand and qkv are replaced, and the manifest must declare exactly that.
        static const char* kDenseReplaces[] = {"nr_wide_expand", "nr_wide_mid", "nr_wide_mix", "nr_wide_qkv", "nr_wide_project"};
        static const char* kDense32Replaces[] = {"nr_swin32_expand", "nr_swin32_qkv"};
        for (NSString* key in tiers) {
          const std::string ks = [key UTF8String];
          const bool c32 = ks == "32" || ks == "b0";
          const uint32_t ch = ks == "64" ? 64u : ks == "128" ? 128u : ks == "256" ? 256u : c32 ? 32u : 0u;
          if (!ch) { error = "encoder dense: tier '" + ks + "' is not 64, 128, 256, 32 or b0"; return false; }
          NSDictionary* t = tiers[key];
          if (![t isKindOfClass:[NSDictionary class]]) { error = "encoder dense: tier " + ks + " is not an object"; return false; }
          auto field = [&](NSString* k) -> std::string { id v = t[k]; return [v isKindOfClass:[NSString class]] ? std::string([v UTF8String]) : std::string(); };
          const std::string lib = field(@"library"), lib_sha = field(@"sha256"), engine = field(@"engine_library"), engine_sha = field(@"engine_sha256");
          const std::string want_lib = ks == "b0" ? "enc_dense32b0.metallib" : ch == 32 ? "enc_dense32.metallib" :
                                       ch == 64 ? "enc_dense64.metallib" : ch == 128 ? "enc_dense128.metallib" : "enc_dense256.metallib";
          const std::string want_engine = ks == "b0" ? "pre.metallib" : (ch == 32 || ch == 64) ? "prefix.metallib" : ch == 128 ? "wide128.metallib" : "wide256.metallib";
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
          bool fused = false;
          if (id fz = t[@"fused"]) {
            const bool expand_activate = schema_v2 && [fz isKindOfClass:[NSArray class]] && [(NSArray*)fz count] == 1 &&
                                         [((NSArray*)fz)[0] isKindOfClass:[NSString class]] && [((NSString*)((NSArray*)fz)[0]) isEqualToString:@"expand+activate"];
            if (!expand_activate) { error = "encoder dense: tier " + ks + " may only declare fused [\"expand+activate\"], and only in schema v2"; return false; }
            fused = true;
          }
          // A fused tier's library is the gated superset <name>f.metallib (the same dense kernels plus the fused one). It is registered
          // under the unfused name below, so every dense stage resolves unchanged and only the fused dispatch differs.
          const std::string declared_lib = fused ? want_lib.substr(0, want_lib.size() - 9) + "f.metallib" : want_lib;
          if (lib != declared_lib || engine != want_engine) {
            error = "encoder dense: tier " + ks + " must declare " + declared_lib + " against engine " + want_engine; return false; }
#else
          if (lib != want_lib || engine != want_engine) {
            error = "encoder dense: tier " + ks + " must declare " + want_lib + " against engine " + want_engine; return false; }
#endif
          if (c32) {
            NSArray* stages = t[@"stages"];
            const bool hybrid = [stages isKindOfClass:[NSArray class]] && stages.count == 2 &&
                                [stages[0] isKindOfClass:[NSString class]] && [stages[1] isKindOfClass:[NSString class]] &&
                                [stages[0] isEqualToString:@"expand"] && [stages[1] isEqualToString:@"qkv"];
            if (!hybrid) { error = "encoder dense: tier " + ks + " must declare stages [\"expand\", \"qkv\"], the only C32 stages gated for this candidate"; return false; }
          }
          if (!allowed.count(ks)) continue;
          // The binding: the stages this library replaces must resolve to the override engine library, with the exact
          // hash, that the dense kernels were proven identical to. The pinned engine or any other override is refused.
          auto osha = out.qkv_override_sha256.find(engine);
          if (osha == out.qkv_override_sha256.end() || osha->second != engine_sha) {
            error = "encoder dense: tier " + ks + " was proven against " + engine + "@" + engine_sha.substr(0, 16) + ", but the loaded override for it is " +
                    (osha == out.qkv_override_sha256.end() ? std::string("absent") : osha->second.substr(0, 16)); return false; }
          auto byl = out.qkv_override_by_library.find(engine);
          const std::vector<const char*> replaces = c32 ? std::vector<const char*>(std::begin(kDense32Replaces), std::end(kDense32Replaces))
                                                        : std::vector<const char*>(std::begin(kDenseReplaces), std::end(kDenseReplaces));
          for (const char* fn : replaces)
            if (byl == out.qkv_override_by_library.end() || !byl->second.count(fn)) {
              error = "encoder dense: tier " + ks + ": " + std::string(fn) + " does not come from the override engine the dense stage replaces"; return false; }
          std::vector<uint8_t> lb;
          if (!read_file(dir + "/" + lib, lb, error)) { error = "encoder dense: " + lib + ": " + error; return false; }
          const std::string got = sha256_hex(lb.data(), lb.size());
          if (got != lib_sha) { error = "encoder dense: " + lib + " has " + got.substr(0, 16) + ", manifest declares " + lib_sha.substr(0, 16); return false; }
          NSError* lerr = nil;
          dispatch_data_t d = dispatch_data_create(lb.data(), lb.size(), dispatch_get_main_queue(), DISPATCH_DATA_DESTRUCTOR_DEFAULT);
          id<MTLLibrary> l = [device newLibraryWithData:d error:&lerr];
          if (!l) { error = "encoder dense: library load failed for " + lib + ": " + std::string(lerr.localizedDescription.UTF8String); return false; }
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
          out.encdense_libraries[want_lib] = l;
          out.encdense_sha256[want_lib] = got;
          if (fused) out.encdense_fuse_mask |= ks == "b0" ? 0x20u : ch == 32 ? 1u : ch == 64 ? 2u : ch == 128 ? 4u : 8u;
#else
          out.encdense_libraries[lib] = l;
          out.encdense_sha256[lib] = got;
#endif
          out.encdense_tier_mask |= ks == "b0" ? 0x20u : ch == 32 ? 1u : ch == 64 ? 2u : ch == 128 ? 4u : 8u;
        }
        if (out.encdense_tier_mask == 0) { error = "encoder dense: no tier selected"; return false; }
        out.encdense_root = dir;
        out.encdense_active = true;
        for (const auto& kv : out.encdense_libraries) {
          const bool c32lib = kv.first.rfind("enc_dense32", 0) == 0;
          const std::vector<const char*> fns = c32lib ? std::vector<const char*>{"nrd32w_prepack", "nrd32w_decode", "nrd32w_quant_perm", "nrd32w_linear_ws8"}
                                                      : std::vector<const char*>{"nrd_prepack", "nrd_decode", "nrd_quant_perm", "nrd_seed", "nrd_linear_ws8"};
          for (const char* fn : fns)
            if (!out.pipeline(kv.first, fn, error)) return false;
          // the kernel's lane -> accumulator-element mapping is defined for 32-lane SIMD groups only
          if (out.pipelines[kv.first + (c32lib ? ":nrd32w_linear_ws8" : ":nrd_linear_ws8")].threadExecutionWidth != 32) {
            error = "encoder dense: " + kv.first + " linear kernel threadExecutionWidth is not 32 on this device; refusing"; return false; }
#ifdef NR_EXPERIMENTAL_DENSE_FUSE
          const uint32_t lib_bit = kv.first == "enc_dense32b0.metallib" ? 0x20u : kv.first == "enc_dense32.metallib" ? 1u :
                                   kv.first == "enc_dense64.metallib" ? 2u : kv.first == "enc_dense128.metallib" ? 4u : 8u;
          if (out.encdense_fuse_mask & lib_bit) {
            const std::string ffn = c32lib ? "nrd32w_expand_activate_ws8" : "nrd_expand_activate_ws8";
            if (!out.pipeline(kv.first, ffn, error)) return false;
            if (out.pipelines[kv.first + ":" + ffn].threadExecutionWidth != 32) {
              error = "encoder dense: " + kv.first + " fused kernel threadExecutionWidth is not 32 on this device; refusing"; return false; }
          }
#endif
        }
      }
    }
#endif
    size_t pn = 0; const PipelineName* ps = encode_pipelines(pn);
    for (size_t i = 0; i < pn; ++i)
      if (!out.pipeline(ps[i].library, ps[i].function, error)) return false;
    for (const char* lib : kWideLibraries)
      for (const char* fn : kWideFunctions)
        if (!out.pipeline(lib, fn, error)) return false;
    if (!out.pipeline("wide256.metallib", "nr_encoder_native_export", error)) return false;
    out.loaded = true;
    return true;
  }
}
} // namespace nr_exp
