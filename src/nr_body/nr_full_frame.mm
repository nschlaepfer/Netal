// nr_full_frame.mm — fully in-memory single-process NR frame: input colour texture -> Codex's encoder runtime API (Block0, layout, encoder 1-22;
// build-v1 object linked by path) -> Fable body runtime v2 (middle 25-132, decoder/output 133-156) -> caller-owned output surface. No activation
// file crosses either boundary; the only files read are the input image, the pinned model constants named by the two runtimes' descriptors, and
// (CPU comparison only) an optional RTX reference frame. Prints one JSON receipt. Usage:
//   nr_full_frame CHAIN.json ENCODER_ASSET_ROOT INPUT.rgba16le POISON_HEX [--reference OUT.rgba16le] [--save-final PATH] [--receipt PATH] [--frame INPUT POISON REFERENCE]...
// Additional --frame triples run in the SAME process on the SAME prepared encoder and body contexts (resident weights, pipelines, scratch), each with a fresh texture.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "nr_body_runtime.h"
#include "nr_temporal/nr_temporal.h"
#include "nr_encoder_runtime.h"
#if NR_FULL_FRAME_TEMPORAL_PRODUCER
#include "nr_encoder_temporal_runtime.h"   // v4: Codex's experimental non-reset producer (nr_encoder_run_temporal_frame), used only for reset=false frames of a temporal sequence; the build script defines this only when the linked encoder build ships the header
#endif
#if NR_FULL_FRAME_TELEMETRY
#include "nr_encoder_telemetry.h"   // v10: Codex's optional checked-mode encoder telemetry (enable after prepare, copy after the frame's own checks, outside timing); defined only when the linked encoder build ships the header
#endif
#include <CommonCrypto/CommonDigest.h>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <unistd.h>
#include <mach-o/dyld.h>
#include <vector>
static std::string Sha(const void* p, size_t n) { unsigned char d[32]; CC_SHA256(p, (CC_LONG)n, d); char h[65]; for (int i = 0; i < 32; ++i) snprintf(h + 2 * i, 3, "%02x", d[i]); return std::string(h, 64); }
static std::vector<uint8_t> Read(const std::string& p) { std::ifstream s(p, std::ios::binary | std::ios::ate); if (!s) throw std::runtime_error("open failed: " + p); const size_t n = size_t(s.tellg()); s.seekg(0); std::vector<uint8_t> v(n); s.read(reinterpret_cast<char*>(v.data()), std::streamsize(n)); return v; }
static std::string SelfSha() { char buf[4096]; uint32_t n = sizeof(buf); if (_NSGetExecutablePath(buf, &n) != 0) return ""; auto v = Read(buf); return Sha(v.data(), v.size()); }
static std::string J(const std::string& s) { std::string o = "\""; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o + "\""; }
static std::string Reg(const char* name, const NrBodyRegion& r) { std::ostringstream o; o << "  " << J(name) << ": {\"offset\": " << r.offset << ", \"bytes\": " << r.bytes << ", \"sha256\": " << J(r.provenance_sha256) << "}"; return o.str(); }
int main(int argc, const char** argv) { @autoreleasepool {
  if (argc < 5) { std::cerr << "usage: nr_full_frame CHAIN.json ENCODER_ASSET_ROOT INPUT.rgba16le POISON_HEX [--reference OUT.rgba16le] [--save-final PATH] [--receipt PATH]\n"; return 2; }
  std::string reference, save_final, receipt, save_dir, diag_dir, reset_sequence, motion_file; bool temporal = false; unsigned encoder_sampler_variant = 0; bool recovery_test = false; bool encoder_telemetry = false; /* v10: --encoder-telemetry (measurement builds only) */ bool body_checked = false; /* v11: --body-checked = body production=false (checked mode: every intermediate snapshotted and SHA-256 hashed; decoder per-launch checks on) */ long encoder_diag_frame = -1; std::string encoder_diag_dir; /* v8: optional encoder diagnostics for ONE frame index (default off) */ struct Extra { std::string input, poison, reference; }; std::vector<Extra> extra;
  for (int i = 5; i < argc; i += 2) { std::string k = argv[i]; const bool is_flag = (k == "--temporal" || k == "--recovery-test" || k == "--encoder-telemetry" || k == "--body-checked"); if (!is_flag && i + 1 >= argc) { std::cerr << "option " << k << " needs a value\n"; return 2; }   // v11: a trailing flag is parsed (v10 silently dropped it: --body-checked / --encoder-telemetry as the last argument were ignored)
    if (k == "--reference") reference = argv[i + 1]; else if (k == "--save-final") save_final = argv[i + 1]; else if (k == "--receipt") receipt = argv[i + 1]; else if (k == "--save-dir") save_dir = argv[i + 1]; else if (k == "--diag-dir") diag_dir = argv[i + 1]; /* per-frame body diagnostics (chain-run.json with every launch/stage hash) under DIR/frame-N */ else if (k == "--reset-sequence") { reset_sequence = argv[i + 1]; temporal = true; } else if (k == "--motion") motion_file = argv[i + 1]; else if (k == "--encoder-sampler-variant") encoder_sampler_variant = unsigned(std::stoul(argv[i + 1])); else if (k == "--temporal") { temporal = true; --i; } else if (k == "--recovery-test") { recovery_test = true; --i; } else if (k == "--encoder-telemetry") { encoder_telemetry = true; --i; } else if (k == "--body-checked") { body_checked = true; --i; } else if (k == "--encoder-diagnostics" && i + 2 < argc) { encoder_diag_frame = std::stol(argv[i + 1]); encoder_diag_dir = argv[i + 2]; ++i; } else if (k == "--frame" && i + 3 < argc) { extra.push_back({argv[i + 1], argv[i + 2], argv[i + 3]}); i += 2; } else { std::cerr << "unknown option " << k << "\n"; return 2; } }
  const uint8_t poison = uint8_t(std::stoul(argv[4], nullptr, 16) & 0xffu); const auto t0 = std::chrono::steady_clock::now();
  std::ostringstream js; js << "{\n \"schema\": \"neuralstudio.nr.full-frame-inmemory.v13\",\n \"executable_sha256\": " << J(SelfSha()) << ",\n \"command\": ["; for (int i = 0; i < argc; ++i) js << (i ? ", " : "") << J(argv[i]); js << "],\n \"body_descriptor_sha256\": " << J([&]{ auto v = Read(argv[1]); return Sha(v.data(), v.size()); }()) << ",\n \"encoder_asset_root\": " << J(argv[2]) << ",\n \"label\": \"fully in-memory single-process frames: texture -> encoder runtime API (reset entry, or the experimental temporal entry on reset=false frames with the shared history state) -> body runtime; no activation files; history only from this process prior native output\",\n \"pid\": " << getpid() << ",\n \"poison\": " << unsigned(poison) << ",\n";
  int code = 2;
  // pre-run file identities (re-hashed immediately after the run: executable, encoder asset files, and every body library file seen at load)
  std::map<std::string, std::string> pre_files; { char self[4096]; uint32_t n = sizeof(self); if (_NSGetExecutablePath(self, &n) == 0) { auto v = Read(self); pre_files[self] = Sha(v.data(), v.size()); }
    for (auto& e : std::filesystem::directory_iterator(argv[2])) if (e.is_regular_file()) { auto v = Read(e.path().string()); pre_files[e.path().string()] = Sha(v.data(), v.size()); } }
  std::map<std::string, std::string> body_libs;   // "lib:<path>" -> load-time hash, collected from frames
  try {
    NSError* err = nil; NSDictionary* chain = [NSJSONSerialization JSONObjectWithData:[NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]] options:0 error:&err]; if (!chain) throw std::runtime_error("body descriptor unreadable");
    id<MTLDevice> dev = MTLCreateSystemDefaultDevice(); id<MTLCommandQueue> queue = [dev newCommandQueue];
    // encoder runtime (Codex, build-v1): prepare once
    NrEncoderContext ectx; ectx.device = dev; ectx.queue = queue; std::string eerr; const auto tp0 = std::chrono::steady_clock::now();
    if (nr_encoder_prepare(ectx, argv[2], eerr) != 0) throw std::runtime_error("encoder prepare rejected: " + eerr);
    auto enable_telemetry = [&]() { if (!encoder_telemetry) return;
#if NR_FULL_FRAME_TELEMETRY
      std::string terr; if (nr_encoder_enable_telemetry(ectx, true, terr) != 0) throw std::runtime_error("encoder telemetry enable rejected: " + terr);
#else
      throw std::runtime_error("--encoder-telemetry requested but this harness build has no telemetry header (linked encoder build ships no nr_encoder_telemetry.h); fail closed");
#endif
    }; enable_telemetry();
    const double prep_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tp0).count();
    std::unique_ptr<NrBodyContext> bctx_p = std::make_unique<NrBodyContext>(); bctx_p->dev = dev; bctx_p->queue = queue; auto bctx_ref = [&]() -> NrBodyContext& { return *bctx_p; };   // v12 harness: the context is unique/non-movable (runtime v17); recovery creates a NEW instance   // body context is resident across frames (v2: pool/scratch/constants/pipelines)
    NrTemporalState tstate; id<MTLTexture> motion_tex = nil;   // EXPERIMENTAL (v5): shared history state; encoder calls stay reset-only until a temporal Block0 producer exists
    if (temporal) { if (!nr_temporal_state_create(tstate, dev, 1920, 1080, poison)) throw std::runtime_error("temporal state allocation failed");
      if (!motion_file.empty()) { auto mv = Read(motion_file); if (mv.size() != 1920u * 1080u * 4u) throw std::runtime_error("motion file must be 1920x1080 RG16F"); MTLTextureDescriptor* dm = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRG16Float width:1920 height:1080 mipmapped:NO]; dm.usage = MTLTextureUsageShaderRead; dm.storageMode = MTLStorageModeShared; motion_tex = [dev newTextureWithDescriptor:dm]; [motion_tex replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:mv.data() bytesPerRow:1920 * 4]; } }
    int force_reset = -1;   // v6 recovery protocol: -1 = follow --reset-sequence, 0/1 = forced
    auto reset_for = [&](unsigned k) { if (force_reset >= 0) return force_reset == 1; if (reset_sequence.empty()) return true; size_t pos = 0; unsigned idx = 0; while (pos < reset_sequence.size()) { size_t c = reset_sequence.find(',', pos); std::string tok = reset_sequence.substr(pos, c == std::string::npos ? std::string::npos : c - pos); if (idx == k) return tok != "0"; ++idx; if (c == std::string::npos) break; pos = c + 1; } return true; };
    bool all_pass = true; unsigned frame_no = 0;
    std::ostringstream fj;   // v5: per-frame record, kept when the frame throws so that codes/counters/state survive in the receipt
    unsigned context_preparations = 1;   // v7: incremented when the recovery protocol recreates the contexts (contexts_prepared_once reports it truthfully)
    auto do_frame = [&](const std::string& input_path, uint8_t poison, const std::string& reference, const std::string& save_final) -> bool {
      const auto tf0 = std::chrono::steady_clock::now();   // v13: FRAME-LOCAL steady clock (Codex 00:35: v10-v12 measured from the process-start clock, so wall_ms_total was cumulative)
      fj.str(""); fj.clear(); bool pass = false; fj << (frame_no ? ",\n" : "") << "  {\"frame\": " << frame_no << ", \"input_file\": " << J(input_path) << ", \"poison\": " << unsigned(poison) << ", \"diagnostics_dir\": " << J(diag_dir.empty() ? std::string() : diag_dir + "/frame-" + std::to_string(frame_no)) << ",\n";
      // the model input: one RGBA16F 1920x1080 texture, owned here, fed unchanged to BOTH runtimes
      auto img = Read(input_path); if (img.size() != 1920u * 1080u * 8u) throw std::runtime_error("input image must be 1920x1080 RGBA16F");
      MTLTextureDescriptor* d = [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float width:1920 height:1080 mipmapped:NO]; d.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite; d.storageMode = MTLStorageModeShared;
      id<MTLTexture> tex = [dev newTextureWithDescriptor:d]; [tex replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:img.data() bytesPerRow:1920 * 8];
      id<MTLTexture> out_surface = [dev newTextureWithDescriptor:d]; { std::vector<uint8_t> pz(img.size(), poison); [out_surface replaceRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0 withBytes:pz.data() bytesPerRow:1920 * 8]; }
      const std::string input_sha = Sha(img.data(), img.size()); fj << " \"input_sha256\": " << J(input_sha) << ",\n";
      // producer: reset frames use the accepted reset-only entry; a reset=false frame of a temporal sequence uses Codex's experimental temporal entry with the SAME
      // shared state the body consumes (frame_index = frames_since_reset, zero-motion admission, sampler_variant 0); the encoder reads history and never advances the state
      const bool frame_reset = reset_for(frame_no); const bool temporal_producer = temporal && !frame_reset;
      NrEncoderFrame ef; ef.color = tex; ef.reset = !temporal_producer; ef.frame_index = temporal_producer ? tstate.frames_since_reset : 0u; NrEncoderResult er; const auto te0 = std::chrono::steady_clock::now();
#if NR_FULL_FRAME_TEMPORAL_PRODUCER
      NrEncoderTemporalFrame etf; etf.frame = ef; etf.temporal = &tstate; etf.motion = motion_tex; etf.sampler_variant = encoder_sampler_variant;
      const bool enc_diag = temporal_producer && encoder_diag_frame >= 0 && unsigned(encoder_diag_frame) == frame_no && !encoder_diag_dir.empty();   // v8: diagnostics path forwarded on exactly one temporal frame
      // the encoder requires a FRESH directory name (it creates it); only the parent must exist
      const int ec = temporal_producer ? nr_encoder_run_temporal_frame(ectx, etf, poison, er, enc_diag ? encoder_diag_dir.c_str() : nullptr) : nr_encoder_run_frame(ectx, ef, poison, er, nullptr);
#else
      if (temporal_producer) throw std::runtime_error("reset=false frame requested but this harness build has no temporal producer (linked encoder build ships no nr_encoder_temporal_runtime.h); fail closed");
      const int ec = nr_encoder_run_frame(ectx, ef, poison, er, nullptr);
#endif
      const double enc_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - te0).count();
      fj << " \"encoder\": {\"code\": " << ec << ", \"error\": " << J(er.error) << ", \"entry\": " << (temporal_producer ? "\"nr_encoder_run_temporal_frame\"" : "\"nr_encoder_run_frame\"") << ", \"reset\": " << (ef.reset ? "true" : "false") << ", \"frame_index\": " << ef.frame_index << ", \"sampler_variant\": " << (temporal_producer ? int(encoder_sampler_variant) : -1) << ", \"diagnostics_dir\": " << J(enc_diag ? encoder_diag_dir : std::string()) << ", \"bundle_root\": " << J(er.bundle_root) << ", \"input_sha256\": " << J(er.input_sha256) << ", \"feature_sha256\": " << J(er.feature_sha256)
         << ", \"coverage_values_checked\": " << er.coverage_values_checked << ", \"guarded_outputs_checked\": " << er.guarded_outputs_checked << ", \"input_immutable\": " << (er.input_immutable ? "true" : "false") << ", \"gpu_complete\": " << (er.gpu_complete ? "true" : "false") << ", \"prepare_ms\": " << prep_ms << ", \"frame_ms\": " << enc_ms << "},\n";
      if (ec != 0) throw std::runtime_error("encoder frame rejected: " + er.error);
      if (er.input_sha256 != input_sha) throw std::runtime_error("encoder saw a different input than the harness hashed");
      // hand the live regions to the body exactly as returned (buffer, offset, length, sha); nothing is copied through the filesystem
      auto reg = [&](const NrEncoderRegion& r) { NrBodyRegion b; b.buffer = r.buffer; b.offset = r.offset; b.bytes = r.bytes; b.provenance_sha256 = r.sha256; b.producer = std::string(temporal_producer ? "nr_encoder_run_temporal_frame" : "nr_encoder_run_frame") + " bundle " + er.bundle_root; return b; };
      NrBodyFrameInputs in; in.entry = reg(er.entry); for (int i = 0; i < 4; ++i) in.skips[i] = reg(er.skips[i]); in.block0_skip = reg(er.block0_skip); in.input_color = tex; in.output_surface = out_surface; in.poison = poison; in.production = !body_checked;
      if (temporal) { in.temporal = &tstate; in.reset = frame_reset; in.motion = motion_tex; }
      if (const char* f = getenv("NR_BODY_FAULT")) { const char* fr = getenv("NR_BODY_FAULT_FRAME"); if (!fr || unsigned(std::stoul(fr)) == frame_no) in.test_fault = f; }   // fault-injection (test builds only; the production runtime rejects test_fault)
      fj << " \"temporal\": {\"enabled\": " << (temporal ? "true" : "false") << ", \"body_reset\": " << (frame_reset ? "true" : "false") << ", \"encoder_reset\": " << (ef.reset ? "true" : "false") << ", \"frames_since_reset_before\": " << (temporal ? tstate.frames_since_reset : 0u) << ", \"note\": " << (temporal_producer ? "\"reset=false: LIVE temporal producer (history = this process prior native output) feeding the temporal body; final output compared to the reference\"" : "\"reset frame: accepted reset-only producer and body\"") << "},\n";
      fj << " \"regions\": {\n" << Reg("entry", in.entry) << ",\n" << Reg("skip_c256", in.skips[0]) << ",\n" << Reg("skip_c128", in.skips[1]) << ",\n" << Reg("skip_c64", in.skips[2]) << ",\n" << Reg("skip_c32", in.skips[3]) << ",\n" << Reg("block0_skip", in.block0_skip) << "\n },\n";
      // consumer: body runtime (same device/queue), live regions in, caller-owned output surface out
      NrBodyResult br; const auto tb0 = std::chrono::steady_clock::now();
      const std::string fdiag = diag_dir.empty() ? std::string() : diag_dir + "/frame-" + std::to_string(frame_no); const int bc = nr_body_run_frame(bctx_ref(), chain, in, fdiag.empty() ? nullptr : fdiag.c_str(), br); const double body_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - tb0).count();
      fj << " \"body\": {\"code\": " << bc << ", \"error\": " << J(br.error) << ", \"middle_view_sha256\": " << J(br.middle_view_sha256) << ", \"decoder_pre_image_sha256\": " << J(br.decoder_pre_image_sha256) << ", \"pool_final_sha256\": " << J(br.pool_final_sha256)
         << ", \"final_surface_sha256\": " << J(br.final_surface_sha256) << ", \"final_pixels\": " << br.final_pixels << ", \"final_pixels_not_written\": " << br.final_pixels_not_written << ", \"final_pixels_written_more_than_once\": " << br.final_pixels_written_more_than_once
         << ", \"input_texture_sha256_before\": " << J(br.input_texture_sha256_before) << ", \"input_texture_sha256_after\": " << J(br.input_texture_sha256_after) << ", \"input_texture_immutable\": " << (br.input_texture_immutable ? "true" : "false")
         << ", \"middle_gpu_ms\": " << br.middle_gpu_ms << ", \"decoder_gpu_ms\": " << br.decoder_gpu_ms << ", \"wall_ms\": " << body_ms << ", \"checks_skipped\": [";
      for (size_t i = 0; i < br.checks_skipped.size(); ++i) fj << (i ? ", " : "") << J(br.checks_skipped[i]); fj << "], \"checks_enforced\": " << br.checks_enforced << ", \"check_failures\": ["; for (size_t i = 0; i < br.check_failures.size(); ++i) fj << (i ? ", " : "") << J(br.check_failures[i]);
      fj << "], \"constants_rehashed_after_frame\": " << br.constants_rehashed << ", \"constants_unchanged_after_frame\": " << (br.constants_unchanged_after_frame ? "true" : "false") << ", \"temporal_record_sha256\": " << J(br.temporal_record_sha256) << ", \"motion_sha256\": " << J(br.motion_sha256) << ", \"history_sha256_before\": " << J(br.history_sha256_before) << ", \"history_sha256_after\": " << J(br.history_sha256_after) << ", \"frames_since_reset\": " << (temporal ? tstate.frames_since_reset : 0u) << ", \"loaded\": {"; { bool f1 = true; for (auto& kv : br.provenance) { fj << (f1 ? "" : ", ") << J(kv.first) << ": " << J(kv.second); f1 = false; } } fj << "}},\n";
      fj << " \"inventory\": {\"command_buffers\": " << br.inventory_command_buffers << ", \"waits\": " << br.inventory_waits << ", \"blit_encoders\": " << br.inventory_blit_encoders << ", \"compute_encoders\": " << br.inventory_compute_encoders << "},\n";
      fj << " \"body_mode\": " << (body_checked ? "\"checked\"" : "\"production\"") << ",\n \"resident_evidence\": {\"validators_expected\": " << br.validators_expected << ", \"validators_run\": " << br.validators_run << ", \"coverage_faults\": " << br.coverage_faults << ", \"digests_expected\": " << br.digests_expected << ", \"digests_complete\": " << br.digests_complete << ", \"digest_cpu_crosschecks\": " << br.digest_cpu_crosschecks << ", \"digest_cpu_mismatches\": " << br.digest_cpu_mismatches << ", \"digest_table_sha256\": " << J(br.digest_table_sha256) << ", \"frame_gpu_ms\": " << br.frame_gpu_ms << ", \"decoder_pre_image_source\": " << J(br.decoder_pre_image_source) << ", \"decoder_pre_image_expected_sha256\": " << J(br.decoder_pre_image_expected_sha256) << "},\n";   // v10 schema: v12 resident runtime evidence (zeros/empty on v11 runtimes)
      fj << " \"history_copy_value_differences\": " << br.history_copy_value_differences << ",\n \"temporal_state_advanced\": " << (br.temporal_state_advanced ? "true" : "false") << ",\n \"frames_since_reset_after\": " << (temporal ? tstate.frames_since_reset : 0u) << ",\n \"ex2_half_faults\": {\"argument\": " << br.ex2_half_arg_faults << ", \"domain\": " << br.ex2_half_domain_faults << "},\n \"test_fault\": " << J(in.test_fault ? in.test_fault : "") << ",\n";
      if (bc != 0) throw std::runtime_error("body rejected: " + br.error);
      // the caller-owned surface is the product: hash it independently of the runtime's own report
      std::vector<uint8_t> fin(img.size()); [out_surface getBytes:fin.data() bytesPerRow:1920 * 8 fromRegion:MTLRegionMake2D(0, 0, 1920, 1080) mipmapLevel:0]; const std::string fin_sha = Sha(fin.data(), fin.size());
      // producer regions must be unchanged by the body (immutability across the boundary)
      bool regions_immutable = true; auto chk = [&](const NrBodyRegion& r) { if (Sha(static_cast<const uint8_t*>(r.buffer.contents) + r.offset, r.bytes) != r.provenance_sha256) regions_immutable = false; }; chk(in.entry); for (int i = 0; i < 4; ++i) chk(in.skips[i]); chk(in.block0_skip);
      std::string ref_sha; long mism = -1; const bool plumbing_only = false; /* v4: every frame is a real producer frame (reset or LIVE temporal); the v3 plumbing mode is gone */ if (!reference.empty() && !plumbing_only) { auto rf = Read(reference); ref_sha = Sha(rf.data(), rf.size()); if (rf.size() == fin.size()) { mism = 0; for (size_t i = 0; i < fin.size(); i += 2) if (fin[i] != rf[i] || fin[i + 1] != rf[i + 1]) ++mism; } }
      if (!save_final.empty()) std::ofstream(save_final, std::ios::binary).write(reinterpret_cast<const char*>(fin.data()), std::streamsize(fin.size()));
      pass = bc == 0 && br.final_pixels_not_written == 0 && br.final_pixels_written_more_than_once == 0 && br.input_texture_immutable && regions_immutable && fin_sha == br.final_surface_sha256 && (reference.empty() || plumbing_only || (mism == 0 && fin_sha == ref_sha));
      const auto t_end = std::chrono::steady_clock::now(); const double frame_wall_ms = std::chrono::duration<double, std::milli>(t_end - tf0).count(), process_elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t0).count();   // v13 boundary: frame clock starts at do_frame entry (before the encoder call) and ENDS here - after the body completed, the caller-side final readback/hash, the region immutability re-hash and the reference comparison, BEFORE the telemetry copy/serialisation and receipt writing; process_elapsed_ms is the cumulative clock since process start at the same instant
      std::string telemetry_json = "null", telemetry_err;
      if (encoder_telemetry && pass) {
#if NR_FULL_FRAME_TELEMETRY
        std::string tj; if (nr_encoder_copy_telemetry(ectx, tj, telemetry_err) == 0) telemetry_json = tj; else telemetry_json = "null";
#endif
      } else if (encoder_telemetry) telemetry_err = "frame did not pass: telemetry not copied (success-only publication)";
      fj << " \"encoder_telemetry\": " << telemetry_json << ",\n \"encoder_telemetry_error\": " << J(telemetry_err) << ",\n \"encoder_telemetry_enabled\": " << (encoder_telemetry ? "true" : "false") << ",\n";
      fj << " \"no_reference_no_parity_meaning\": " << (reference.empty() ? "true" : "false") << ",\n \"surface_sha256_caller_readback\": " << J(fin_sha) << ",\n \"producer_regions_unchanged_by_body\": " << (regions_immutable ? "true" : "false") << ",\n \"reference\": {\"file\": " << J(reference) << ", \"sha256\": " << J(ref_sha) << ", \"mismatching_values\": " << mism << ", \"cpu_comparison_only\": true},\n"
         << " \"frame_wall_ms\": " << frame_wall_ms << ",\n \"frame_wall_boundary\": \"do_frame entry (before encoder) -> after body completion + caller readback/hash + region re-hash + reference comparison; excludes telemetry copy and receipt writing\",\n \"process_elapsed_ms\": " << process_elapsed_ms << ",\n \"plumbing_only_no_parity_meaning\": " << (plumbing_only ? "true" : "false")
         << ",\n \"status\": " << (reference.empty() ? "\"reference_unavailable\"" : (pass ? "\"comparison_passed\"" : "\"comparison_failed\"")) << ",\n \"result\": " << (pass ? "\"pass\"" : "\"fail\"") << "\n}\n";   // v9 status vocabulary: rejected (frame threw) | reference_unavailable | comparison_passed | comparison_failed; audit_accepted / promoted are never set by the harness
      for (auto& kv : br.provenance) if (kv.first.rfind("lib:", 0) == 0) body_libs[kv.first.substr(4)] = kv.second;
      js << fj.str(); all_pass = all_pass && pass; ++frame_no; return pass;
    };
    js << " \"frames\": [\n";
    try { do_frame(argv[3], poison, reference, !save_final.empty() ? save_final : (save_dir.empty() ? std::string() : save_dir + "/final-0.rgba16le")); } catch (const std::exception& e) { const std::string partial = fj.str(); const bool body_ran = partial.find("\"body\": {") != std::string::npos; if (!partial.empty()) js << partial << " \"status\": " << (body_ran ? "\"execution_failed\"" : "\"rejected\"") << ",\n \"error\": " << J(e.what()) << ",\n \"result\": \"fail\"\n}"; else js << (frame_no ? ",\n" : "") << "  {\"frame\": " << frame_no << ", \"status\": \"rejected\", \"error\": " << J(e.what()) << ", \"result\": \"fail\"}"; all_pass = false; ++frame_no; }
    for (auto& x : extra) { try { do_frame(x.input, uint8_t(std::stoul(x.poison, nullptr, 16) & 0xffu), x.reference, save_dir.empty() ? std::string() : save_dir + "/final-" + std::to_string(frame_no) + ".rgba16le"); } catch (const std::exception& e) { const std::string partial = fj.str(); const bool body_ran = partial.find("\"body\": {") != std::string::npos; if (!partial.empty()) js << partial << " \"status\": " << (body_ran ? "\"execution_failed\"" : "\"rejected\"") << ",\n \"error\": " << J(e.what()) << ",\n \"result\": \"fail\"\n}"; else js << ",\n  {\"frame\": " << frame_no << ", \"status\": \"rejected\", \"error\": " << J(e.what()) << ", \"result\": \"fail\"}"; all_pass = false; ++frame_no; } }
    std::ostringstream rj; std::string recovery_result = "not_requested";
    if (recovery_test && !all_pass) {   // v6: recovery contract after a failed frame - (A) a reset frame on the SAME context/state must be refused; (B) fresh contexts + reset must reproduce the first-frame reference; (C) the fresh state must continue (reset=false) to the frame-1 reference
      // v7: three distinguishable outcomes - "rejected" (admission/runtime exception), "pass" (frame ran and matched its reference), "comparison-fail" (frame ran, reference differs); the expectation names one of them
      auto attempt = [&](const char* label, const std::string& input, uint8_t pz, const std::string& ref, int reset_value, const char* expect) {
        force_reset = reset_value; const unsigned before = frame_no; std::string err; std::string observed;
        try { observed = do_frame(input, pz, ref, save_dir.empty() ? std::string() : save_dir + "/recovery-" + label + ".rgba16le") ? "pass" : "comparison-fail"; }
        catch (const std::exception& e) { err = e.what(); const std::string partial = fj.str(); const bool body_ran = partial.find("\"body\": {") != std::string::npos; const bool admission_refusal = err.find("live input rejected") != std::string::npos; observed = (body_ran && !admission_refusal) ? "execution_failed" : "rejected";   /* v10: a body-level admission refusal (code 2, "live input rejected", e.g. the damaged-context rule of step A) is a rejection; execution_failed = the body ran and failed a check or threw after dispatch (v9 mislabelled step A) */ if (!partial.empty()) js << partial << " \"status\": " << J(observed) << ",\n \"error\": " << J(e.what()) << ",\n \"result\": \"fail\"\n}"; else js << ",\n  {\"frame\": " << frame_no << ", \"error\": " << J(e.what()) << ", \"result\": \"fail\"}"; ++frame_no; }
        force_reset = -1; const bool ok = observed == expect; rj << (rj.str().empty() ? "" : ",\n") << "  {\"step\": " << J(label) << ", \"frame\": " << before << ", \"expected\": " << J(expect) << ", \"observed\": " << J(observed) << ", \"error\": " << J(err) << ", \"ok\": " << (ok ? "true" : "false") << "}"; return ok; };
      const bool a = attempt("A-same-context-reset-must-be-refused", argv[3], poison, reference, 1, "rejected");
      // (B) fresh body context, fresh temporal state, fresh encoder context
      bctx_p = std::make_unique<NrBodyContext>(); bctx_p->dev = dev; bctx_p->queue = queue; tstate = NrTemporalState{}; if (!nr_temporal_state_create(tstate, dev, 1920, 1080, poison)) throw std::runtime_error("temporal state re-allocation failed");
      ectx = NrEncoderContext{}; ectx.device = dev; ectx.queue = queue; if (nr_encoder_prepare(ectx, argv[2], eerr) != 0) throw std::runtime_error("encoder re-prepare rejected: " + eerr); ++context_preparations; enable_telemetry();
      const bool b = attempt("B-fresh-context-reset-must-match-first-reference", argv[3], poison, reference, 1, "pass");
      const bool c = extra.empty() ? true : attempt("C-fresh-state-continues-to-frame-1-reference", extra[0].input, uint8_t(std::stoul(extra[0].poison, nullptr, 16) & 0xffu), extra[0].reference, 0, "pass");
      // (D) wrong-reference control: a SECOND fresh context runs the reset frame against the frame-1 reference (CPU comparison only; never a runtime input) and must report comparison-fail, not pass and not rejection
      bool d = true;
      if (!extra.empty()) { bctx_p = std::make_unique<NrBodyContext>(); bctx_p->dev = dev; bctx_p->queue = queue; tstate = NrTemporalState{}; if (!nr_temporal_state_create(tstate, dev, 1920, 1080, poison)) throw std::runtime_error("temporal state re-allocation failed");
        ectx = NrEncoderContext{}; ectx.device = dev; ectx.queue = queue; if (nr_encoder_prepare(ectx, argv[2], eerr) != 0) throw std::runtime_error("encoder re-prepare rejected: " + eerr); ++context_preparations; enable_telemetry();
        d = attempt("D-wrong-reference-control-must-be-comparison-fail", argv[3], poison, extra[0].reference, 1, "comparison-fail"); }
      recovery_result = (a && b && c && d) ? "pass" : "fail"; }
    // immediate post-run re-hash of the files that defined this run
    bool files_unchanged = true; std::ostringstream pf; pf << " \"post_run_file_identity\": {"; bool p1 = true;
    for (auto& kv : pre_files) { auto v = Read(kv.first); const std::string h = Sha(v.data(), v.size()); const bool same = h == kv.second; files_unchanged = files_unchanged && same; pf << (p1 ? "\n  " : ",\n  ") << J(kv.first) << ": {\"pre\": " << J(kv.second) << ", \"post\": " << J(h) << ", \"unchanged\": " << (same ? "true" : "false") << "}"; p1 = false; }
    for (auto& kv : body_libs) { auto v = Read(kv.first); const std::string h = Sha(v.data(), v.size()); const bool same = h == kv.second; files_unchanged = files_unchanged && same; pf << (p1 ? "\n  " : ",\n  ") << J(kv.first) << ": {\"load\": " << J(kv.second) << ", \"post\": " << J(h) << ", \"unchanged\": " << (same ? "true" : "false") << "}"; p1 = false; }
    pf << "\n },\n \"files_unchanged_after_run\": " << (files_unchanged ? "true" : "false") << ",\n"; all_pass = all_pass && files_unchanged;
    js << "\n ],\n \"status_vocabulary\": \"per frame: rejected | reference_unavailable | comparison_passed | comparison_failed; per run: execution_completed (this receipt exists) and result (all frames comparison_passed or reference_unavailable and every check passed); audit_accepted and promoted are set only by the independent auditor and the owner, never here\",\n \"execution_completed\": true,\n \"audit_accepted\": null,\n \"promoted\": null,\n \"recovery\": {\"requested\": " << (recovery_test ? "true" : "false") << ", \"result\": " << J(recovery_result) << ", \"steps\": [\n" << rj.str() << "\n ]},\n" << pf.str() << " \"frames_in_process\": " << frame_no << ",\n \"contexts_prepared_once\": " << (context_preparations == 1 ? "true" : "false") << ", \"context_preparations\": " << context_preparations << ",\n \"wall_ms_total\": " << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() << ",\n \"result\": " << (all_pass ? "\"pass\"" : "\"fail\"") << "\n}\n";
    const bool pass = all_pass;
    code = pass ? 0 : 1;
  } catch (const std::exception& e) { js << " \"error\": " << J(e.what()) << ",\n \"result\": \"fail\"\n}\n"; code = 2; }
  if (!receipt.empty()) std::ofstream(receipt) << js.str(); else std::cout << js.str();   // the linked encoder runtime prints progress to stdout, so receipts go to a file
  return code;
} }
