// SPDX-License-Identifier: MIT AND Apache-2.0
// Copyright (c) Netal Contributors.
//
// Native Metal DLSS 5 Neural Rendering runtime implementation (src/nr_fast/).
// Portions of the mlxdlss.neural-rendering-transformer.v1 weight schema,
// dynamic topology inference, 12-bit attention-bias fragment swizzle, and
// E4M3 decoding are adapted from:
//   WaveCut / MLX-DLSS Contributors (https://github.com/iamwavecut/MLX-DLSS)
//   Licensed under the Apache License, Version 2.0.

#import "nr_fast_runtime.h"

#import <CommonCrypto/CommonDigest.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <mach-o/dyld.h>
#import <sys/stat.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <sstream>

namespace netal_nr_fast {

namespace {

constexpr size_t kBufferAlign = 256;
constexpr size_t kMaxSafeTensorHeaderBytes = 64ull * 1024ull * 1024ull;
constexpr float kDefaultBlendScale = 0.73974609375f;
constexpr float kSmallBlendScale = 0.76708984375f;

inline size_t AlignUp(size_t value, size_t alignment = kBufferAlign) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Bit-exact host E4M3 <-> FP16 helpers matching mlxdlss-native-e4m3-v1.
uint16_t DecodeE4M3Byte(uint8_t b) {
  uint16_t h_bits = static_cast<uint16_t>(b & 0x7fu) << 7;
  _Float16 conv = 0;
  std::memcpy(&conv, &h_bits, sizeof(uint16_t));
  conv = static_cast<_Float16>(conv * static_cast<_Float16>(256.0f));
  _Float16 signed_val = (b & 0x80u) ? static_cast<_Float16>(-conv) : conv;
  uint16_t out_bits = 0;
  std::memcpy(&out_bits, &signed_val, sizeof(uint16_t));
  return out_bits;
}

const std::array<uint16_t, 256>& GetE4M3Lut() {
  static const std::array<uint16_t, 256> lut = []() {
    std::array<uint16_t, 256> table{};
    for (int i = 0; i < 256; ++i) {
      table[static_cast<size_t>(i)] = DecodeE4M3Byte(static_cast<uint8_t>(i));
    }
    return table;
  }();
  return lut;
}

uint16_t FloatToFp16Bits(float v) {
  _Float16 h = static_cast<_Float16>(v);
  uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(uint16_t));
  return bits;
}

float Fp16BitsToFloat(uint16_t bits) {
  _Float16 h = 0;
  std::memcpy(&h, &bits, sizeof(uint16_t));
  return static_cast<float>(h);
}

uint8_t EncodeFloatToE4M3Byte(float value) {
  uint32_t fp8_max = 543u << 21;
  uint32_t denorm_mask = 141u << 23;
  uint32_t value_bits = 0;
  std::memcpy(&value_bits, &value, sizeof(uint32_t));
  uint32_t sign = value_bits & 0x80000000u;
  value_bits ^= sign;
  uint8_t bits = 0;
  if (value_bits >= fp8_max) {
    bits = 0x7Eu;
  } else if (value_bits < (121u << 23)) {
    float f_val = 0.0f;
    float f_denorm = 0.0f;
    std::memcpy(&f_val, &value_bits, sizeof(float));
    std::memcpy(&f_denorm, &denorm_mask, sizeof(float));
    float sum = f_val + f_denorm;
    std::memcpy(&value_bits, &sum, sizeof(uint32_t));
    bits = static_cast<uint8_t>(value_bits - denorm_mask);
  } else {
    uint8_t mantissa_odd = (value_bits >> 20) & 1u;
    value_bits += (static_cast<uint32_t>(7 - 127) << 23) + 0x7FFFFu;
    value_bits += mantissa_odd;
    bits = static_cast<uint8_t>(value_bits >> 20);
  }
  bits |= static_cast<uint8_t>(sign >> 24);
  return bits;
}

// 12-bit attention-bias fragment swizzle (NeuralRenderingAttentionBiasLayout).
int AttentionBiasSourceIndex(int query, int key) {
  int qy = query / 8;
  int qx = query % 8;
  int ky = key / 8;
  int kx = key % 8;
  auto bit = [](int v, int pos) -> int { return (v >> pos) & 1; };
  return (bit(qy, 2) << 11) | (bit(qx, 2) << 10) | (bit(ky, 2) << 9) | (bit(kx, 2) << 8) |
         (bit(qy, 0) << 7) | (bit(qx, 1) << 6) | (bit(qx, 0) << 5) | (bit(ky, 0) << 4) |
         (bit(kx, 1) << 3) | (bit(ky, 1) << 2) | (bit(qy, 1) << 1) | bit(kx, 0);
}

const std::array<int, 4096>& GetBiasSwizzleIndices() {
  static const std::array<int, 4096> table = []() {
    std::array<int, 4096> idx{};
    for (int q = 0; q < 64; ++q) {
      for (int k = 0; k < 64; ++k) {
        idx[static_cast<size_t>(q * 64 + k)] = AttentionBiasSourceIndex(q, k);
      }
    }
    return idx;
  }();
  return table;
}

bool ComputeFileSha256Hex(const std::string& path, std::string* out_hex, std::string* out_err) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    if (out_err) *out_err = "cannot open file for SHA-256: " + path;
    return false;
  }
  CC_SHA256_CTX ctx;
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
  CC_SHA256_Init(&ctx);
  std::vector<char> chunk(1024 * 1024);
  while (file) {
    file.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
    std::streamsize got = file.gcount();
    if (got > 0) {
      CC_SHA256_Update(&ctx, chunk.data(), static_cast<CC_LONG>(got));
    }
  }
  if (file.bad()) {
    if (out_err) *out_err = "I/O error while hashing file: " + path;
    return false;
  }
  unsigned char digest[CC_SHA256_DIGEST_LENGTH];
  CC_SHA256_Final(digest, &ctx);
#pragma clang diagnostic pop
  static const char* kHex = "0123456789abcdef";
  std::string hex(CC_SHA256_DIGEST_LENGTH * 2, '0');
  for (int i = 0; i < CC_SHA256_DIGEST_LENGTH; ++i) {
    hex[static_cast<size_t>(i * 2)] = kHex[(digest[i] >> 4) & 0xf];
    hex[static_cast<size_t>(i * 2 + 1)] = kHex[digest[i] & 0xf];
  }
  *out_hex = std::move(hex);
  return true;
}

enum class RawDType {
  F16,
  F32,
  FP8_E4M3,
};

struct RawTensorSlice {
  RawDType dtype = RawDType::F16;
  std::vector<int64_t> shape;
  uint64_t byte_begin = 0;
  uint64_t byte_end = 0;
  size_t elem_count = 0;
};

struct ParsedSafeTensors {
  std::map<std::string, RawTensorSlice> tensors;
  std::string metadata_format;
  NSData* file_data = nil;
  size_t payload_offset = 0;
};

bool CheckPathSafety(const std::string& path, bool reject_symlinks, bool expect_dir, std::string* out_err) {
  struct stat lst {};
  if (lstat(path.c_str(), &lst) != 0) {
    if (out_err) *out_err = "path does not exist: " + path;
    return false;
  }
  if (reject_symlinks && S_ISLNK(lst.st_mode)) {
    if (out_err) *out_err = "symlink rejected by policy: " + path;
    return false;
  }
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    if (out_err) *out_err = "cannot stat target path: " + path;
    return false;
  }
  if (expect_dir && !S_ISDIR(st.st_mode)) {
    if (out_err) *out_err = "expected directory: " + path;
    return false;
  }
  if (!expect_dir && !S_ISREG(st.st_mode)) {
    if (out_err) *out_err = "expected regular file: " + path;
    return false;
  }
  return true;
}

bool IsStrictIntegerNSNumber(NSNumber* num) {
  if (!num) return false;
  if (CFGetTypeID((__bridge CFTypeRef)num) == CFBooleanGetTypeID()) return false;
  const char* obj_type = [num objCType];
  if (!obj_type) return false;
  // Reject floating-point numbers ('f', 'd').
  if (obj_type[0] == 'f' || obj_type[0] == 'd') {
    double dv = [num doubleValue];
    if (!std::isfinite(dv) || std::floor(dv) != dv) return false;
  }
  return true;
}

bool ParseSafeTensorsFile(
    const std::string& file_path,
    bool reject_symlinks,
    ParsedSafeTensors* out_parsed,
    SafeTensorSummary* out_summary,
    std::string* out_err) {
  if (!CheckPathSafety(file_path, reject_symlinks, false, out_err)) {
    return false;
  }
  NSError* ns_err = nil;
  NSData* data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:file_path.c_str()]
                                        options:NSDataReadingMappedIfSafe
                                          error:&ns_err];
  if (!data) {
    if (out_err) *out_err = "failed to map safetensors file: " + file_path;
    return false;
  }
  const size_t total_size = [data length];
  if (total_size < 8) {
    if (out_err) *out_err = "truncated safetensors file (< 8 bytes)";
    return false;
  }
  const auto* bytes = static_cast<const uint8_t*>([data bytes]);
  uint64_t header_len = 0;
  std::memcpy(&header_len, bytes, sizeof(uint64_t));  // little-endian on arm64
  if (header_len == 0 || header_len > kMaxSafeTensorHeaderBytes) {
    if (out_err) *out_err = "invalid safetensors header length";
    return false;
  }
  if (header_len > total_size - 8) {
    if (out_err) *out_err = "safetensors header exceeds file size";
    return false;
  }
  const size_t payload_offset = static_cast<size_t>(8 + header_len);
  const size_t payload_size = total_size - payload_offset;

  NSData* hdr_data = [NSData dataWithBytesNoCopy:(void*)(bytes + 8)
                                          length:static_cast<NSUInteger>(header_len)
                                    freeWhenDone:NO];
  // Strict UTF-8 check on header bytes.
  NSString* hdr_utf8 = [[NSString alloc] initWithData:hdr_data encoding:NSUTF8StringEncoding];
  if (!hdr_utf8) {
    if (out_err) *out_err = "safetensors header is not valid UTF-8";
    return false;
  }
  id json_obj = [NSJSONSerialization JSONObjectWithData:hdr_data options:0 error:&ns_err];
  if (!json_obj || ![json_obj isKindOfClass:[NSDictionary class]]) {
    if (out_err) *out_err = "safetensors header is not a valid JSON object";
    return false;
  }

  NSDictionary* root = (NSDictionary*)json_obj;
  std::vector<std::pair<uint64_t, uint64_t>> intervals;
  intervals.reserve([root count]);

  size_t f16_cnt = 0;
  size_t f32_cnt = 0;
  size_t fp8_cnt = 0;
  size_t total_fp16_bytes = 0;

  for (id key_obj in root) {
    if (![key_obj isKindOfClass:[NSString class]]) {
      if (out_err) *out_err = "non-string key in safetensors header";
      return false;
    }
    std::string key = [(NSString*)key_obj UTF8String];
    id val_obj = root[key_obj];
    if (key == "__metadata__") {
      if (![val_obj isKindOfClass:[NSDictionary class]]) {
        if (out_err) *out_err = "__metadata__ must be an object";
        return false;
      }
      id fmt_obj = ((NSDictionary*)val_obj)[@"format"];
      if ([fmt_obj isKindOfClass:[NSString class]]) {
        out_parsed->metadata_format = [(NSString*)fmt_obj UTF8String];
      }
      continue;
    }

    if (![val_obj isKindOfClass:[NSDictionary class]]) {
      if (out_err) *out_err = "tensor entry is not an object: " + key;
      return false;
    }
    NSDictionary* entry = (NSDictionary*)val_obj;
    id dtype_obj = entry[@"dtype"];
    id shape_obj = entry[@"shape"];
    id offsets_obj = entry[@"data_offsets"];
    if (![dtype_obj isKindOfClass:[NSString class]] ||
        ![shape_obj isKindOfClass:[NSArray class]] ||
        ![offsets_obj isKindOfClass:[NSArray class]]) {
      if (out_err) *out_err = "missing/invalid dtype, shape, or data_offsets for " + key;
      return false;
    }

    std::string dtype_str = [(NSString*)dtype_obj UTF8String];
    RawDType dtype;
    size_t elem_bytes = 0;
    if (dtype_str == "F16") {
      dtype = RawDType::F16;
      elem_bytes = 2;
      ++f16_cnt;
    } else if (dtype_str == "F32") {
      dtype = RawDType::F32;
      elem_bytes = 4;
      ++f32_cnt;
    } else if (dtype_str == "F8_E4M3" || dtype_str == "U8") {
      dtype = RawDType::FP8_E4M3;
      elem_bytes = 1;
      ++fp8_cnt;
    } else {
      if (out_err) *out_err = "unsupported safetensors dtype '" + dtype_str + "' on tensor " + key;
      return false;
    }

    NSArray* shape_arr = (NSArray*)shape_obj;
    std::vector<int64_t> shape;
    shape.reserve([shape_arr count]);
    size_t elem_count = 1;
    for (id dim_obj in shape_arr) {
      if (![dim_obj isKindOfClass:[NSNumber class]] || !IsStrictIntegerNSNumber((NSNumber*)dim_obj)) {
        if (out_err) *out_err = "invalid non-integer dimension in shape of " + key;
        return false;
      }
      int64_t d = [(NSNumber*)dim_obj longLongValue];
      if (d <= 0 || d > (1ll << 30)) {
        if (out_err) *out_err = "non-positive or excessive dimension in shape of " + key;
        return false;
      }
      if (elem_count > (std::numeric_limits<size_t>::max() / static_cast<size_t>(d))) {
        if (out_err) *out_err = "shape element count overflow on " + key;
        return false;
      }
      elem_count *= static_cast<size_t>(d);
      shape.push_back(d);
    }

    NSArray* off_arr = (NSArray*)offsets_obj;
    if ([off_arr count] != 2 ||
        ![off_arr[0] isKindOfClass:[NSNumber class]] ||
        ![off_arr[1] isKindOfClass:[NSNumber class]] ||
        !IsStrictIntegerNSNumber((NSNumber*)off_arr[0]) ||
        !IsStrictIntegerNSNumber((NSNumber*)off_arr[1])) {
      if (out_err) *out_err = "invalid data_offsets on " + key;
      return false;
    }
    int64_t b0 = [(NSNumber*)off_arr[0] longLongValue];
    int64_t b1 = [(NSNumber*)off_arr[1] longLongValue];
    if (b0 < 0 || b1 < b0) {
      if (out_err) *out_err = "negative or inverted data_offsets on " + key;
      return false;
    }
    uint64_t ub0 = static_cast<uint64_t>(b0);
    uint64_t ub1 = static_cast<uint64_t>(b1);
    if (ub1 > payload_size) {
      if (out_err) *out_err = "tensor data_offsets exceed file payload on " + key;
      return false;
    }
    uint64_t expected_bytes = static_cast<uint64_t>(elem_count) * static_cast<uint64_t>(elem_bytes);
    if (ub1 - ub0 != expected_bytes) {
      if (out_err) *out_err = "tensor byte span does not match shape*dtype on " + key;
      return false;
    }
    if (ub1 > ub0) {
      intervals.emplace_back(ub0, ub1);
    }

    total_fp16_bytes += elem_count * sizeof(uint16_t);
    out_parsed->tensors.emplace(
        std::move(key),
        RawTensorSlice{dtype, std::move(shape), ub0, ub1, elem_count});
  }

  if (out_parsed->tensors.empty()) {
    if (out_err) *out_err = "safetensors file contains no tensors";
    return false;
  }

  std::sort(intervals.begin(), intervals.end());
  for (size_t i = 1; i < intervals.size(); ++i) {
    if (intervals[i].first < intervals[i - 1].second) {
      if (out_err) *out_err = "overlapping tensor byte slices in safetensors payload";
      return false;
    }
  }

  out_parsed->file_data = data;
  out_parsed->payload_offset = payload_offset;
  if (out_summary) {
    out_summary->tensor_count = out_parsed->tensors.size();
    out_summary->f16_count = f16_cnt;
    out_summary->f32_count = f32_cnt;
    out_summary->fp8_e4m3_count = fp8_cnt;
    out_summary->total_weight_bytes_fp16 = total_fp16_bytes;
    out_summary->source_format = !out_parsed->metadata_format.empty()
                                     ? out_parsed->metadata_format
                                     : (fp8_cnt > 0 ? "fp8-e4m3" : "safetensors-fp16");
  }
  return true;
}

bool LoadAndValidatePackageOrSafeTensors(
    const std::string& path,
    bool reject_symlinks,
    bool verify_sha256,
    ParsedSafeTensors* out_parsed,
    SafeTensorSummary* out_summary,
    std::string* out_err) {
  struct stat st {};
  if (stat(path.c_str(), &st) != 0) {
    if (out_err) *out_err = "model path not found: " + path;
    return false;
  }
  if (S_ISDIR(st.st_mode)) {
    if (!CheckPathSafety(path, reject_symlinks, true, out_err)) return false;
    std::string manifest_path = path + "/manifest.json";
    if (!CheckPathSafety(manifest_path, reject_symlinks, false, out_err)) return false;

    NSData* mdata = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:manifest_path.c_str()]];
    if (!mdata) {
      if (out_err) *out_err = "cannot read manifest.json in " + path;
      return false;
    }
    NSError* jerr = nil;
    id jobj = [NSJSONSerialization JSONObjectWithData:mdata options:0 error:&jerr];
    if (!jobj || ![jobj isKindOfClass:[NSDictionary class]]) {
      if (out_err) *out_err = "invalid JSON in manifest.json";
      return false;
    }
    NSDictionary* manifest = (NSDictionary*)jobj;
    id arch_obj = manifest[@"architecture"];
    if (![arch_obj isKindOfClass:[NSString class]]) {
      if (out_err) *out_err = "manifest.json missing architecture";
      return false;
    }
    std::string arch = [(NSString*)arch_obj UTF8String];
    if (arch != "mlxdlss.neural-rendering-transformer.v1" &&
        arch != "nrk.neural-rendering-transformer.v1") {
      if (out_err) *out_err = "unsupported model architecture: " + arch;
      return false;
    }

    id weights_obj = manifest[@"weights"];
    if (![weights_obj isKindOfClass:[NSDictionary class]]) {
      if (out_err) *out_err = "manifest.json missing weights dictionary";
      return false;
    }
    NSDictionary* weights_dict = (NSDictionary*)weights_obj;
    id file_obj = weights_dict[@"file"];
    if (![file_obj isKindOfClass:[NSString class]]) {
      if (out_err) *out_err = "manifest.json missing weights.file";
      return false;
    }
    std::string rel_file = [(NSString*)file_obj UTF8String];
    if (rel_file.empty() || rel_file[0] == '/' || rel_file.find("..") != std::string::npos) {
      if (out_err) *out_err = "unsafe weights.file in manifest.json: " + rel_file;
      return false;
    }
    std::string weights_path = path + "/" + rel_file;
    if (verify_sha256 && [weights_dict[@"sha256"] isKindOfClass:[NSString class]]) {
      std::string expected_sha = [(NSString*)weights_dict[@"sha256"] UTF8String];
      std::transform(expected_sha.begin(), expected_sha.end(), expected_sha.begin(), ::tolower);
      std::string actual_sha;
      if (!ComputeFileSha256Hex(weights_path, &actual_sha, out_err)) {
        return false;
      }
      if (actual_sha != expected_sha) {
        if (out_err) {
          *out_err = "weights SHA-256 mismatch: expected " + expected_sha + ", got " + actual_sha;
        }
        return false;
      }
    }

    if (!ParseSafeTensorsFile(weights_path, reject_symlinks, out_parsed, out_summary, out_err)) {
      return false;
    }

    // Verify manifest tensor specs if present.
    if ([weights_dict[@"tensors"] isKindOfClass:[NSArray class]]) {
      NSArray* specs = (NSArray*)weights_dict[@"tensors"];
      for (id s_obj in specs) {
        if (![s_obj isKindOfClass:[NSDictionary class]]) continue;
        NSDictionary* spec = (NSDictionary*)s_obj;
        if (![spec[@"name"] isKindOfClass:[NSString class]] ||
            ![spec[@"shape"] isKindOfClass:[NSArray class]]) {
          if (out_err) *out_err = "malformed tensor spec in manifest.json";
          return false;
        }
        std::string tname = [(NSString*)spec[@"name"] UTF8String];
        auto it = out_parsed->tensors.find(tname);
        if (it == out_parsed->tensors.end()) {
          if (out_err) *out_err = "missing manifest tensor in safetensors: " + tname;
          return false;
        }
        NSArray* exp_shape = (NSArray*)spec[@"shape"];
        if (it->second.shape.size() != [exp_shape count]) {
          if (out_err) *out_err = "rank mismatch against manifest for " + tname;
          return false;
        }
        for (NSUInteger i = 0; i < [exp_shape count]; ++i) {
          if (it->second.shape[i] != [exp_shape[i] longLongValue]) {
            if (out_err) *out_err = "shape mismatch against manifest for " + tname;
            return false;
          }
        }
      }
    }
    if (out_summary) {
      out_summary->source_format = "dlssmodel-package";
    }
    return true;
  }

  return ParseSafeTensorsFile(path, reject_symlinks, out_parsed, out_summary, out_err);
}

// ============================================================================
// Topology Inference & Block Metadata
// ============================================================================

enum class BlockFamily {
  Pre,
  Window,
  Downsample,
  SplitWindow,
  SplitDownsample,
  GlobalAttention,
  Bridge,
  Upsample,
  Post,
};

struct InferredFamily {
  BlockFamily family;
  int channels;
  bool operator==(const InferredFamily& o) const {
    return family == o.family && channels == o.channels;
  }
  bool operator!=(const InferredFamily& o) const { return !(*this == o); }
};

TopologySpec MakeTopologySpec(const int depths[5], int global_blocks, float blend_scale) {
  TopologySpec spec;
  int sum_d = 0;
  for (int i = 0; i < 5; ++i) {
    spec.depths[i] = depths[i];
    sum_d += depths[i];
  }
  spec.global_block_count = global_blocks;
  spec.total_blocks = 3 + global_blocks + 2 * sum_d;
  spec.blend_scale = blend_scale;

  const bool is_ht = (global_blocks == 8 && depths[0] == 4 && depths[1] == 4 &&
                      depths[2] == 6 && depths[3] == 8 && depths[4] == 8);
  const bool is_small = (global_blocks == 8 && depths[0] == 2 && depths[1] == 2 &&
                         depths[2] == 2 && depths[3] == 2 && depths[4] == 2);
  if (is_ht) {
    spec.kind = TopologyKind::HT71;
    spec.name = "ht";
  } else if (is_small) {
    spec.kind = TopologyKind::Small31;
    spec.name = "small";
  } else {
    spec.kind = TopologyKind::Custom;
    spec.name = "custom";
  }
  std::ostringstream oss;
  oss << depths[0] << "," << depths[1] << "," << depths[2] << "," << depths[3] << "," << depths[4];
  if (global_blocks != 8) oss << "/" << global_blocks;
  spec.spec_string = oss.str();
  return spec;
}

std::vector<InferredFamily> BuildExpectedFamilies(const TopologySpec& spec) {
  std::vector<InferredFamily> res;
  res.reserve(static_cast<size_t>(spec.total_blocks));
  res.push_back({BlockFamily::Pre, 32});
  const int enc_ch[4] = {32, 64, 128, 256};
  for (int s = 0; s < 4; ++s) {
    for (int i = 0; i < spec.depths[s]; ++i) {
      bool last = (i == spec.depths[s] - 1);
      res.push_back({last ? BlockFamily::Downsample : BlockFamily::Window, enc_ch[s]});
    }
  }
  for (int i = 0; i < spec.depths[4]; ++i) {
    bool last = (i == spec.depths[4] - 1);
    res.push_back({last ? BlockFamily::SplitDownsample : BlockFamily::SplitWindow, 512});
  }
  for (int i = 0; i < spec.global_block_count; ++i) {
    res.push_back({BlockFamily::GlobalAttention, 1024});
  }
  res.push_back({BlockFamily::Bridge, 1024});
  for (int i = 0; i < spec.depths[4]; ++i) {
    res.push_back({BlockFamily::SplitWindow, 512});
  }
  for (int s = 3; s >= 0; --s) {
    for (int j = 0; j < spec.depths[s]; ++j) {
      res.push_back({j == 0 ? BlockFamily::Upsample : BlockFamily::Window, enc_ch[s]});
    }
  }
  res.push_back({BlockFamily::Post, 32});
  return res;
}

bool InferTopologyFromTensors(
    const ParsedSafeTensors& parsed,
    TopologySpec* out_spec,
    std::string* out_err) {
  int max_block = -1;
  for (const auto& kv : parsed.tensors) {
    const std::string& k = kv.first;
    if (k.compare(0, 5, "block") != 0) continue;
    size_t dot = k.find('.');
    if (dot == std::string::npos || dot <= 5) continue;
    std::string num_str = k.substr(5, dot - 5);
    bool digits = !num_str.empty() && std::all_of(num_str.begin(), num_str.end(), ::isdigit);
    if (!digits) continue;
    int idx = std::stoi(num_str);
    max_block = std::max(max_block, idx);
  }
  if (max_block < 10) {
    if (out_err) *out_err = "no valid blockN tensors found for neural-rendering topology";
    return false;
  }

  auto shape_of = [&](const std::string& name) -> const std::vector<int64_t>* {
    auto it = parsed.tensors.find(name);
    return (it == parsed.tensors.end()) ? nullptr : &it->second.shape;
  };

  std::vector<InferredFamily> families(static_cast<size_t>(max_block + 1));
  for (int b = 0; b <= max_block; ++b) {
    std::string p0 = "block" + std::to_string(b) + ".layer0.";
    if (shape_of(p0 + "first_projection_weight") != nullptr) {
      bool has_l4 = shape_of("block" + std::to_string(b) + ".layer4.weight") != nullptr;
      families[static_cast<size_t>(b)] = {
          has_l4 ? BlockFamily::SplitDownsample : BlockFamily::SplitWindow, 512};
      continue;
    }
    const auto* w_glob = shape_of(p0 + "weight");
    if (w_glob && w_glob->size() == 2 && (*w_glob)[0] == 1024 && (*w_glob)[1] == 4096) {
      families[static_cast<size_t>(b)] = {BlockFamily::GlobalAttention, 1024};
      continue;
    }
    if (shape_of(p0 + "conv_weight") != nullptr) {
      families[static_cast<size_t>(b)] = {BlockFamily::Bridge, 1024};
      continue;
    }
    const auto* qkv = shape_of(p0 + "qkv_weight");
    if (!qkv || qkv->empty()) {
      if (out_err) *out_err = "unrecognized block family at block" + std::to_string(b);
      return false;
    }
    int ch = static_cast<int>((*qkv)[0]);
    if (shape_of(p0 + "input_adapter_weight") != nullptr) {
      families[static_cast<size_t>(b)] = {BlockFamily::Pre, ch};
    } else if (shape_of(p0 + "inp_merge_sin") != nullptr) {
      families[static_cast<size_t>(b)] = {BlockFamily::Post, ch};
    } else if (shape_of(p0 + "sin") != nullptr) {
      families[static_cast<size_t>(b)] = {BlockFamily::Upsample, ch};
    } else {
      const auto* w0 = shape_of(p0 + "weight0");
      if (w0 && w0->size() == 2 && (*w0)[0] == ch && (*w0)[1] == 2 * ch) {
        families[static_cast<size_t>(b)] = {BlockFamily::Downsample, ch};
      } else {
        families[static_cast<size_t>(b)] = {BlockFamily::Window, ch};
      }
    }
  }

  if (families[0] != InferredFamily{BlockFamily::Pre, 32}) {
    if (out_err) *out_err = "block0 is not a 32-channel Pre block";
    return false;
  }

  size_t pos = 1;
  auto take = [&](BlockFamily fam, int ch) -> int {
    size_t start = pos;
    while (pos < families.size() && families[pos].family == fam &&
           (ch < 0 || families[pos].channels == ch)) {
      ++pos;
    }
    return static_cast<int>(pos - start);
  };

  int depths[5] = {0, 0, 0, 0, 0};
  const int enc_ch[4] = {32, 64, 128, 256};
  for (int s = 0; s < 4; ++s) {
    int reg = take(BlockFamily::Window, enc_ch[s]);
    if (take(BlockFamily::Downsample, enc_ch[s]) != 1) {
      if (out_err) *out_err = "missing single downsample block in encoder stage " + std::to_string(s);
      return false;
    }
    depths[s] = reg + 1;
  }
  int sp_reg = take(BlockFamily::SplitWindow, 512);
  if (take(BlockFamily::SplitDownsample, 512) != 1) {
    if (out_err) *out_err = "missing split downsample transition block";
    return false;
  }
  depths[4] = sp_reg + 1;

  int glob_cnt = take(BlockFamily::GlobalAttention, 1024);
  if (glob_cnt < 1 || take(BlockFamily::Bridge, 1024) != 1) {
    if (out_err) *out_err = "missing global stage or bridge block";
    return false;
  }

  // Read blend_scale from post block if present.
  float blend_scale = kDefaultBlendScale;
  int post_idx = 3 + glob_cnt + 2 * (depths[0] + depths[1] + depths[2] + depths[3] + depths[4]) - 1;
  std::string bs_name = "block" + std::to_string(post_idx) + ".layer0.blend_scale";
  auto bs_it = parsed.tensors.find(bs_name);
  if (bs_it != parsed.tensors.end() && bs_it->second.elem_count == 1) {
    const uint8_t* ptr =
        static_cast<const uint8_t*>([parsed.file_data bytes]) + parsed.payload_offset + bs_it->second.byte_begin;
    if (bs_it->second.dtype == RawDType::F16) {
      uint16_t u16 = 0;
      std::memcpy(&u16, ptr, 2);
      blend_scale = Fp16BitsToFloat(u16);
    } else if (bs_it->second.dtype == RawDType::F32) {
      std::memcpy(&blend_scale, ptr, 4);
    } else {
      blend_scale = Fp16BitsToFloat(GetE4M3Lut()[ptr[0]]);
    }
  }

  TopologySpec spec = MakeTopologySpec(depths, glob_cnt, blend_scale);
  std::vector<InferredFamily> expected = BuildExpectedFamilies(spec);
  if (families != expected) {
    if (out_err) *out_err = "block sequence does not match mirrored topology " + spec.spec_string;
    return false;
  }
  *out_spec = std::move(spec);
  return true;
}

// Window origin (pad_top, pad_left) for a block index under `spec`.
void GetWindowPadding(const TopologySpec& spec, int block_idx, uint32_t* pad_top, uint32_t* pad_left) {
  auto phase_to_pad = [&](int phase) {
    switch (phase & 3) {
      case 0: *pad_top = 0; *pad_left = 0; break;
      case 1: *pad_top = 4; *pad_left = 4; break;
      case 2: *pad_top = 0; *pad_left = 4; break;
      case 3: *pad_top = 4; *pad_left = 0; break;
    }
  };
  if (block_idx == 0) {
    phase_to_pad(0);
    return;
  }
  if (block_idx == spec.total_blocks - 1) {
    phase_to_pad(1);  // postPhase = 1 -> (-4, -4)
    return;
  }
  int enc_start = 1;
  for (int s = 0; s < 4; ++s) {
    int enc_end = enc_start + spec.depths[s] - 1;
    if (block_idx >= enc_start && block_idx <= enc_end) {
      phase_to_pad(block_idx - enc_start);
      return;
    }
    enc_start = enc_end + 1;
  }
  int sp_enc_start = enc_start;
  int sp_enc_end = sp_enc_start + spec.depths[4] - 1;
  if (block_idx >= sp_enc_start && block_idx <= sp_enc_end) {
    phase_to_pad(block_idx - sp_enc_start);
    return;
  }
  int bridge_idx = sp_enc_end + spec.global_block_count + 1;
  int sp_dec_start = bridge_idx + 1;
  int sp_dec_end = sp_dec_start + spec.depths[4] - 1;
  if (block_idx >= sp_dec_start && block_idx <= sp_dec_end) {
    phase_to_pad(spec.depths[4] + block_idx - sp_dec_start);
    return;
  }
  int dec_start = sp_dec_end + 1;
  for (int s = 3; s >= 0; --s) {
    int dec_end = dec_start + spec.depths[s] - 1;
    if (block_idx >= dec_start && block_idx <= dec_end) {
      phase_to_pad(spec.depths[s] + block_idx - dec_start);
      return;
    }
    dec_start = dec_end + 1;
  }
  *pad_top = 0;
  *pad_left = 0;
}

// ============================================================================
// GPU Parameter Mirror Structs (matching nr_fast_kernels.metal)
// ============================================================================

struct KCountParams {
  uint32_t count;
  uint32_t aux0;
  uint32_t aux1;
  uint32_t aux2;
};

struct KPool2PadParams {
  uint32_t in_height;
  uint32_t in_width;
  uint32_t out_height;
  uint32_t out_width;
  uint32_t channels;
  uint32_t publish;
};

struct KUpsampleMergeParams {
  uint32_t height;
  uint32_t width;
  uint32_t channels;
  uint32_t low_width;
  uint32_t has_low_scale;
  uint32_t publish;
};

struct KWindow1H32Params {
  uint32_t height;
  uint32_t width;
  uint32_t pad_top;
  uint32_t pad_left;
  uint32_t publish;
  uint32_t low_width;
  uint32_t merge_flags;
  uint32_t emit_full;
};

struct KWindowMHParams {
  uint32_t height;
  uint32_t width;
  uint32_t pad_top;
  uint32_t pad_left;
  uint32_t head_count;
  uint32_t publish;
};

struct KGemmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t epilogue;
};

struct KGroupedGemmParams {
  uint32_t M;
  uint32_t N;
  uint32_t K;
  uint32_t groups;
  uint32_t epilogue;
};

struct KGlobalAttnParams {
  uint32_t tokens;
  uint32_t pitch;
  uint32_t tiles;
  uint32_t groups;
  uint32_t heads;
};

struct KPreFeatureParams {
  uint32_t in_width;
  uint32_t in_height;
  uint32_t net_width;
  uint32_t net_height;
  uint32_t logical_width;
  uint32_t logical_height;
  uint32_t frame_index;
  uint32_t scaled_network;
  uint32_t has_history;
  uint32_t has_motion;
  uint32_t has_depth;
  uint32_t depth_inverted;
  uint32_t srgb_transfer;
  float normalized_style;
  float local_tone;
  float local_structure;
  float motion_scale_x;
  float motion_scale_y;
};

struct KPostTemporalParams {
  uint32_t net_width;
  uint32_t net_height;
  uint32_t hist_width;
  uint32_t hist_height;
  uint32_t out_width;
  uint32_t out_height;
  uint32_t scaled_network;
  uint32_t has_history;
  uint32_t direct_write_output;
  uint32_t srgb_transfer;
  float blend_scale;
  float intensity;
};

struct KDetailComposeParams {
  uint32_t net_width;
  uint32_t net_height;
  uint32_t out_width;
  uint32_t out_height;
  uint32_t radius;
  uint32_t scaled_network;
  uint32_t srgb_transfer;
  float detail_strength;
  float colour_strength;
  float intensity;
};

enum : uint32_t {
  EPILOGUE_NONE = 0u,
  EPILOGUE_E4M3 = 1u,
  EPILOGUE_QUAD_GATE = 2u,
  EPILOGUE_QUAD_GATE_E4M3 = 3u,
  EPILOGUE_COS_RES = 4u,
  EPILOGUE_COS_RES_E4M3 = 5u,
};

// ============================================================================
// Prepared Per-Block Byte Offsets inside `weights_buf_`
// ============================================================================

struct PreparedBlockOffsets {
  BlockFamily family = BlockFamily::Window;
  int channels = 32;
  int heads = 1;
  uint32_t pad_top = 0;
  uint32_t pad_left = 0;

  // Common window/attention offsets:
  size_t adapter_off = 0;         // [16, 32] (Pre)
  size_t w0_off = 0;              // downsample [C, 2C] or upsample [2C, C] or bridge [1024, 512]
  size_t sin_off = 0;             // upsample sin [C] or bridge inp_upsample_sin [512] or post inp_merge_sin [32]
  size_t cos_merge_off = 0;       // post inp_merge_cos [32]
  size_t head_weight_off = 0;     // post packed [32, 8]

  // FFN:
  size_t ffn_w1_off = 0;          // 1h32 [32, 128], branched dense [C, 4C], split first_proj [512, 512], global [1024, 4096]
  size_t ffn_w2_off = 0;          // 1h32 [128, 32], branched grouped [G, 128, 32], split group_exp [8, 64, 256], global [4096, 1024]
  size_t ffn_w3_off = 0;          // branched out_proj [C, C], split group_proj [8, 256, 64]
  size_t ffn_w4_off = 0;          // split weight3 [512, 512]
  size_t ffn_cos_off = 0;         // [C]

  // Attention:
  size_t qkv_off = 0;             // [C, 3C]
  size_t attn_scale_off = 0;      // [heads] (FP16; global blocks pre-multiplied by sqrt(32))
  size_t attn_bias_off = 0;       // [heads * 64 * 64] (swizzled iff heads == 1 || heads == 16)
  size_t attn_proj_off = 0;       // [C, C]
  size_t attn_cos_off = 0;        // [C]
};

struct TensorProvider {
  virtual ~TensorProvider() = default;
  virtual bool FetchFp16(
      const std::string& name,
      const std::vector<int64_t>& expected_shape,
      std::vector<uint16_t>* out_fp16,
      float scale_factor,
      std::string* out_err) = 0;
};

class SafeTensorProvider final : public TensorProvider {
 public:
  explicit SafeTensorProvider(const ParsedSafeTensors& parsed) : parsed_(parsed) {}

  bool FetchFp16(
      const std::string& name,
      const std::vector<int64_t>& expected_shape,
      std::vector<uint16_t>* out_fp16,
      float scale_factor,
      std::string* out_err) override {
    auto it = parsed_.tensors.find(name);
    if (it == parsed_.tensors.end()) {
      if (out_err) *out_err = "missing required weight tensor: " + name;
      return false;
    }
    const RawTensorSlice& s = it->second;
    if (s.shape != expected_shape) {
      if (out_err) *out_err = "unexpected shape on weight tensor: " + name;
      return false;
    }
    out_fp16->resize(s.elem_count);
    const uint8_t* src =
        static_cast<const uint8_t*>([parsed_.file_data bytes]) + parsed_.payload_offset + s.byte_begin;
    const bool apply_scale = (scale_factor != 1.0f);

    if (s.dtype == RawDType::F16 && !apply_scale) {
      std::memcpy(out_fp16->data(), src, s.elem_count * sizeof(uint16_t));
    } else if (s.dtype == RawDType::F16) {
      for (size_t i = 0; i < s.elem_count; ++i) {
        uint16_t bits = 0;
        std::memcpy(&bits, src + i * 2, 2);
        (*out_fp16)[i] = FloatToFp16Bits(Fp16BitsToFloat(bits) * scale_factor);
      }
    } else if (s.dtype == RawDType::F32) {
      for (size_t i = 0; i < s.elem_count; ++i) {
        float fv = 0.0f;
        std::memcpy(&fv, src + i * 4, 4);
        if (!std::isfinite(fv)) {
          if (out_err) *out_err = "non-finite float32 weight in " + name;
          return false;
        }
        (*out_fp16)[i] = FloatToFp16Bits(fv * scale_factor);
      }
    } else {
      const auto& lut = GetE4M3Lut();
      if (!apply_scale) {
        for (size_t i = 0; i < s.elem_count; ++i) {
          (*out_fp16)[i] = lut[src[i]];
        }
      } else {
        for (size_t i = 0; i < s.elem_count; ++i) {
          (*out_fp16)[i] = FloatToFp16Bits(Fp16BitsToFloat(lut[src[i]]) * scale_factor);
        }
      }
    }
    return true;
  }

 private:
  const ParsedSafeTensors& parsed_;
};

class SyntheticTensorProvider final : public TensorProvider {
 public:
  SyntheticTensorProvider(uint64_t seed, float weight_scale)
      : base_seed_(seed), weight_scale_(weight_scale) {}

  bool FetchFp16(
      const std::string& name,
      const std::vector<int64_t>& expected_shape,
      std::vector<uint16_t>* out_fp16,
      float scale_factor,
      std::string*) override {
    size_t n = 1;
    for (int64_t d : expected_shape) n *= static_cast<size_t>(d);
    out_fp16->resize(n);

    // Hash tensor name + base_seed_ with SplitMix64 so every weight tensor is
    // deterministic, bounded, and well-conditioned across 31/71 blocks.
    uint64_t state = base_seed_ ^ 0x9E3779B97F4A7C15ULL;
    for (unsigned char c : name) {
      state ^= static_cast<uint64_t>(c);
      state *= 0xBF58476D1CE4E5B9ULL;
    }
    auto next_u64 = [&]() -> uint64_t {
      state += 0x9E3779B97F4A7C15ULL;
      uint64_t z = state;
      z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
      z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
      return z ^ (z >> 31);
    };
    auto next_sym = [&]() -> float {
      double u = static_cast<double>(next_u64() >> 11) * (1.0 / 9007199254740992.0);
      return static_cast<float>(u * 2.0 - 1.0);
    };

    const bool is_cos_skip = (name.find("cos_skip") != std::string::npos ||
                              name.find("inp_merge_cos") != std::string::npos);
    const bool is_sin = (name.find("sin") != std::string::npos);
    const bool is_attn_scale = (name.find("attn_scale") != std::string::npos);
    const bool is_attn_bias = (name.find("attn_bias") != std::string::npos);
    const bool is_head_out =
        (name.find("out_gain") != std::string::npos || name.find("out_conv_weight") != std::string::npos);

    int64_t fan_in = expected_shape.empty() ? 1 : expected_shape[0];
    if (expected_shape.size() == 5) {
      fan_in = expected_shape[2] * expected_shape[3];
    } else if (expected_shape.size() == 4) {
      fan_in = expected_shape[1] * expected_shape[2];
    } else if (expected_shape.size() == 3) {
      fan_in = expected_shape[1];
    }
    float std_scale = weight_scale_ / std::sqrt(static_cast<float>(std::max<int64_t>(1, fan_in)));

    for (size_t i = 0; i < n; ++i) {
      float val = 0.0f;
      if (is_cos_skip) {
        val = 0.85f + 0.05f * next_sym();
      } else if (is_sin) {
        val = 0.25f + 0.05f * next_sym();
      } else if (is_attn_scale) {
        val = (0.75f + 0.1f * next_sym()) * scale_factor;
      } else if (is_attn_bias) {
        val = 0.05f * next_sym();
      } else if (is_head_out) {
        val = next_sym() * 0.45f * scale_factor;
      } else {
        val = next_sym() * std_scale * scale_factor;
      }
      // Quantize synthetic matrix weights to E4M3 grid just like shipping checkpoints.
      uint8_t e4m3 = EncodeFloatToE4M3Byte(val);
      (*out_fp16)[i] = is_attn_bias || is_cos_skip || is_sin || is_attn_scale
                           ? FloatToFp16Bits(val)
                           : GetE4M3Lut()[e4m3];
    }
    return true;
  }

 private:
  uint64_t base_seed_;
  float weight_scale_;
};

class WeightPacker {
 public:
  size_t AppendAligned(const std::vector<uint16_t>& words) {
    size_t byte_off = AlignUp(bytes_.size(), kBufferAlign);
    bytes_.resize(byte_off + words.size() * sizeof(uint16_t), 0);
    std::memcpy(bytes_.data() + byte_off, words.data(), words.size() * sizeof(uint16_t));
    return byte_off;
  }

  const std::vector<uint8_t>& bytes() const { return bytes_; }

 private:
  std::vector<uint8_t> bytes_;
};

bool PrepareModelWeights(
    const TopologySpec& spec,
    TensorProvider& provider,
    std::vector<PreparedBlockOffsets>* out_blocks,
    std::vector<uint8_t>* out_packed_bytes,
    std::string* out_err) {
  std::vector<InferredFamily> families = BuildExpectedFamilies(spec);
  out_blocks->assign(families.size(), PreparedBlockOffsets{});
  WeightPacker packer;
  std::vector<uint16_t> raw;

  auto fetch_pack = [&](const std::string& name,
                        const std::vector<int64_t>& shape,
                        size_t* dst_off,
                        float scale = 1.0f) -> bool {
    if (!provider.FetchFp16(name, shape, &raw, scale, out_err)) return false;
    *dst_off = packer.AppendAligned(raw);
    return true;
  };

  auto fetch_bias_pack = [&](const std::string& name, int heads, size_t* dst_off) -> bool {
    if (!provider.FetchFp16(name, {heads, 64, 64}, &raw, 1.0f, out_err)) return false;
    // Apply 12-bit fragment swizzle ONLY when headCount == 1 or headCount == 16.
    if (heads == 1 || heads == 16) {
      const auto& swz = GetBiasSwizzleIndices();
      std::vector<uint16_t> remapped(raw.size());
      for (int h = 0; h < heads; ++h) {
        const uint16_t* src_h = raw.data() + static_cast<size_t>(h) * 4096u;
        uint16_t* dst_h = remapped.data() + static_cast<size_t>(h) * 4096u;
        for (size_t idx = 0; idx < 4096u; ++idx) {
          dst_h[idx] = src_h[static_cast<size_t>(swz[idx])];
        }
      }
      *dst_off = packer.AppendAligned(remapped);
    } else {
      *dst_off = packer.AppendAligned(raw);
    }
    return true;
  };

  for (size_t b = 0; b < families.size(); ++b) {
    PreparedBlockOffsets& blk = (*out_blocks)[b];
    blk.family = families[b].family;
    blk.channels = families[b].channels;
    blk.heads = blk.channels / 32;
    GetWindowPadding(spec, static_cast<int>(b), &blk.pad_top, &blk.pad_left);

    const std::string bstr = "block" + std::to_string(b);
    const std::string l0 = bstr + ".layer0.";

    if (blk.family == BlockFamily::Bridge) {
      if (!fetch_pack(l0 + "conv_weight", {1024, 512}, &blk.w0_off)) return false;
      if (!fetch_pack(l0 + "inp_upsample_sin", {512}, &blk.sin_off)) return false;
      continue;
    }

    if (blk.family == BlockFamily::GlobalAttention) {
      if (!fetch_pack(bstr + ".layer0.weight", {1024, 4096}, &blk.ffn_w1_off)) return false;
      if (!fetch_pack(bstr + ".layer1.weight", {4096, 1024}, &blk.ffn_w2_off)) return false;
      if (!fetch_pack(bstr + ".layer1.ffn_cos_skip", {1024}, &blk.ffn_cos_off)) return false;
      if (!fetch_pack(bstr + ".layer2.qkv_weight", {1024, 3072}, &blk.qkv_off)) return false;
      // Global blocks pre-multiply attn_scale by sqrt(1024 / 32) = sqrt(32).
      const float glob_scale = std::sqrt(32.0f);
      if (!fetch_pack(bstr + ".layer2.attn_scale", {32}, &blk.attn_scale_off, glob_scale)) return false;
      if (!fetch_pack(bstr + ".layer4.projection_weight", {1024, 1024}, &blk.attn_proj_off)) return false;
      if (!fetch_pack(bstr + ".layer4.attn_cos_skip", {1024}, &blk.attn_cos_off)) return false;
      continue;
    }

    if (blk.family == BlockFamily::SplitWindow || blk.family == BlockFamily::SplitDownsample) {
      if (!fetch_pack(bstr + ".layer0.first_projection_weight", {512, 512}, &blk.ffn_w1_off)) return false;
      if (!fetch_pack(bstr + ".layer0.group_expand_weight", {8, 64, 256}, &blk.ffn_w2_off)) return false;
      if (!fetch_pack(bstr + ".layer0.group_project_weight", {8, 256, 64}, &blk.ffn_w3_off)) return false;
      if (!fetch_pack(bstr + ".layer1.weight3", {512, 512}, &blk.ffn_w4_off)) return false;
      if (!fetch_pack(bstr + ".layer1.ffn_cos_skip", {512}, &blk.ffn_cos_off)) return false;
      if (!fetch_pack(bstr + ".layer2.qkv_weight", {512, 1536}, &blk.qkv_off)) return false;
      if (!fetch_pack(bstr + ".layer2.attn_scale", {16}, &blk.attn_scale_off)) return false;
      if (!fetch_bias_pack(bstr + ".layer2.attn_bias", 16, &blk.attn_bias_off)) return false;
      if (!fetch_pack(bstr + ".layer3.projection_weight", {512, 512}, &blk.attn_proj_off)) return false;
      if (!fetch_pack(bstr + ".layer3.attn_cos_skip", {512}, &blk.attn_cos_off)) return false;
      if (blk.family == BlockFamily::SplitDownsample) {
        if (!fetch_pack(bstr + ".layer4.weight", {512, 1024}, &blk.w0_off)) return false;
      }
      continue;
    }

    // Window / Pre / Downsample / Upsample / Post blocks:
    const int C = blk.channels;
    const int H = blk.heads;
    if (blk.family == BlockFamily::Pre) {
      if (!fetch_pack(l0 + "input_adapter_weight", {16, 32}, &blk.adapter_off)) return false;
    } else if (blk.family == BlockFamily::Downsample) {
      if (!fetch_pack(l0 + "weight0", {C, C * 2}, &blk.w0_off)) return false;
    } else if (blk.family == BlockFamily::Upsample) {
      if (!fetch_pack(l0 + "weight0", {C * 2, C}, &blk.w0_off)) return false;
      if (!fetch_pack(l0 + "sin", {C}, &blk.sin_off)) return false;
    } else if (blk.family == BlockFamily::Post) {
      if (!fetch_pack(l0 + "inp_merge_sin", {32}, &blk.sin_off)) return false;
      if (!fetch_pack(l0 + "inp_merge_cos", {32}, &blk.cos_merge_off)) return false;
      std::vector<uint16_t> gain;
      std::vector<uint16_t> conv;
      if (!provider.FetchFp16(l0 + "out_gain", {16, 4}, &gain, 1.0f, out_err)) return false;
      if (!provider.FetchFp16(l0 + "out_conv_weight", {16, 4}, &conv, 1.0f, out_err)) return false;
      // Pack [16, 4] out_gain + [16, 4] out_conv_weight into [32, 8] headWeight.
      std::vector<uint16_t> head_w(32 * 8, 0);
      for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 4; ++c) {
          head_w[static_cast<size_t>(r * 8 + c)] = gain[static_cast<size_t>(r * 4 + c)];
          head_w[static_cast<size_t>((16 + r) * 8 + 4 + c)] = conv[static_cast<size_t>(r * 4 + c)];
        }
      }
      blk.head_weight_off = packer.AppendAligned(head_w);
    }

    if (C == 32) {
      if (!fetch_pack(l0 + "weight1", {32, 128}, &blk.ffn_w1_off)) return false;
      if (!fetch_pack(l0 + "weight2", {128, 32}, &blk.ffn_w2_off)) return false;
    } else {
      // Pre-transpose ffn_expand_weight [G, 4, G, 32, 32] (axes 2, 3, 0, 1, 4) -> [C, 4*C].
      if (!provider.FetchFp16(l0 + "ffn_expand_weight", {H, 4, H, 32, 32}, &raw, 1.0f, out_err)) {
        return false;
      }
      std::vector<uint16_t> dense_exp(static_cast<size_t>(C * 4 * C));
      for (int go = 0; go < H; ++go) {
        for (int br = 0; br < 4; ++br) {
          for (int gi = 0; gi < H; ++gi) {
            for (int k = 0; k < 32; ++k) {
              const int dst_row = gi * 32 + k;
              const int dst_col_base = (go * 4 + br) * 32;
              const size_t src_base =
                  static_cast<size_t>((((go * 4 + br) * H + gi) * 32 + k) * 32);
              std::memcpy(
                  dense_exp.data() + static_cast<size_t>(dst_row * (4 * C) + dst_col_base),
                  raw.data() + src_base,
                  32 * sizeof(uint16_t));
            }
          }
        }
      }
      blk.ffn_w1_off = packer.AppendAligned(dense_exp);
      // ffn_branch_projection_weight [G, 4, 32, 32] is already contiguous [G, 128, 32].
      if (!fetch_pack(l0 + "ffn_branch_projection_weight", {H, 4, 32, 32}, &blk.ffn_w2_off)) return false;
      if (!fetch_pack(l0 + "ffn_output_projection_weight", {C, C}, &blk.ffn_w3_off)) return false;
    }

    if (!fetch_pack(l0 + "ffn_cos_skip", {C}, &blk.ffn_cos_off)) return false;
    if (!fetch_pack(l0 + "qkv_weight", {C, C * 3}, &blk.qkv_off)) return false;
    if (!fetch_pack(l0 + "attn_scale", {H}, &blk.attn_scale_off)) return false;
    if (!fetch_bias_pack(l0 + "attn_bias", H, &blk.attn_bias_off)) return false;
    if (!fetch_pack(l0 + "projection_weight", {C, C}, &blk.attn_proj_off)) return false;
    if (!fetch_pack(l0 + "attn_cos_skip", {C}, &blk.attn_cos_off)) return false;
  }

  *out_packed_bytes = packer.bytes();
  return true;
}

std::string FindDefaultMetallibPath() {
  uint32_t sz = 0;
  _NSGetExecutablePath(nullptr, &sz);
  if (sz > 0) {
    std::vector<char> buf(sz + 1, 0);
    if (_NSGetExecutablePath(buf.data(), &sz) == 0) {
      std::string exe(buf.data());
      size_t slash = exe.find_last_of('/');
      if (slash != std::string::npos) {
        std::string candidate = exe.substr(0, slash) + "/nr_fast_kernels.metallib";
        struct stat st {};
        if (stat(candidate.c_str(), &st) == 0 && S_ISREG(st.st_mode)) {
          return candidate;
        }
      }
    }
  }
  return {};
}

std::vector<float> BuildGaussianWeights(float sigma, uint32_t* out_radius) {
  int extent = std::clamp(static_cast<int>(std::ceil(3.0f * sigma)), 1, 32);
  *out_radius = static_cast<uint32_t>(extent);
  std::vector<float> k(static_cast<size_t>(2 * extent + 1));
  float sum = 0.0f;
  for (int d = -extent; d <= extent; ++d) {
    float fd = static_cast<float>(d);
    float w = std::exp(-fd * fd / (2.0f * sigma * sigma));
    k[static_cast<size_t>(d + extent)] = w;
    sum += w;
  }
  for (float& w : k) w /= sum;
  return k;
}

}  // namespace

// ============================================================================
// Public Standalone Utilities
// ============================================================================

uint16_t DecodeNativeE4M3ToFp16Bits(uint8_t byte_val) {
  return GetE4M3Lut()[byte_val];
}

uint8_t EncodeFp16BitsToNativeE4M3(uint16_t fp16_bits) {
  return EncodeFloatToE4M3Byte(Fp16BitsToFloat(fp16_bits));
}

bool ValidateModelFile(
    const std::string& path,
    SafeTensorSummary* out_summary,
    std::string* out_error,
    bool reject_symlinks,
    bool verify_manifest_sha256) {
  @autoreleasepool {
    ParsedSafeTensors parsed;
    SafeTensorSummary summary;
    if (!LoadAndValidatePackageOrSafeTensors(
            path, reject_symlinks, verify_manifest_sha256, &parsed, &summary, out_error)) {
      return false;
    }
    if (!InferTopologyFromTensors(parsed, &summary.topology, out_error)) {
      return false;
    }
    if (out_summary) *out_summary = std::move(summary);
    return true;
  }
}

// ============================================================================
// Context::Impl
// ============================================================================

struct ScratchArenaLayout {
  uint32_t net_w = 0;
  uint32_t net_h = 0;
  uint32_t out_w = 0;
  uint32_t out_h = 0;

  size_t feat_f16_off = 0;
  size_t head_f16_off = 0;
  size_t base_net_rgb_off = 0;
  size_t delta_net_rgb_off = 0;
  size_t delta_full_rgb_off = 0;
  size_t blur_h_rgb_off = 0;

  size_t full_skip_off = 0;
  size_t skip_offs[4] = {0, 0, 0, 0};
  size_t split_skip_off = 0;
  size_t ping_off = 0;
  size_t pong_off = 0;
  size_t tmp0_off = 0;
  size_t tmp1_off = 0;
  size_t tmp2_off = 0;
  size_t total_bytes = 0;
};

struct Context::Impl {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> internal_queue = nil;
  id<MTLLibrary> library = nil;

  // Compute pipelines:
  id<MTLComputePipelineState> pso_f32_to_f16 = nil;
  id<MTLComputePipelineState> pso_f16_to_f32 = nil;
  id<MTLComputePipelineState> pso_pool2_pad = nil;
  id<MTLComputePipelineState> pso_upsample_merge = nil;

  id<MTLComputePipelineState> pso_win_1h32 = nil;
  id<MTLComputePipelineState> pso_win_1h32_adapter_pool = nil;
  id<MTLComputePipelineState> pso_win_1h32_pool = nil;
  id<MTLComputePipelineState> pso_win_1h32_merge_store = nil;
  id<MTLComputePipelineState> pso_win_1h32_merge_head = nil;

  id<MTLComputePipelineState> pso_win_mh = nil;
  id<MTLComputePipelineState> pso_branched_ffn_64 = nil;
  id<MTLComputePipelineState> pso_branched_ffn_128 = nil;
  id<MTLComputePipelineState> pso_win_attn_block_64 = nil;

  id<MTLComputePipelineState> pso_gemm = nil;
  id<MTLComputePipelineState> pso_gemm_bm64 = nil;
  id<MTLComputePipelineState> pso_grouped_gemm = nil;

  id<MTLComputePipelineState> pso_glob_pub_qkv = nil;
  id<MTLComputePipelineState> pso_glob_resident = nil;
  id<MTLComputePipelineState> pso_glob_split_norm = nil;
  id<MTLComputePipelineState> pso_glob_denom = nil;
  id<MTLComputePipelineState> pso_glob_values = nil;

  id<MTLComputePipelineState> pso_pre_features = nil;
  id<MTLComputePipelineState> pso_post_temporal = nil;
  id<MTLComputePipelineState> pso_upsample_compose = nil;
  id<MTLComputePipelineState> pso_detail_blur_h = nil;
  id<MTLComputePipelineState> pso_detail_blur_v = nil;
  id<MTLComputePipelineState> pso_bake_delta_blur_h = nil;
  id<MTLComputePipelineState> pso_bake_delta_tex = nil;
  id<MTLComputePipelineState> pso_compose_baked_tex = nil;

  TopologySpec topology;
  bool synthetic = false;
  bool use_residency_set = true;
  std::string weight_format;
  std::vector<PreparedBlockOffsets> blocks;

  id<MTLBuffer> weights_buf = nil;
  size_t weights_bytes = 0;

  ScratchArenaLayout scratch_layout;
  id<MTLBuffer> scratch_buf = nil;

  // Separate scratch arena exclusively for `encode_compose_delta` when
  // `detail_strength != colour_strength`, avoiding any hazard with `scratch_buf`
  // while `encode()` executes concurrently on a background worker queue.
  std::mutex async_compose_mutex;
  id<MTLBuffer> async_compose_scratch_buf = nil;
  size_t async_compose_blur_h_off = 0;
  uint32_t async_compose_out_w = 0;
  uint32_t async_compose_out_h = 0;

  // Double-buffered persistent history textures (read previous, write next) so
  // single-encoder dispatches never alias read/write on the same MTLTexture.
  id<MTLTexture> history_tex[2] = {nil, nil};
  int active_history_idx = 0;
  bool history_valid = false;
  uint32_t history_w = 0;
  uint32_t history_h = 0;

  // Dummy 1x1 texture bound to unused optional texture slots.
  id<MTLTexture> dummy_tex = nil;

  id residency_set_obj = nil;  // id<MTLResidencySet> on macOS 15+

  bool InitPipelines(const std::string& metallib_override, std::string* out_err) {
    std::string mpath = metallib_override.empty() ? FindDefaultMetallibPath() : metallib_override;
    if (mpath.empty()) {
      if (out_err) {
        *out_err = "nr_fast_kernels.metallib path not provided and not found beside executable";
      }
      return false;
    }
    NSError* err = nil;
    NSURL* url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:mpath.c_str()]];
    library = [device newLibraryWithURL:url error:&err];
    if (!library) {
      if (out_err) {
        *out_err = "failed to load Metal library '" + mpath + "': " +
                   (err ? std::string([[err localizedDescription] UTF8String]) : "unknown");
      }
      return false;
    }

    auto make_pso = [&](const char* fn_name, __strong id<MTLComputePipelineState>* out_pso) -> bool {
      id<MTLFunction> fn = [library newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
      if (!fn) {
        if (out_err) *out_err = std::string("missing Metal kernel function: ") + fn_name;
        return false;
      }
      NSError* perr = nil;
      *out_pso = [device newComputePipelineStateWithFunction:fn error:&perr];
      if (!*out_pso) {
        if (out_err) {
          *out_err = std::string("failed to create pipeline for ") + fn_name + ": " +
                     (perr ? std::string([[perr localizedDescription] UTF8String]) : "unknown");
        }
        return false;
      }
      return true;
    };

    if (!make_pso("nr_fast_f32_to_f16", &pso_f32_to_f16) ||
        !make_pso("nr_fast_f16_to_f32", &pso_f16_to_f32) ||
        !make_pso("nr_fast_pool2_pad", &pso_pool2_pad) ||
        !make_pso("nr_fast_upsample_merge", &pso_upsample_merge) ||
        !make_pso("nr_fast_fused_window_1h32", &pso_win_1h32) ||
        !make_pso("nr_fast_fused_window_1h32_adapter_pool", &pso_win_1h32_adapter_pool) ||
        !make_pso("nr_fast_fused_window_1h32_pool", &pso_win_1h32_pool) ||
        !make_pso("nr_fast_fused_window_1h32_merge_store", &pso_win_1h32_merge_store) ||
        !make_pso("nr_fast_fused_window_1h32_merge_head", &pso_win_1h32_merge_head) ||
        !make_pso("nr_fast_window_attention_mh", &pso_win_mh) ||
        !make_pso("nr_fast_branched_ffn_64", &pso_branched_ffn_64) ||
        !make_pso("nr_fast_branched_ffn_128", &pso_branched_ffn_128) ||
        !make_pso("nr_fast_window_attn_block_64", &pso_win_attn_block_64) ||
        !make_pso("nr_fast_gemm_f16", &pso_gemm) ||
        !make_pso("nr_fast_gemm_f16_bm64", &pso_gemm_bm64) ||
        !make_pso("nr_fast_grouped_gemm_f16", &pso_grouped_gemm) ||
        !make_pso("nr_fast_global_publish_qkv", &pso_glob_pub_qkv) ||
        !make_pso("nr_fast_global_attn_resident", &pso_glob_resident) ||
        !make_pso("nr_fast_global_qkv_split_norm", &pso_glob_split_norm) ||
        !make_pso("nr_fast_global_attn_denom", &pso_glob_denom) ||
        !make_pso("nr_fast_global_attn_values", &pso_glob_values) ||
        !make_pso("nr_fast_pre_features", &pso_pre_features) ||
        !make_pso("nr_fast_post_temporal", &pso_post_temporal) ||
        !make_pso("nr_fast_upsample_compose", &pso_upsample_compose) ||
        !make_pso("nr_fast_detail_blur_h", &pso_detail_blur_h) ||
        !make_pso("nr_fast_detail_blur_v_compose", &pso_detail_blur_v) ||
        !make_pso("nr_fast_bake_delta_blur_h", &pso_bake_delta_blur_h) ||
        !make_pso("nr_fast_bake_delta_tex", &pso_bake_delta_tex) ||
        !make_pso("nr_fast_compose_baked_tex", &pso_compose_baked_tex)) {
      return false;
    }

    internal_queue = [device newCommandQueue];
    if (!internal_queue) {
      if (out_err) *out_err = "failed to create internal Metal command queue";
      return false;
    }

    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:1
                                                          height:1
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    dummy_tex = [device newTextureWithDescriptor:td];
    return dummy_tex != nil;
  }

  bool UploadWeights(const std::vector<uint8_t>& packed, std::string* out_err) {
    weights_bytes = AlignUp(packed.size(), kBufferAlign);
    id<MTLBuffer> staging = [device newBufferWithLength:weights_bytes
                                                options:MTLResourceStorageModeShared];
    if (!staging) {
      if (out_err) *out_err = "failed to allocate staging weight buffer";
      return false;
    }
    std::memset([staging contents], 0, weights_bytes);
    std::memcpy([staging contents], packed.data(), packed.size());

    weights_buf =
        [device newBufferWithLength:weights_bytes
                            options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
    if (!weights_buf) {
      weights_buf = staging;
      RebuildResidencySet();
      return true;
    }

    id<MTLCommandBuffer> cmd = [internal_queue commandBuffer];
    id<MTLBlitCommandEncoder> blit = [cmd blitCommandEncoder];
    [blit copyFromBuffer:staging sourceOffset:0 toBuffer:weights_buf destinationOffset:0 size:weights_bytes];
    [blit endEncoding];
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      if (out_err) *out_err = "failed to blit prepared weights to private GPU buffer";
      return false;
    }
    RebuildResidencySet();
    return true;
  }

  void RebuildResidencySet() {
    if (!use_residency_set) return;
    if (@available(macOS 15.0, *)) {
      if ([device respondsToSelector:@selector(newResidencySetWithDescriptor:error:)]) {
        MTLResidencySetDescriptor* desc = [[MTLResidencySetDescriptor alloc] init];
        desc.initialCapacity = 8;
        NSError* err = nil;
        id<MTLResidencySet> rset = [device newResidencySetWithDescriptor:desc error:&err];
        if (rset) {
          if (weights_buf) [rset addAllocation:weights_buf];
          if (scratch_buf) [rset addAllocation:scratch_buf];
          if (dummy_tex) [rset addAllocation:dummy_tex];
          if (history_tex[0]) [rset addAllocation:history_tex[0]];
          if (history_tex[1]) [rset addAllocation:history_tex[1]];
          [rset commit];
          [rset requestResidency];
          residency_set_obj = rset;
        }
      }
    }
  }

  bool EnsureScratch(uint32_t net_w, uint32_t net_h, uint32_t out_w, uint32_t out_h, std::string* out_err) {
    if (scratch_buf && scratch_layout.net_w == net_w && scratch_layout.net_h == net_h &&
        scratch_layout.out_w == out_w && scratch_layout.out_h == out_h) {
      return true;
    }

    ScratchArenaLayout lay{};
    lay.net_w = net_w;
    lay.net_h = net_h;
    lay.out_w = out_w;
    lay.out_h = out_h;

    const size_t H = net_h;
    const size_t W = net_w;
    const size_t H0 = H / 2, W0 = W / 2;
    const size_t H1 = H / 4, W1 = W / 4;
    const size_t H2 = H / 8, W2 = W / 8;
    const size_t H3 = H / 16, W3 = W / 16;
    const size_t H3p = ((H3 + 7) / 8) * 8, W3p = ((W3 + 7) / 8) * 8;
    const size_t Hs = H3p / 2, Ws = W3p / 2;
    const size_t Hsp = ((Hs + 7) / 8) * 8, Wsp = ((Ws + 7) / 8) * 8;
    const size_t Hg = Hsp / 2, Wg = Wsp / 2;
    const size_t T_glob = Hg * Wg;
    const size_t pitch = ((T_glob + 7) / 8) * 8;
    const size_t groups = ((pitch / 8) + 3) / 4;

    size_t cursor = 0;
    auto alloc = [&](size_t bytes) -> size_t {
      size_t off = AlignUp(cursor, kBufferAlign);
      cursor = off + AlignUp(std::max<size_t>(bytes, 256), kBufferAlign);
      return off;
    };

    lay.feat_f16_off = alloc(H * W * 16 * sizeof(uint16_t));
    lay.head_f16_off = alloc(H * W * 4 * sizeof(uint16_t));
    lay.base_net_rgb_off = alloc(H * W * 4 * sizeof(float));
    lay.delta_net_rgb_off = alloc(H * W * 4 * sizeof(float));
    if (out_w > 0 && out_h > 0) {
      lay.delta_full_rgb_off = alloc(static_cast<size_t>(out_h) * out_w * 4 * sizeof(float));
      lay.blur_h_rgb_off = alloc(static_cast<size_t>(out_h) * out_w * 4 * sizeof(float));
    }

    lay.full_skip_off = alloc(H * W * 32 * sizeof(uint16_t));
    lay.skip_offs[0] = alloc(H0 * W0 * 32 * sizeof(uint16_t));
    lay.skip_offs[1] = alloc(H1 * W1 * 64 * sizeof(uint16_t));
    lay.skip_offs[2] = alloc(H2 * W2 * 128 * sizeof(uint16_t));
    lay.skip_offs[3] = alloc(H3 * W3 * 256 * sizeof(uint16_t));
    lay.split_skip_off = alloc(Hs * Ws * 512 * sizeof(uint16_t));

    size_t max_state_elems = std::max({
        H0 * W0 * 32,
        H1 * W1 * 64,
        H2 * W2 * 128,
        H3 * W3 * 256,
        Hs * Ws * 512,
        T_glob * 1024,
    });
    lay.ping_off = alloc(max_state_elems * sizeof(uint16_t));
    lay.pong_off = alloc(max_state_elems * sizeof(uint16_t));

    // Workspace temporaries:
    size_t max_tmp0_elems = std::max({
        H1 * W1 * 64,
        H2 * W2 * 384,
        H3 * W3 * 1024,
        Hs * Ws * 1536,
        T_glob * 4096,
        32 * T_glob + T_glob * 1024,
    });
    size_t max_tmp1_elems = std::max({
        H2 * W2 * 128,
        H3 * W3 * 256,
        Hs * Ws * 2048,
        32 * (pitch * 32 + pitch * 32 + groups * 1024) + 1024,
        3 * 32 * T_glob * 32,
    });
    size_t max_tmp2_elems = std::max({
        H2 * W2 * 128,
        H3 * W3 * 256,
        Hs * Ws * 512,
        T_glob * 1024,
    });

    lay.tmp0_off = alloc(max_tmp0_elems * sizeof(uint16_t));
    lay.tmp1_off = alloc(max_tmp1_elems * sizeof(uint16_t));
    lay.tmp2_off = alloc(max_tmp2_elems * sizeof(uint16_t));
    lay.total_bytes = cursor;

    if (!scratch_buf || [scratch_buf length] < lay.total_bytes) {
      scratch_buf =
          [device newBufferWithLength:lay.total_bytes
                              options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
      if (!scratch_buf) {
        scratch_buf = [device newBufferWithLength:lay.total_bytes options:MTLResourceStorageModeShared];
      }
      if (!scratch_buf) {
        if (out_err) *out_err = "failed to allocate GPU scratch arena";
        return false;
      }
      RebuildResidencySet();
    }
    scratch_layout = lay;
    return true;
  }

  bool EnsureAsyncComposeScratch(
      uint32_t out_w,
      uint32_t out_h,
      id<MTLBuffer>* out_buf,
      size_t* out_blur_h_off,
      std::string* out_err) {
    std::lock_guard<std::mutex> lock(async_compose_mutex);
    const size_t plane_bytes =
        AlignUp(std::max<size_t>(static_cast<size_t>(out_w) * out_h * 4u * sizeof(float), 256u), kBufferAlign);
    const size_t needed = plane_bytes * 2u;
    if (!async_compose_scratch_buf || [async_compose_scratch_buf length] < needed ||
        async_compose_out_w != out_w || async_compose_out_h != out_h) {
      if (!async_compose_scratch_buf || [async_compose_scratch_buf length] < needed) {
        id<MTLBuffer> buf =
            [device newBufferWithLength:needed
                                options:MTLResourceStorageModePrivate | MTLResourceHazardTrackingModeUntracked];
        if (!buf) {
          buf = [device newBufferWithLength:needed options:MTLResourceStorageModeShared];
        }
        if (!buf) {
          if (out_err) *out_err = "failed to allocate async compose scratch buffer";
          return false;
        }
        async_compose_scratch_buf = buf;
      }
      async_compose_blur_h_off = plane_bytes;
      async_compose_out_w = out_w;
      async_compose_out_h = out_h;
    }
    *out_buf = async_compose_scratch_buf;
    *out_blur_h_off = async_compose_blur_h_off;
    return true;
  }

  bool EnsureHistoryTextures(uint32_t w, uint32_t h, std::string* out_err) {
    if (history_tex[0] && history_tex[1] && history_w == w && history_h == h) {
      return true;
    }
    MTLTextureDescriptor* td =
        [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA16Float
                                                           width:w
                                                          height:h
                                                       mipmapped:NO];
    td.usage = MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite;
    td.storageMode = MTLStorageModePrivate;
    history_tex[0] = [device newTextureWithDescriptor:td];
    history_tex[1] = [device newTextureWithDescriptor:td];
    if (!history_tex[0] || !history_tex[1]) {
      if (out_err) *out_err = "failed to allocate persistent history textures";
      return false;
    }
    history_w = w;
    history_h = h;
    active_history_idx = 0;
    history_valid = false;
    RebuildResidencySet();
    return true;
  }

  // Encodes the entire 31-block or 71-block U-Net on `enc` with buffer barriers.
  void EncodeUNetCore(
      id<MTLComputeCommandEncoder> enc,
      id<MTLBuffer> feat_buf,
      size_t feat_off,
      id<MTLBuffer> head_buf,
      size_t head_off,
      uint32_t H,
      uint32_t W,
      uint32_t& dispatches) {
    const ScratchArenaLayout& s = scratch_layout;
    auto buf_barrier = [&]() { [enc memoryBarrierWithScope:MTLBarrierScopeBuffers]; };

    auto dispatch_1d_groups = [&](id<MTLComputePipelineState> pso, uint32_t groups, uint32_t tpg) {
      [enc setComputePipelineState:pso];
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
      ++dispatches;
    };

    auto dispatch_threads_1d = [&](id<MTLComputePipelineState> pso, uint32_t total) {
      if (total == 0) return;
      [enc setComputePipelineState:pso];
      uint32_t tpg = std::min<uint32_t>(total, 256u);
      uint32_t groups = (total + tpg - 1u) / tpg;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];
      ++dispatches;
    };

    auto emit_gemm = [&](size_t a_off,
                         size_t b_w_off,
                         size_t skip_off,
                         size_t cos_w_off,
                         size_t c_off,
                         uint32_t M,
                         uint32_t N,
                         uint32_t K,
                         uint32_t epilogue) {
      KGemmParams gp{M, N, K, epilogue};
      const bool use_bm64 = (M >= 1024u);
      [enc setComputePipelineState:(use_bm64 ? pso_gemm_bm64 : pso_gemm)];
      [enc setBuffer:scratch_buf offset:a_off atIndex:0];
      [enc setBuffer:weights_buf offset:b_w_off atIndex:1];
      [enc setBuffer:scratch_buf offset:skip_off atIndex:2];
      [enc setBuffer:weights_buf offset:cos_w_off atIndex:3];
      [enc setBuffer:scratch_buf offset:c_off atIndex:4];
      [enc setBytes:&gp length:sizeof(gp) atIndex:5];
      uint32_t gx = (N + 63u) / 64u;
      uint32_t gy = use_bm64 ? ((M + 63u) / 64u) : ((M + 31u) / 32u);
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, 1) threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      ++dispatches;
      buf_barrier();
    };

    auto emit_grouped_gemm = [&](size_t a_off,
                                 size_t b_w_off,
                                 size_t c_off,
                                 uint32_t M,
                                 uint32_t N,
                                 uint32_t K,
                                 uint32_t groups,
                                 uint32_t epilogue) {
      KGroupedGemmParams gp{M, N, K, groups, epilogue};
      [enc setComputePipelineState:pso_grouped_gemm];
      [enc setBuffer:scratch_buf offset:a_off atIndex:0];
      [enc setBuffer:weights_buf offset:b_w_off atIndex:1];
      [enc setBuffer:scratch_buf offset:c_off atIndex:2];
      [enc setBytes:&gp length:sizeof(gp) atIndex:3];
      uint32_t gx = (N + 31u) / 32u;
      uint32_t gy = (M + 31u) / 32u;
      [enc dispatchThreadgroups:MTLSizeMake(gx, gy, groups)
          threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
      ++dispatches;
      buf_barrier();
    };

    auto emit_pool2_pad = [&](size_t in_off,
                              size_t out_off,
                              uint32_t ih,
                              uint32_t iw,
                              uint32_t oh,
                              uint32_t ow,
                              uint32_t ch,
                              uint32_t pub) {
      KPool2PadParams pp{ih, iw, oh, ow, ch, pub};
      [enc setBuffer:scratch_buf offset:in_off atIndex:0];
      [enc setBuffer:scratch_buf offset:out_off atIndex:1];
      [enc setBytes:&pp length:sizeof(pp) atIndex:2];
      dispatch_threads_1d(pso_pool2_pad, oh * ow * (ch / 4u));
      buf_barrier();
    };

    auto emit_upsample_merge = [&](size_t low_off,
                                   size_t skip_off,
                                   size_t low_sc_w_off,
                                   size_t skip_sc_w_off,
                                   size_t out_off,
                                   uint32_t h,
                                   uint32_t w,
                                   uint32_t ch,
                                   uint32_t low_w,
                                   uint32_t has_low_sc,
                                   uint32_t pub) {
      KUpsampleMergeParams up{h, w, ch, low_w, has_low_sc, pub};
      [enc setBuffer:scratch_buf offset:low_off atIndex:0];
      [enc setBuffer:scratch_buf offset:skip_off atIndex:1];
      [enc setBuffer:weights_buf offset:low_sc_w_off atIndex:2];
      [enc setBuffer:weights_buf offset:skip_sc_w_off atIndex:3];
      [enc setBuffer:scratch_buf offset:out_off atIndex:4];
      [enc setBytes:&up length:sizeof(up) atIndex:5];
      dispatch_threads_1d(pso_upsample_merge, h * w * (ch / 4u));
      buf_barrier();
    };

    auto emit_win_1h32 = [&](const PreparedBlockOffsets& blk,
                             size_t in_off,
                             size_t out_off,
                             uint32_t h,
                             uint32_t w,
                             uint32_t pub) {
      KWindow1H32Params wp{h, w, blk.pad_top, blk.pad_left, pub, 0, 0, 0};
      uint32_t win_cnt = ((h + blk.pad_top + 7u) / 8u) * ((w + blk.pad_left + 7u) / 8u);
      [enc setBuffer:scratch_buf offset:in_off atIndex:0];
      [enc setBuffer:weights_buf offset:blk.ffn_w1_off atIndex:1];
      [enc setBuffer:weights_buf offset:blk.ffn_w2_off atIndex:2];
      [enc setBuffer:weights_buf offset:blk.ffn_cos_off atIndex:3];
      [enc setBuffer:weights_buf offset:blk.qkv_off atIndex:4];
      [enc setBuffer:weights_buf offset:blk.attn_scale_off atIndex:5];
      [enc setBuffer:weights_buf offset:blk.attn_bias_off atIndex:6];
      [enc setBuffer:weights_buf offset:blk.attn_proj_off atIndex:7];
      [enc setBuffer:weights_buf offset:blk.attn_cos_off atIndex:8];
      [enc setBuffer:scratch_buf offset:out_off atIndex:9];
      [enc setBytes:&wp length:sizeof(wp) atIndex:10];
      dispatch_1d_groups(pso_win_1h32, win_cnt, 256);
      buf_barrier();
    };

    auto emit_window_mh_core = [&](const PreparedBlockOffsets& blk,
                                   size_t qkv_off,
                                   size_t out_off,
                                   uint32_t h,
                                   uint32_t w) {
      KWindowMHParams wp{h, w, blk.pad_top, blk.pad_left, static_cast<uint32_t>(blk.heads), 1u};
      uint32_t win_cnt = ((h + blk.pad_top + 7u) / 8u) * ((w + blk.pad_left + 7u) / 8u) *
                         static_cast<uint32_t>(blk.heads);
      [enc setBuffer:scratch_buf offset:qkv_off atIndex:0];
      [enc setBuffer:weights_buf offset:blk.attn_scale_off atIndex:1];
      [enc setBuffer:weights_buf offset:blk.attn_bias_off atIndex:2];
      [enc setBuffer:scratch_buf offset:out_off atIndex:3];
      [enc setBytes:&wp length:sizeof(wp) atIndex:4];
      dispatch_1d_groups(pso_win_mh, win_cnt, 256);
      buf_barrier();
    };

    auto emit_branched_block = [&](const PreparedBlockOffsets& blk,
                                   size_t in_off,
                                   size_t out_off,
                                   uint32_t h,
                                   uint32_t w,
                                   uint32_t pub) {
      const uint32_t C = static_cast<uint32_t>(blk.channels);
      const uint32_t M = h * w;
      if (C == 32u) {
        emit_win_1h32(blk, in_off, out_off, h, w, pub);
        return;
      }
      if (C == 64u) {
        KCountParams cp{M, 0, 0, 0};
        uint32_t ffn_groups = (M + 31u) / 32u;
        [enc setBuffer:scratch_buf offset:in_off atIndex:0];
        [enc setBuffer:weights_buf offset:blk.ffn_w1_off atIndex:1];
        [enc setBuffer:weights_buf offset:blk.ffn_w2_off atIndex:2];
        [enc setBuffer:weights_buf offset:blk.ffn_w3_off atIndex:3];
        [enc setBuffer:weights_buf offset:blk.ffn_cos_off atIndex:4];
        [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:5];
        [enc setBytes:&cp length:sizeof(cp) atIndex:6];
        dispatch_1d_groups(pso_branched_ffn_64, ffn_groups, 128);
        buf_barrier();

        KWindowMHParams wp{h, w, blk.pad_top, blk.pad_left, 2u, pub};
        uint32_t win_cnt = ((h + blk.pad_top + 7u) / 8u) * ((w + blk.pad_left + 7u) / 8u);
        [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:0];
        [enc setBuffer:weights_buf offset:blk.qkv_off atIndex:1];
        [enc setBuffer:weights_buf offset:blk.attn_scale_off atIndex:2];
        [enc setBuffer:weights_buf offset:blk.attn_bias_off atIndex:3];
        [enc setBuffer:weights_buf offset:blk.attn_proj_off atIndex:4];
        [enc setBuffer:weights_buf offset:blk.attn_cos_off atIndex:5];
        [enc setBuffer:scratch_buf offset:out_off atIndex:6];
        [enc setBytes:&wp length:sizeof(wp) atIndex:7];
        dispatch_1d_groups(pso_win_attn_block_64, win_cnt, 256);
        buf_barrier();
        return;
      }
      if (C == 128u) {
        KCountParams cp{M, 0, 0, 0};
        uint32_t ffn_groups = (M + 31u) / 32u;
        [enc setBuffer:scratch_buf offset:in_off atIndex:0];
        [enc setBuffer:weights_buf offset:blk.ffn_w1_off atIndex:1];
        [enc setBuffer:weights_buf offset:blk.ffn_w2_off atIndex:2];
        [enc setBuffer:weights_buf offset:blk.ffn_w3_off atIndex:3];
        [enc setBuffer:weights_buf offset:blk.ffn_cos_off atIndex:4];
        [enc setBuffer:scratch_buf offset:s.tmp2_off atIndex:5];
        [enc setBytes:&cp length:sizeof(cp) atIndex:6];
        dispatch_1d_groups(pso_branched_ffn_128, ffn_groups, 128);
        buf_barrier();

        emit_gemm(s.tmp2_off, blk.qkv_off, 0, 0, s.tmp0_off, M, 384u, 128u, EPILOGUE_NONE);
        emit_window_mh_core(blk, s.tmp0_off, s.tmp1_off, h, w);
        emit_gemm(
            s.tmp1_off,
            blk.attn_proj_off,
            s.tmp2_off,
            blk.attn_cos_off,
            out_off,
            M,
            128u,
            128u,
            pub ? EPILOGUE_COS_RES_E4M3 : EPILOGUE_COS_RES);
        return;
      }

      // C == 256: tiled GEMM + zero-transpose grouped GEMM + MH window core.
      emit_gemm(in_off, blk.ffn_w1_off, 0, 0, s.tmp0_off, M, C * 4u, C, EPILOGUE_QUAD_GATE_E4M3);
      emit_grouped_gemm(s.tmp0_off, blk.ffn_w2_off, s.tmp1_off, M, 32u, 128u, C / 32u, EPILOGUE_E4M3);
      emit_gemm(
          s.tmp1_off, blk.ffn_w3_off, in_off, blk.ffn_cos_off, s.tmp2_off, M, C, C, EPILOGUE_COS_RES_E4M3);
      emit_gemm(s.tmp2_off, blk.qkv_off, 0, 0, s.tmp0_off, M, C * 3u, C, EPILOGUE_NONE);
      emit_window_mh_core(blk, s.tmp0_off, s.tmp1_off, h, w);
      emit_gemm(
          s.tmp1_off,
          blk.attn_proj_off,
          s.tmp2_off,
          blk.attn_cos_off,
          out_off,
          M,
          C,
          C,
          pub ? EPILOGUE_COS_RES_E4M3 : EPILOGUE_COS_RES);
    };

    auto emit_split_block = [&](const PreparedBlockOffsets& blk,
                                size_t in_off,
                                size_t out_off,
                                uint32_t h,
                                uint32_t w) {
      const uint32_t M = h * w;
      emit_gemm(in_off, blk.ffn_w1_off, 0, 0, s.tmp0_off, M, 512u, 512u, EPILOGUE_E4M3);
      emit_grouped_gemm(s.tmp0_off, blk.ffn_w2_off, s.tmp1_off, M, 256u, 64u, 8u, EPILOGUE_QUAD_GATE);
      emit_grouped_gemm(s.tmp1_off, blk.ffn_w3_off, s.tmp0_off, M, 64u, 256u, 8u, EPILOGUE_E4M3);
      emit_gemm(
          s.tmp0_off, blk.ffn_w4_off, in_off, blk.ffn_cos_off, s.tmp2_off, M, 512u, 512u, EPILOGUE_COS_RES);
      emit_gemm(s.tmp2_off, blk.qkv_off, 0, 0, s.tmp0_off, M, 1536u, 512u, EPILOGUE_NONE);
      emit_window_mh_core(blk, s.tmp0_off, s.tmp1_off, h, w);
      emit_gemm(
          s.tmp1_off,
          blk.attn_proj_off,
          s.tmp2_off,
          blk.attn_cos_off,
          out_off,
          M,
          512u,
          512u,
          EPILOGUE_COS_RES_E4M3);
    };

    // Spatial pyramid dimensions:
    const uint32_t stage_h[4] = {H / 2u, H / 4u, H / 8u, H / 16u};
    const uint32_t stage_w[4] = {W / 2u, W / 4u, W / 8u, W / 16u};
    const uint32_t h3p = ((stage_h[3] + 7u) / 8u) * 8u;
    const uint32_t w3p = ((stage_w[3] + 7u) / 8u) * 8u;
    const uint32_t hs = h3p / 2u;
    const uint32_t ws = w3p / 2u;
    const uint32_t hsp = ((hs + 7u) / 8u) * 8u;
    const uint32_t wsp = ((ws + 7u) / 8u) * 8u;
    const uint32_t hg = hsp / 2u;
    const uint32_t wg = wsp / 2u;
    const uint32_t t_glob = hg * wg;

    // 1. Block 0 (Pre: adapter + 1h32 + full_skip + 2x2 pool):
    size_t b_idx = 0;
    {
      const PreparedBlockOffsets& pre = blocks[b_idx++];
      size_t pooled_dst = (topology.depths[0] == 1) ? s.skip_offs[0] : s.ping_off;
      KWindow1H32Params wp{H, W, 0, 0, 1u, 0, 0, 1u};
      uint32_t win_cnt = ((H + 7u) / 8u) * ((W + 7u) / 8u);
      [enc setBuffer:feat_buf offset:feat_off atIndex:0];
      [enc setBuffer:weights_buf offset:pre.adapter_off atIndex:1];
      [enc setBuffer:weights_buf offset:pre.ffn_w1_off atIndex:2];
      [enc setBuffer:weights_buf offset:pre.ffn_w2_off atIndex:3];
      [enc setBuffer:weights_buf offset:pre.ffn_cos_off atIndex:4];
      [enc setBuffer:weights_buf offset:pre.qkv_off atIndex:5];
      [enc setBuffer:weights_buf offset:pre.attn_scale_off atIndex:6];
      [enc setBuffer:weights_buf offset:pre.attn_bias_off atIndex:7];
      [enc setBuffer:weights_buf offset:pre.attn_proj_off atIndex:8];
      [enc setBuffer:weights_buf offset:pre.attn_cos_off atIndex:9];
      [enc setBuffer:scratch_buf offset:s.full_skip_off atIndex:10];
      [enc setBuffer:scratch_buf offset:pooled_dst atIndex:11];
      [enc setBytes:&wp length:sizeof(wp) atIndex:12];
      dispatch_1d_groups(pso_win_1h32_adapter_pool, win_cnt, 256);
      buf_barrier();
    }

    // 2. Encoder stages 0..3:
    for (int st = 0; st < 4; ++st) {
      const int reg_cnt = topology.depths[st] - 1;
      size_t cur = s.ping_off;
      size_t nxt = s.pong_off;
      for (int i = 0; i < reg_cnt; ++i) {
        const PreparedBlockOffsets& blk = blocks[b_idx++];
        size_t dst = (i == reg_cnt - 1) ? s.skip_offs[st] : nxt;
        emit_branched_block(blk, cur, dst, stage_h[st], stage_w[st], 1u);
        cur = dst;
        nxt = (cur == s.ping_off) ? s.pong_off : s.ping_off;
      }

      // Downsample transition block of stage `st`:
      const PreparedBlockOffsets& ds = blocks[b_idx++];
      uint32_t out_h = (st < 3) ? stage_h[st + 1] : hs;
      uint32_t out_w = (st < 3) ? stage_w[st + 1] : ws;
      size_t next_init_dst =
          (st < 3) ? ((topology.depths[st + 1] == 1) ? s.skip_offs[st + 1] : s.ping_off) : s.ping_off;

      if (st == 0) {
        // Fused 1h32 + 2x2 pool + E4M3:
        KWindow1H32Params wp{stage_h[0], stage_w[0], ds.pad_top, ds.pad_left, 0u, 0, 0, 0};
        uint32_t win_cnt = ((stage_h[0] + ds.pad_top + 7u) / 8u) * ((stage_w[0] + ds.pad_left + 7u) / 8u);
        [enc setBuffer:scratch_buf offset:s.skip_offs[0] atIndex:0];
        [enc setBuffer:weights_buf offset:ds.ffn_w1_off atIndex:1];
        [enc setBuffer:weights_buf offset:ds.ffn_w2_off atIndex:2];
        [enc setBuffer:weights_buf offset:ds.ffn_cos_off atIndex:3];
        [enc setBuffer:weights_buf offset:ds.qkv_off atIndex:4];
        [enc setBuffer:weights_buf offset:ds.attn_scale_off atIndex:5];
        [enc setBuffer:weights_buf offset:ds.attn_bias_off atIndex:6];
        [enc setBuffer:weights_buf offset:ds.attn_proj_off atIndex:7];
        [enc setBuffer:weights_buf offset:ds.attn_cos_off atIndex:8];
        [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:9];
        [enc setBytes:&wp length:sizeof(wp) atIndex:10];
        dispatch_1d_groups(pso_win_1h32_pool, win_cnt, 256);
        buf_barrier();
      } else {
        emit_branched_block(ds, s.skip_offs[st], s.pong_off, stage_h[st], stage_w[st], 0u);
        emit_pool2_pad(
            s.pong_off, s.tmp0_off, stage_h[st], stage_w[st], out_h, out_w, static_cast<uint32_t>(ds.channels), 1u);
      }
      emit_gemm(
          s.tmp0_off,
          ds.w0_off,
          0,
          0,
          next_init_dst,
          out_h * out_w,
          static_cast<uint32_t>(ds.channels * 2),
          static_cast<uint32_t>(ds.channels),
          EPILOGUE_E4M3);
    }

    // 3. Split Encoder stage (depths[4] blocks, last block also downsamples to global):
    {
      size_t cur = s.ping_off;
      size_t nxt = s.pong_off;
      const int sp_cnt = topology.depths[4];
      size_t transition_w0_off = 0;
      for (int i = 0; i < sp_cnt; ++i) {
        const PreparedBlockOffsets& blk = blocks[b_idx++];
        if (i == sp_cnt - 1) transition_w0_off = blk.w0_off;
        size_t dst = (i == sp_cnt - 1) ? s.split_skip_off : nxt;
        emit_split_block(blk, cur, dst, hs, ws);
        cur = dst;
        nxt = (cur == s.ping_off) ? s.pong_off : s.ping_off;
      }
      // Split downsample transition: pad8 + averagePool2 (no E4M3 on pool!) + GEMM + E4M3.
      emit_pool2_pad(s.split_skip_off, s.tmp0_off, hs, ws, hg, wg, 512u, 0u);
      emit_gemm(s.tmp0_off, transition_w0_off, 0, 0, s.ping_off, t_glob, 1024u, 512u, EPILOGUE_E4M3);
    }

    // 4. Global Attention stage (1024ch, 32 heads):
    size_t cur_off = s.ping_off;
    size_t nxt_off = s.pong_off;
    {
      const uint32_t pitch = ((t_glob + 7u) / 8u) * 8u;
      const uint32_t tiles = pitch / 8u;
      const uint32_t groups = (tiles + 3u) / 4u;
      KGlobalAttnParams gap{t_glob, pitch, tiles, groups, 32u};

      for (int g = 0; g < topology.global_block_count; ++g) {
        const PreparedBlockOffsets& blk = blocks[b_idx++];
        emit_gemm(cur_off, blk.ffn_w1_off, 0, 0, s.tmp0_off, t_glob, 4096u, 1024u, EPILOGUE_QUAD_GATE_E4M3);
        emit_gemm(
            s.tmp0_off, blk.ffn_w2_off, cur_off, blk.ffn_cos_off, s.tmp2_off, t_glob, 1024u, 4096u, EPILOGUE_COS_RES);
        emit_gemm(s.tmp2_off, blk.qkv_off, 0, 0, s.tmp0_off, t_glob, 3072u, 1024u, EPILOGUE_NONE);

        size_t attended_off = s.tmp0_off;
        if (t_glob <= 1024u) {
          size_t q_off = s.tmp1_off;
          size_t k_off = AlignUp(q_off + 32ull * pitch * 32ull * sizeof(uint16_t), kBufferAlign);
          size_t v_off = AlignUp(k_off + 32ull * pitch * 32ull * sizeof(uint16_t), kBufferAlign);

          [enc setComputePipelineState:pso_glob_pub_qkv];
          [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:0];
          [enc setBuffer:weights_buf offset:blk.attn_scale_off atIndex:1];
          [enc setBuffer:scratch_buf offset:q_off atIndex:2];
          [enc setBuffer:scratch_buf offset:k_off atIndex:3];
          [enc setBuffer:scratch_buf offset:v_off atIndex:4];
          [enc setBytes:&gap length:sizeof(gap) atIndex:5];
          [enc dispatchThreadgroups:MTLSizeMake(1, groups, 12)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          ++dispatches;
          buf_barrier();

          [enc setComputePipelineState:pso_glob_resident];
          [enc setBuffer:scratch_buf offset:q_off atIndex:0];
          [enc setBuffer:scratch_buf offset:k_off atIndex:1];
          [enc setBuffer:scratch_buf offset:v_off atIndex:2];
          [enc setBuffer:scratch_buf offset:attended_off atIndex:3];
          [enc setBytes:&gap length:sizeof(gap) atIndex:4];
          [enc dispatchThreadgroups:MTLSizeMake(1, tiles, 32)
              threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
          ++dispatches;
          buf_barrier();
        } else {
          size_t q_off = s.tmp1_off;
          size_t k_off = AlignUp(q_off + 32ull * t_glob * 32ull * sizeof(uint16_t), kBufferAlign);
          size_t v_off = AlignUp(k_off + 32ull * t_glob * 32ull * sizeof(uint16_t), kBufferAlign);
          size_t r_off = s.tmp0_off;
          attended_off = AlignUp(r_off + 32ull * t_glob * sizeof(uint16_t), kBufferAlign);

          [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:0];
          [enc setBuffer:weights_buf offset:blk.attn_scale_off atIndex:1];
          [enc setBuffer:scratch_buf offset:q_off atIndex:2];
          [enc setBuffer:scratch_buf offset:k_off atIndex:3];
          [enc setBuffer:scratch_buf offset:v_off atIndex:4];
          [enc setBytes:&gap length:sizeof(gap) atIndex:5];
          dispatch_threads_1d(pso_glob_split_norm, t_glob * 32u * 3u);
          buf_barrier();

          uint32_t row_tiles = (t_glob + 7u) / 8u;
          [enc setComputePipelineState:pso_glob_denom];
          [enc setBuffer:scratch_buf offset:q_off atIndex:0];
          [enc setBuffer:scratch_buf offset:k_off atIndex:1];
          [enc setBuffer:scratch_buf offset:r_off atIndex:2];
          [enc setBytes:&gap length:sizeof(gap) atIndex:3];
          [enc dispatchThreadgroups:MTLSizeMake(1, row_tiles, 32)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          ++dispatches;
          buf_barrier();

          [enc setComputePipelineState:pso_glob_values];
          [enc setBuffer:scratch_buf offset:q_off atIndex:0];
          [enc setBuffer:scratch_buf offset:k_off atIndex:1];
          [enc setBuffer:scratch_buf offset:v_off atIndex:2];
          [enc setBuffer:scratch_buf offset:r_off atIndex:3];
          [enc setBuffer:scratch_buf offset:attended_off atIndex:4];
          [enc setBytes:&gap length:sizeof(gap) atIndex:5];
          [enc dispatchThreadgroups:MTLSizeMake(1, row_tiles, 32)
              threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
          ++dispatches;
          buf_barrier();
        }

        emit_gemm(
            attended_off,
            blk.attn_proj_off,
            s.tmp2_off,
            blk.attn_cos_off,
            nxt_off,
            t_glob,
            1024u,
            1024u,
            EPILOGUE_COS_RES_E4M3);
        std::swap(cur_off, nxt_off);
      }
    }

    // 5. Bridge block (1024 -> 512 + upsample merge with split_skip):
    {
      const PreparedBlockOffsets& bridge = blocks[b_idx++];
      emit_gemm(cur_off, bridge.w0_off, 0, 0, s.tmp0_off, t_glob, 512u, 1024u, EPILOGUE_NONE);
      emit_upsample_merge(
          s.tmp0_off, s.split_skip_off, bridge.sin_off, bridge.sin_off, nxt_off, hs, ws, 512u, wg, 0u, 1u);
      std::swap(cur_off, nxt_off);
    }

    // 6. Split Decoder stage (depths[4] blocks):
    for (int i = 0; i < topology.depths[4]; ++i) {
      const PreparedBlockOffsets& blk = blocks[b_idx++];
      emit_split_block(blk, cur_off, nxt_off, hs, ws);
      std::swap(cur_off, nxt_off);
    }

    // 7. Decoder stages 3..0 (256, 128, 64, 32):
    uint32_t low_h = hs;
    uint32_t low_w = ws;
    for (int st = 3; st >= 0; --st) {
      const uint32_t h = stage_h[st];
      const uint32_t w = stage_w[st];
      const PreparedBlockOffsets& up_blk = blocks[b_idx++];
      const uint32_t C = static_cast<uint32_t>(up_blk.channels);

      emit_gemm(cur_off, up_blk.w0_off, 0, 0, s.tmp0_off, low_h * low_w, C, C * 2u, EPILOGUE_NONE);
      if (st == 0) {
        // Fused 1h32 merge + window block + store:
        KWindow1H32Params wp{h, w, up_blk.pad_top, up_blk.pad_left, 1u, low_w, 2u, 0u};
        uint32_t win_cnt = ((h + up_blk.pad_top + 7u) / 8u) * ((w + up_blk.pad_left + 7u) / 8u);
        [enc setBuffer:scratch_buf offset:s.tmp0_off atIndex:0];
        [enc setBuffer:scratch_buf offset:s.skip_offs[0] atIndex:1];
        [enc setBuffer:weights_buf offset:up_blk.sin_off atIndex:2];
        [enc setBuffer:weights_buf offset:up_blk.sin_off atIndex:3];
        [enc setBuffer:weights_buf offset:up_blk.ffn_w1_off atIndex:4];
        [enc setBuffer:weights_buf offset:up_blk.ffn_w2_off atIndex:5];
        [enc setBuffer:weights_buf offset:up_blk.ffn_cos_off atIndex:6];
        [enc setBuffer:weights_buf offset:up_blk.qkv_off atIndex:7];
        [enc setBuffer:weights_buf offset:up_blk.attn_scale_off atIndex:8];
        [enc setBuffer:weights_buf offset:up_blk.attn_bias_off atIndex:9];
        [enc setBuffer:weights_buf offset:up_blk.attn_proj_off atIndex:10];
        [enc setBuffer:weights_buf offset:up_blk.attn_cos_off atIndex:11];
        [enc setBuffer:scratch_buf offset:nxt_off atIndex:12];
        [enc setBytes:&wp length:sizeof(wp) atIndex:13];
        dispatch_1d_groups(pso_win_1h32_merge_store, win_cnt, 256);
        buf_barrier();
        std::swap(cur_off, nxt_off);
      } else {
        emit_upsample_merge(
            s.tmp0_off, s.skip_offs[st], up_blk.sin_off, up_blk.sin_off, nxt_off, h, w, C, low_w, 0u, 1u);
        emit_branched_block(up_blk, nxt_off, cur_off, h, w, 1u);
      }

      for (int j = 1; j < topology.depths[st]; ++j) {
        const PreparedBlockOffsets& reg_blk = blocks[b_idx++];
        emit_branched_block(reg_blk, cur_off, nxt_off, h, w, 1u);
        std::swap(cur_off, nxt_off);
      }
      low_h = h;
      low_w = w;
    }

    // 8. Post block (upsample merge + 1h32 + [32, 8] head -> [1, H, W, 4] half):
    {
      const PreparedBlockOffsets& post = blocks[b_idx++];
      KWindow1H32Params wp{H, W, post.pad_top, post.pad_left, 0u, low_w, 1u, 0u};
      uint32_t win_cnt = ((H + post.pad_top + 7u) / 8u) * ((W + post.pad_left + 7u) / 8u);
      [enc setBuffer:scratch_buf offset:cur_off atIndex:0];
      [enc setBuffer:scratch_buf offset:s.full_skip_off atIndex:1];
      [enc setBuffer:weights_buf offset:post.sin_off atIndex:2];
      [enc setBuffer:weights_buf offset:post.cos_merge_off atIndex:3];
      [enc setBuffer:weights_buf offset:post.ffn_w1_off atIndex:4];
      [enc setBuffer:weights_buf offset:post.ffn_w2_off atIndex:5];
      [enc setBuffer:weights_buf offset:post.ffn_cos_off atIndex:6];
      [enc setBuffer:weights_buf offset:post.qkv_off atIndex:7];
      [enc setBuffer:weights_buf offset:post.attn_scale_off atIndex:8];
      [enc setBuffer:weights_buf offset:post.attn_bias_off atIndex:9];
      [enc setBuffer:weights_buf offset:post.attn_proj_off atIndex:10];
      [enc setBuffer:weights_buf offset:post.attn_cos_off atIndex:11];
      [enc setBuffer:weights_buf offset:post.head_weight_off atIndex:12];
      [enc setBuffer:head_buf offset:head_off atIndex:13];
      [enc setBytes:&wp length:sizeof(wp) atIndex:14];
      dispatch_1d_groups(pso_win_1h32_merge_head, win_cnt, 256);
    }
  }
};

// ============================================================================
// Context Public Methods
// ============================================================================

Context::Context(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Context::~Context() = default;

std::unique_ptr<Context> Context::LoadFromPath(
    NRMetalDevice device,
    const std::string& model_path,
    const LoadOptions& options,
    std::string* out_error) {
  @autoreleasepool {
    if (!device) {
      if (out_error) *out_error = "Metal device is null";
      return nullptr;
    }
    ParsedSafeTensors parsed;
    SafeTensorSummary summary;
    if (!LoadAndValidatePackageOrSafeTensors(
            model_path,
            options.reject_symlinks,
            options.verify_manifest_sha256,
            &parsed,
            &summary,
            out_error)) {
      return nullptr;
    }
    if (!InferTopologyFromTensors(parsed, &summary.topology, out_error)) {
      return nullptr;
    }

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->topology = summary.topology;
    impl->synthetic = false;
    impl->use_residency_set = options.use_residency_set;
    impl->weight_format = summary.source_format;

    if (!impl->InitPipelines(options.metallib_path, out_error)) {
      return nullptr;
    }

    SafeTensorProvider provider(parsed);
    std::vector<uint8_t> packed;
    if (!PrepareModelWeights(impl->topology, provider, &impl->blocks, &packed, out_error)) {
      return nullptr;
    }
    if (!impl->UploadWeights(packed, out_error)) {
      return nullptr;
    }
    return std::unique_ptr<Context>(new Context(std::move(impl)));
  }
}

std::unique_ptr<Context> Context::PrepareSynthetic(
    NRMetalDevice device,
    TopologyKind kind,
    const SyntheticOptions& options,
    std::string* out_error) {
  @autoreleasepool {
    if (!device) {
      if (out_error) *out_error = "Metal device is null";
      return nullptr;
    }
    const int ht_depths[5] = {4, 4, 6, 8, 8};
    const int sm_depths[5] = {2, 2, 2, 2, 2};
    TopologySpec spec = (kind == TopologyKind::Small31)
                            ? MakeTopologySpec(sm_depths, 8, kSmallBlendScale)
                            : MakeTopologySpec(ht_depths, 8, kDefaultBlendScale);

    auto impl = std::make_unique<Impl>();
    impl->device = device;
    impl->topology = spec;
    impl->synthetic = true;
    impl->use_residency_set = options.use_residency_set;
    impl->weight_format = "synthetic-fp16";

    if (!impl->InitPipelines(options.metallib_path, out_error)) {
      return nullptr;
    }

    SyntheticTensorProvider provider(options.seed, options.weight_scale);
    std::vector<uint8_t> packed;
    if (!PrepareModelWeights(impl->topology, provider, &impl->blocks, &packed, out_error)) {
      return nullptr;
    }
    if (!impl->UploadWeights(packed, out_error)) {
      return nullptr;
    }
    return std::unique_ptr<Context>(new Context(std::move(impl)));
  }
}

const TopologySpec& Context::topology() const { return impl_->topology; }
bool Context::is_synthetic() const { return impl_->synthetic; }
const std::string& Context::weight_format() const { return impl_->weight_format; }
size_t Context::weight_buffer_bytes() const { return impl_->weights_bytes; }
size_t Context::scratch_buffer_bytes() const {
  return impl_->scratch_buf ? [impl_->scratch_buf length] : 0;
}
NRMetalTexture Context::history_texture() const {
  return impl_->history_tex[impl_->active_history_idx];
}

void Context::reset_history() {
  impl_->history_valid = false;
}

bool Context::encode(
    const FrameInputs& inputs,
    NRMetalCommandBuffer cmd,
    FrameTicket& out_ticket,
    Report& out_report,
    std::string* out_error) {
  @autoreleasepool {
    const auto t0 = std::chrono::steady_clock::now();
    if (!inputs.color_texture || !cmd) {
      if (out_error) *out_error = "color_texture and command buffer are required";
      return false;
    }
    if (!inputs.output_texture && !inputs.export_delta_buffer && !inputs.export_baked_delta_texture) {
      if (out_error) {
        *out_error = "output_texture, export_delta_buffer, or export_baked_delta_texture is required";
      }
      return false;
    }
    if (inputs.output_texture && inputs.color_texture == inputs.output_texture) {
      if (out_error) *out_error = "color_texture and output_texture must be distinct textures";
      return false;
    }

    const uint32_t in_w = static_cast<uint32_t>([inputs.color_texture width]);
    const uint32_t in_h = static_cast<uint32_t>([inputs.color_texture height]);
    const uint32_t out_w = inputs.params.output_width > 0
                               ? inputs.params.output_width
                               : (inputs.output_texture ? static_cast<uint32_t>([inputs.output_texture width]) : in_w);
    const uint32_t out_h = inputs.params.output_height > 0
                               ? inputs.params.output_height
                               : (inputs.output_texture ? static_cast<uint32_t>([inputs.output_texture height]) : in_h);

    if (out_w == 0 || out_h == 0 ||
        (inputs.output_texture &&
         (out_w > [inputs.output_texture width] || out_h > [inputs.output_texture height]))) {
      if (out_error) *out_error = "invalid output dimensions";
      return false;
    }

    // Resolve internal network resolution & geometry mode.
    uint32_t net_w = inputs.params.network_width;
    uint32_t net_h = inputs.params.network_height;
    bool scaled_net = false;
    if (net_w == 0 || net_h == 0) {
      auto align64 = [](uint32_t v) -> uint32_t {
        uint32_t m = std::max<uint32_t>(320u, v);
        return ((m + 63u) / 64u) * 64u;
      };
      net_w = align64(in_w);
      net_h = align64(in_h);
      scaled_net = (out_w != in_w || out_h != in_h);
    } else {
      if (net_w < 128u || net_h < 128u || (net_w % 32u) != 0u || (net_h % 32u) != 0u) {
        if (out_error) {
          *out_error = "network_width and network_height must be multiples of 32 >= 128";
        }
        return false;
      }
      if (inputs.params.geometry_mode == GeometryMode::ScaledNetwork || net_w < in_w || net_h < in_h ||
          out_w != in_w || out_h != in_h) {
        scaled_net = true;
      } else {
        scaled_net = false;
      }
    }

    const uint32_t hist_w = scaled_net ? net_w : in_w;
    const uint32_t hist_h = scaled_net ? net_h : in_h;
    const size_t delta_plane_bytes = static_cast<size_t>(hist_w) * hist_h * 4u * sizeof(float);
    if (inputs.export_delta_buffer && [inputs.export_delta_buffer length] < delta_plane_bytes) {
      if (out_error) *out_error = "export_delta_buffer is smaller than hist_w * hist_h * sizeof(float4)";
      return false;
    }
    if (inputs.export_baked_delta_texture &&
        ([inputs.export_baked_delta_texture width] < hist_w ||
         [inputs.export_baked_delta_texture height] < hist_h)) {
      if (out_error) {
        *out_error = "export_baked_delta_texture is smaller than hist_w x hist_h";
      }
      return false;
    }

    if (!impl_->EnsureScratch(net_w, net_h, inputs.output_texture ? out_w : 0u, inputs.output_texture ? out_h : 0u, out_error) ||
        !impl_->EnsureHistoryTextures(hist_w, hist_h, out_error)) {
      return false;
    }

    if (@available(macOS 15.0, *)) {
      if (impl_->residency_set_obj && [cmd respondsToSelector:@selector(useResidencySet:)]) {
        [cmd useResidencySet:(id<MTLResidencySet>)impl_->residency_set_obj];
      }
    }

    const bool use_temporal = (!inputs.params.reset && impl_->history_valid);
    const int prev_hist_idx = impl_->active_history_idx;
    const int next_hist_idx = 1 - prev_hist_idx;

    int style_idx = 0;
    float tone = 1.0f;
    float structure = 1.0f;
    switch (inputs.params.profile) {
      case ControlProfile::Standard: style_idx = 0; tone = 1.0f; structure = 1.0f; break;
      case ControlProfile::Natural: style_idx = 1; tone = 1.0f; structure = 1.0f; break;
      case ControlProfile::Cinematic: style_idx = 2; tone = 1.0f; structure = 1.0f; break;
      case ControlProfile::Neutral: style_idx = 0; tone = 0.0f; structure = 0.0f; break;
    }
    if (inputs.params.style_index_override >= 0) style_idx = inputs.params.style_index_override;
    if (inputs.params.local_tone_override >= 0.0f) tone = inputs.params.local_tone_override;
    if (inputs.params.local_structure_override >= 0.0f) structure = inputs.params.local_structure_override;

    const float detail_s = inputs.params.detail_strength;
    const float colour_s = inputs.params.colour_strength;
    const float intensity = inputs.params.intensity;
    const bool direct_write =
        (inputs.export_delta_buffer == nil && inputs.export_baked_delta_texture == nil &&
         inputs.output_texture != nil &&
         !scaled_net && out_w == in_w && out_h == in_h && detail_s == 1.0f && colour_s == 1.0f);

    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!enc) {
      if (out_error) *out_error = "failed to create MTLComputeCommandEncoder";
      return false;
    }

    uint32_t dispatches = 0;
    const ScratchArenaLayout& s = impl_->scratch_layout;

    auto dispatch_2d = [&](id<MTLComputePipelineState> pso, uint32_t w, uint32_t h) {
      [enc setComputePipelineState:pso];
      MTLSize tpg = MTLSizeMake(16, 16, 1);
      MTLSize groups = MTLSizeMake((w + 15u) / 16u, (h + 15u) / 16u, 1);
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
      ++dispatches;
    };

    // Stage A: GPU Feature Preprocessor + Catmull-Rom Temporal Reprojection.
    KPreFeatureParams pfp{};
    pfp.in_width = in_w;
    pfp.in_height = in_h;
    pfp.net_width = net_w;
    pfp.net_height = net_h;
    pfp.logical_width = in_w;
    pfp.logical_height = in_h;
    pfp.frame_index = inputs.params.frame_index;
    pfp.scaled_network = scaled_net ? 1u : 0u;
    pfp.has_history = use_temporal ? 1u : 0u;
    pfp.has_motion = (use_temporal && inputs.motion_texture != nil) ? 1u : 0u;
    pfp.has_depth =
        (use_temporal && inputs.depth_texture != nil && inputs.params.use_closest_depth) ? 1u : 0u;
    pfp.depth_inverted = inputs.params.depth_inverted ? 1u : 0u;
    pfp.srgb_transfer = inputs.params.srgb_transfer ? 1u : 0u;
    pfp.normalized_style = static_cast<float>(style_idx) / 128.0f;
    pfp.local_tone = tone;
    pfp.local_structure = structure;
    pfp.motion_scale_x = inputs.params.motion_scale_x;
    pfp.motion_scale_y = inputs.params.motion_scale_y;

    [enc setTexture:inputs.color_texture atIndex:0];
    [enc setTexture:(use_temporal ? impl_->history_tex[prev_hist_idx] : impl_->dummy_tex) atIndex:1];
    [enc setTexture:(inputs.motion_texture ? inputs.motion_texture : impl_->dummy_tex) atIndex:2];
    [enc setTexture:(inputs.depth_texture ? inputs.depth_texture : impl_->dummy_tex) atIndex:3];
    [enc setBuffer:impl_->scratch_buf offset:s.feat_f16_off atIndex:0];
    [enc setBuffer:impl_->scratch_buf offset:s.base_net_rgb_off atIndex:1];
    [enc setBytes:&pfp length:sizeof(pfp) atIndex:2];
    dispatch_2d(impl_->pso_pre_features, net_w, net_h);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Stage B: 31-block or 71-block Transformer U-Net Core.
    impl_->EncodeUNetCore(
        enc, impl_->scratch_buf, s.feat_f16_off, impl_->scratch_buf, s.head_f16_off, net_h, net_w, dispatches);
    [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

    // Stage C: Temporal Blend + History Update (+ Direct Output if 1:1 no-split).
    KPostTemporalParams ptp{};
    ptp.net_width = net_w;
    ptp.net_height = net_h;
    ptp.hist_width = hist_w;
    ptp.hist_height = hist_h;
    ptp.out_width = out_w;
    ptp.out_height = out_h;
    ptp.scaled_network = scaled_net ? 1u : 0u;
    ptp.has_history = use_temporal ? 1u : 0u;
    ptp.direct_write_output = direct_write ? 1u : 0u;
    ptp.srgb_transfer = inputs.params.srgb_transfer ? 1u : 0u;
    ptp.blend_scale = impl_->topology.blend_scale;
    ptp.intensity = intensity;

    id<MTLBuffer> delta_buf = inputs.export_delta_buffer ? inputs.export_delta_buffer : impl_->scratch_buf;
    const size_t delta_off = inputs.export_delta_buffer ? 0u : s.delta_net_rgb_off;

    [enc setBuffer:impl_->scratch_buf offset:s.head_f16_off atIndex:0];
    [enc setBuffer:impl_->scratch_buf offset:s.feat_f16_off atIndex:1];
    [enc setBuffer:impl_->scratch_buf offset:s.base_net_rgb_off atIndex:2];
    [enc setBuffer:delta_buf offset:delta_off atIndex:3];
    [enc setBytes:&ptp length:sizeof(ptp) atIndex:4];
    [enc setTexture:impl_->history_tex[next_hist_idx] atIndex:0];
    [enc setTexture:(direct_write ? inputs.output_texture : impl_->dummy_tex) atIndex:1];
    dispatch_2d(impl_->pso_post_temporal, hist_w, hist_h);

    // Stage C.2: Optional Worker-Side Pre-Baked RGBA16Float Delta Texture at (hist_w, hist_h).
    // Reuses `s.base_net_rgb_off` (no longer read after post_temporal) for horizontal blur scratch,
    // enabling single-pass (<0.4 ms) hardware-bilinear composition on 60–120 Hz display queues.
    if (inputs.export_baked_delta_texture != nil) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      const bool needs_split_blur = (std::abs(detail_s - colour_s) >= 1e-6f);
      uint32_t bake_radius = 0;
      std::vector<float> bake_weights = {1.0f};
      if (needs_split_blur) {
        const float ratio = static_cast<float>(hist_w) / static_cast<float>(std::max<uint32_t>(1u, out_w));
        const float net_sigma = std::clamp(inputs.params.detail_radius * ratio, 0.5f, 10.6f);
        bake_weights = BuildGaussianWeights(net_sigma, &bake_radius);
      }

      KDetailComposeParams bp{};
      bp.net_width = hist_w;
      bp.net_height = hist_h;
      bp.out_width = out_w;
      bp.out_height = out_h;
      bp.radius = bake_radius;
      bp.scaled_network = scaled_net ? 1u : 0u;
      bp.srgb_transfer = inputs.params.srgb_transfer ? 1u : 0u;
      bp.detail_strength = detail_s;
      bp.colour_strength = colour_s;
      bp.intensity = intensity;

      if (bake_radius > 0u) {
        [enc setBuffer:delta_buf offset:delta_off atIndex:0];
        [enc setBuffer:impl_->scratch_buf offset:s.base_net_rgb_off atIndex:1];
        [enc setBytes:&bp length:sizeof(bp) atIndex:2];
        [enc setBytes:bake_weights.data() length:bake_weights.size() * sizeof(float) atIndex:3];
        dispatch_2d(impl_->pso_bake_delta_blur_h, hist_w, hist_h);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      }

      [enc setBuffer:delta_buf offset:delta_off atIndex:0];
      [enc setBuffer:impl_->scratch_buf offset:s.base_net_rgb_off atIndex:1];
      [enc setTexture:inputs.export_baked_delta_texture atIndex:0];
      [enc setBytes:&bp length:sizeof(bp) atIndex:2];
      [enc setBytes:bake_weights.data() length:bake_weights.size() * sizeof(float) atIndex:3];
      dispatch_2d(impl_->pso_bake_delta_tex, hist_w, hist_h);
    }

    // Stage D: Optional Upsample & Separable Gaussian Detail-Split Composition.
    if (!direct_write && inputs.output_texture != nil) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      uint32_t radius = 12;
      std::vector<float> gweights = BuildGaussianWeights(
          std::clamp(inputs.params.detail_radius, 0.5f, 10.6f), &radius);

      KDetailComposeParams dcp{};
      dcp.net_width = hist_w;
      dcp.net_height = hist_h;
      dcp.out_width = out_w;
      dcp.out_height = out_h;
      dcp.radius = radius;
      dcp.scaled_network = scaled_net ? 1u : 0u;
      dcp.srgb_transfer = inputs.params.srgb_transfer ? 1u : 0u;
      dcp.detail_strength = detail_s;
      dcp.colour_strength = colour_s;
      dcp.intensity = intensity;

      if (std::abs(detail_s - colour_s) < 1e-6f) {
        [enc setTexture:inputs.color_texture atIndex:0];
        [enc setBuffer:delta_buf offset:delta_off atIndex:0];
        [enc setTexture:inputs.output_texture atIndex:1];
        [enc setBytes:&dcp length:sizeof(dcp) atIndex:1];
        dispatch_2d(impl_->pso_upsample_compose, out_w, out_h);
      } else {
        [enc setBuffer:delta_buf offset:delta_off atIndex:0];
        [enc setBuffer:impl_->scratch_buf offset:s.delta_full_rgb_off atIndex:1];
        [enc setBuffer:impl_->scratch_buf offset:s.blur_h_rgb_off atIndex:2];
        [enc setBytes:&dcp length:sizeof(dcp) atIndex:3];
        [enc setBytes:gweights.data() length:gweights.size() * sizeof(float) atIndex:4];
        dispatch_2d(impl_->pso_detail_blur_h, out_w, out_h);
        [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

        [enc setTexture:inputs.color_texture atIndex:0];
        [enc setBuffer:impl_->scratch_buf offset:s.delta_full_rgb_off atIndex:0];
        [enc setBuffer:impl_->scratch_buf offset:s.blur_h_rgb_off atIndex:1];
        [enc setTexture:inputs.output_texture atIndex:1];
        [enc setBytes:&dcp length:sizeof(dcp) atIndex:2];
        [enc setBytes:gweights.data() length:gweights.size() * sizeof(float) atIndex:3];
        dispatch_2d(impl_->pso_detail_blur_v, out_w, out_h);
      }
    }

    [enc endEncoding];

    impl_->active_history_idx = next_hist_idx;
    impl_->history_valid = true;

    out_ticket = FrameTicket{};
    auto st = out_ticket.state;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      st->gpu_start_s = cb.GPUStartTime;
      st->gpu_end_s = cb.GPUEndTime;
      st->gpu_ms = (cb.GPUEndTime > cb.GPUStartTime && cb.GPUStartTime > 0.0)
                       ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0
                       : 0.0;
      bool ok = (cb.status == MTLCommandBufferStatusCompleted);
      if (!ok && cb.error) {
        st->error_message = [[cb.error localizedDescription] UTF8String];
      }
      st->succeeded.store(ok, std::memory_order_release);
      st->completed.store(true, std::memory_order_release);
    }];

    const auto t1 = std::chrono::steady_clock::now();
    out_report = Report{};
    out_report.output_width = out_w;
    out_report.output_height = out_h;
    out_report.network_width = net_w;
    out_report.network_height = net_h;
    out_report.block_count = static_cast<uint32_t>(impl_->topology.total_blocks);
    out_report.dispatch_count = dispatches;
    out_report.cpu_encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out_report.blend_scale = impl_->topology.blend_scale;
    out_report.temporal_active = use_temporal;
    out_report.synthetic_weights = impl_->synthetic;
    out_report.topology = impl_->topology.name + " (" + impl_->topology.spec_string + ")";
    out_report.weight_format = impl_->weight_format;
    return true;
  }
}

bool Context::run_sync(
    const FrameInputs& inputs,
    NRMetalCommandQueue queue,
    Report& out_report,
    std::string* out_error) {
  @autoreleasepool {
    id<MTLCommandQueue> q = queue ? queue : impl_->internal_queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    if (!cmd) {
      if (out_error) *out_error = "failed to create MTLCommandBuffer";
      return false;
    }
    FrameTicket ticket;
    if (!encode(inputs, cmd, ticket, out_report, out_error)) {
      return false;
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      if (out_error) {
        *out_error = cmd.error ? std::string([[cmd.error localizedDescription] UTF8String])
                               : "GPU command buffer failed";
      }
      return false;
    }
    out_report.gpu_ms = (cmd.GPUEndTime > cmd.GPUStartTime && cmd.GPUStartTime > 0.0)
                            ? (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0
                            : ticket.gpu_ms();
    return true;
  }
}

bool Context::encode_compose_delta(
    NRMetalTexture color_texture,
    NRMetalBuffer delta_net_buffer,
    uint32_t delta_width,
    uint32_t delta_height,
    const FrameParams& params,
    NRMetalTexture output_texture,
    NRMetalCommandBuffer cmd,
    std::string* out_error) {
  @autoreleasepool {
    if (!color_texture || !delta_net_buffer || !output_texture || !cmd) {
      if (out_error) {
        *out_error = "color_texture, delta_net_buffer, output_texture, and cmd are required";
      }
      return false;
    }
    if (color_texture == output_texture) {
      if (out_error) *out_error = "color_texture and output_texture must be distinct textures";
      return false;
    }
    if (delta_width == 0 || delta_height == 0) {
      if (out_error) *out_error = "delta_width and delta_height must be non-zero";
      return false;
    }
    const size_t required_delta_bytes =
        static_cast<size_t>(delta_width) * delta_height * 4u * sizeof(float);
    if ([delta_net_buffer length] < required_delta_bytes) {
      if (out_error) *out_error = "delta_net_buffer is smaller than delta_width * delta_height * sizeof(float4)";
      return false;
    }

    const uint32_t in_w = static_cast<uint32_t>([color_texture width]);
    const uint32_t in_h = static_cast<uint32_t>([color_texture height]);
    const uint32_t out_w =
        params.output_width > 0 ? params.output_width : static_cast<uint32_t>([output_texture width]);
    const uint32_t out_h =
        params.output_height > 0 ? params.output_height : static_cast<uint32_t>([output_texture height]);
    if (out_w == 0 || out_h == 0 || out_w > [output_texture width] || out_h > [output_texture height]) {
      if (out_error) *out_error = "invalid output dimensions in encode_compose_delta";
      return false;
    }

    const bool scaled_net =
        (delta_width != out_w || delta_height != out_h ||
         out_w != in_w || out_h != in_h ||
         params.geometry_mode == GeometryMode::ScaledNetwork ||
         (params.network_width > 0 && params.network_width < in_w) ||
         (params.network_height > 0 && params.network_height < in_h));

    const float detail_s = params.detail_strength;
    const float colour_s = params.colour_strength;
    const bool needs_blur = (std::abs(detail_s - colour_s) >= 1e-6f);

    id<MTLBuffer> compose_scratch = nil;
    size_t blur_h_off = 0;
    if (needs_blur) {
      if (!impl_->EnsureAsyncComposeScratch(out_w, out_h, &compose_scratch, &blur_h_off, out_error)) {
        return false;
      }
    }

    uint32_t radius = 12;
    std::vector<float> gweights;
    if (needs_blur) {
      gweights = BuildGaussianWeights(std::clamp(params.detail_radius, 0.5f, 10.6f), &radius);
    }

    KDetailComposeParams dcp{};
    dcp.net_width = delta_width;
    dcp.net_height = delta_height;
    dcp.out_width = out_w;
    dcp.out_height = out_h;
    dcp.radius = radius;
    dcp.scaled_network = scaled_net ? 1u : 0u;
    dcp.srgb_transfer = params.srgb_transfer ? 1u : 0u;
    dcp.detail_strength = detail_s;
    dcp.colour_strength = colour_s;
    dcp.intensity = params.intensity;

    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!enc) {
      if (out_error) *out_error = "failed to create MTLComputeCommandEncoder in encode_compose_delta";
      return false;
    }
    enc.label = @"Netal NR Cached Delta Compose";

    const MTLSize tpg = MTLSizeMake(16, 16, 1);
    const MTLSize groups = MTLSizeMake((out_w + 15u) / 16u, (out_h + 15u) / 16u, 1);

    if (!needs_blur) {
      [enc setComputePipelineState:impl_->pso_upsample_compose];
      [enc setTexture:color_texture atIndex:0];
      [enc setBuffer:delta_net_buffer offset:0 atIndex:0];
      [enc setTexture:output_texture atIndex:1];
      [enc setBytes:&dcp length:sizeof(dcp) atIndex:1];
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
    } else {
      [enc setComputePipelineState:impl_->pso_detail_blur_h];
      [enc setBuffer:delta_net_buffer offset:0 atIndex:0];
      [enc setBuffer:compose_scratch offset:0 atIndex:1];
      [enc setBuffer:compose_scratch offset:blur_h_off atIndex:2];
      [enc setBytes:&dcp length:sizeof(dcp) atIndex:3];
      [enc setBytes:gweights.data() length:gweights.size() * sizeof(float) atIndex:4];
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];

      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];

      [enc setComputePipelineState:impl_->pso_detail_blur_v];
      [enc setTexture:color_texture atIndex:0];
      [enc setBuffer:compose_scratch offset:0 atIndex:0];
      [enc setBuffer:compose_scratch offset:blur_h_off atIndex:1];
      [enc setTexture:output_texture atIndex:1];
      [enc setBytes:&dcp length:sizeof(dcp) atIndex:2];
      [enc setBytes:gweights.data() length:gweights.size() * sizeof(float) atIndex:3];
      [enc dispatchThreadgroups:groups threadsPerThreadgroup:tpg];
    }

    [enc endEncoding];
    return true;
  }
}

bool Context::encode_compose_baked_delta(
    NRMetalTexture color_texture,
    NRMetalTexture baked_delta_texture,
    const FrameParams& params,
    NRMetalTexture output_texture,
    NRMetalCommandBuffer cmd,
    std::string* out_error) {
  @autoreleasepool {
    if (!color_texture || !baked_delta_texture || !output_texture || !cmd) {
      if (out_error) {
        *out_error = "color_texture, baked_delta_texture, output_texture, and cmd are required";
      }
      return false;
    }
    if (color_texture == output_texture) {
      if (out_error) *out_error = "color_texture and output_texture must be distinct textures";
      return false;
    }
    const uint32_t out_w =
        params.output_width > 0 ? params.output_width : static_cast<uint32_t>([output_texture width]);
    const uint32_t out_h =
        params.output_height > 0 ? params.output_height : static_cast<uint32_t>([output_texture height]);
    if (out_w == 0 || out_h == 0 || out_w > [output_texture width] || out_h > [output_texture height]) {
      if (out_error) *out_error = "invalid output dimensions in encode_compose_baked_delta";
      return false;
    }

    KDetailComposeParams dcp{};
    dcp.net_width = static_cast<uint32_t>([baked_delta_texture width]);
    dcp.net_height = static_cast<uint32_t>([baked_delta_texture height]);
    dcp.out_width = out_w;
    dcp.out_height = out_h;
    dcp.radius = 0;
    dcp.scaled_network = 1u;
    dcp.srgb_transfer = params.srgb_transfer ? 1u : 0u;
    dcp.detail_strength = params.detail_strength;
    dcp.colour_strength = params.colour_strength;
    dcp.intensity = params.intensity;

    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!enc) {
      if (out_error) *out_error = "failed to create MTLComputeCommandEncoder in encode_compose_baked_delta";
      return false;
    }
    enc.label = @"Netal NR Baked Delta Compose";
    [enc setComputePipelineState:impl_->pso_compose_baked_tex];
    [enc setTexture:color_texture atIndex:0];
    [enc setTexture:baked_delta_texture atIndex:1];
    [enc setTexture:output_texture atIndex:2];
    [enc setBytes:&dcp length:sizeof(dcp) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake((out_w + 15u) / 16u, (out_h + 15u) / 16u, 1)
        threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
    [enc endEncoding];
    return true;
  }
}

bool Context::encode_tensor(
    NRMetalBuffer input_nhwc16,
    bool input_is_fp16,
    uint32_t height,
    uint32_t width,
    NRMetalBuffer output_nhwc4,
    bool output_is_fp16,
    NRMetalCommandBuffer cmd,
    FrameTicket& out_ticket,
    Report& out_report,
    std::string* out_error) {
  @autoreleasepool {
    const auto t0 = std::chrono::steady_clock::now();
    if (!input_nhwc16 || !output_nhwc4 || !cmd) {
      if (out_error) *out_error = "input_nhwc16, output_nhwc4, and cmd must be non-null";
      return false;
    }
    if (height < 128u || width < 128u || (height % 32u) != 0u || (width % 32u) != 0u) {
      if (out_error) *out_error = "height and width must be multiples of 32 >= 128";
      return false;
    }
    if (!impl_->EnsureScratch(width, height, 0, 0, out_error)) {
      return false;
    }

    if (@available(macOS 15.0, *)) {
      if (impl_->residency_set_obj && [cmd respondsToSelector:@selector(useResidencySet:)]) {
        [cmd useResidencySet:(id<MTLResidencySet>)impl_->residency_set_obj];
      }
    }

    id<MTLComputeCommandEncoder> enc =
        [cmd computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    if (!enc) {
      if (out_error) *out_error = "failed to create MTLComputeCommandEncoder";
      return false;
    }

    uint32_t dispatches = 0;
    const ScratchArenaLayout& s = impl_->scratch_layout;

    id<MTLBuffer> feat_buf = input_nhwc16;
    size_t feat_off = 0;
    if (!input_is_fp16) {
      uint32_t vec4_cnt = height * width * 4u;  // 16 channels / 4
      KCountParams cp{vec4_cnt, 0, 0, 0};
      [enc setComputePipelineState:impl_->pso_f32_to_f16];
      [enc setBuffer:input_nhwc16 offset:0 atIndex:0];
      [enc setBuffer:impl_->scratch_buf offset:s.feat_f16_off atIndex:1];
      [enc setBytes:&cp length:sizeof(cp) atIndex:2];
      uint32_t groups = (vec4_cnt + 255u) / 256u;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      ++dispatches;
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      feat_buf = impl_->scratch_buf;
      feat_off = s.feat_f16_off;
    }

    id<MTLBuffer> head_buf = output_is_fp16 ? output_nhwc4 : impl_->scratch_buf;
    size_t head_off = output_is_fp16 ? 0 : s.head_f16_off;

    impl_->EncodeUNetCore(enc, feat_buf, feat_off, head_buf, head_off, height, width, dispatches);

    if (!output_is_fp16) {
      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
      uint32_t vec4_cnt = height * width;  // 4 channels / 4
      KCountParams cp{vec4_cnt, 0, 0, 0};
      [enc setComputePipelineState:impl_->pso_f16_to_f32];
      [enc setBuffer:impl_->scratch_buf offset:s.head_f16_off atIndex:0];
      [enc setBuffer:output_nhwc4 offset:0 atIndex:1];
      [enc setBytes:&cp length:sizeof(cp) atIndex:2];
      uint32_t groups = (vec4_cnt + 255u) / 256u;
      [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
      ++dispatches;
    }

    [enc endEncoding];

    out_ticket = FrameTicket{};
    auto st = out_ticket.state;
    [cmd addCompletedHandler:^(id<MTLCommandBuffer> cb) {
      st->gpu_start_s = cb.GPUStartTime;
      st->gpu_end_s = cb.GPUEndTime;
      st->gpu_ms = (cb.GPUEndTime > cb.GPUStartTime && cb.GPUStartTime > 0.0)
                       ? (cb.GPUEndTime - cb.GPUStartTime) * 1000.0
                       : 0.0;
      bool ok = (cb.status == MTLCommandBufferStatusCompleted);
      st->succeeded.store(ok, std::memory_order_release);
      st->completed.store(true, std::memory_order_release);
    }];

    const auto t1 = std::chrono::steady_clock::now();
    out_report = Report{};
    out_report.output_width = width;
    out_report.output_height = height;
    out_report.network_width = width;
    out_report.network_height = height;
    out_report.block_count = static_cast<uint32_t>(impl_->topology.total_blocks);
    out_report.dispatch_count = dispatches;
    out_report.cpu_encode_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    out_report.blend_scale = impl_->topology.blend_scale;
    out_report.synthetic_weights = impl_->synthetic;
    out_report.topology = impl_->topology.name + " (" + impl_->topology.spec_string + ")";
    out_report.weight_format = impl_->weight_format;
    return true;
  }
}

bool Context::eval_tensor_sync(
    const void* input_nhwc16,
    bool input_is_fp16,
    uint32_t height,
    uint32_t width,
    void* output_nhwc4,
    bool output_is_fp16,
    NRMetalCommandQueue queue,
    Report& out_report,
    std::string* out_error) {
  @autoreleasepool {
    if (!input_nhwc16 || !output_nhwc4) {
      if (out_error) *out_error = "null host input or output pointer";
      return false;
    }
    const size_t in_bytes =
        static_cast<size_t>(height) * width * 16u * (input_is_fp16 ? sizeof(uint16_t) : sizeof(float));
    const size_t out_bytes =
        static_cast<size_t>(height) * width * 4u * (output_is_fp16 ? sizeof(uint16_t) : sizeof(float));

    id<MTLBuffer> in_buf = [impl_->device newBufferWithBytes:input_nhwc16
                                                      length:in_bytes
                                                     options:MTLResourceStorageModeShared];
    id<MTLBuffer> out_buf = [impl_->device newBufferWithLength:out_bytes
                                                       options:MTLResourceStorageModeShared];
    if (!in_buf || !out_buf) {
      if (out_error) *out_error = "failed to allocate shared tensor I/O buffers";
      return false;
    }

    id<MTLCommandQueue> q = queue ? queue : impl_->internal_queue;
    id<MTLCommandBuffer> cmd = [q commandBuffer];
    FrameTicket ticket;
    if (!encode_tensor(
            in_buf, input_is_fp16, height, width, out_buf, output_is_fp16, cmd, ticket, out_report, out_error)) {
      return false;
    }
    [cmd commit];
    [cmd waitUntilCompleted];
    if (cmd.status != MTLCommandBufferStatusCompleted) {
      if (out_error) {
        *out_error = cmd.error ? std::string([[cmd.error localizedDescription] UTF8String])
                               : "GPU command buffer failed";
      }
      return false;
    }
    out_report.gpu_ms = (cmd.GPUEndTime > cmd.GPUStartTime && cmd.GPUStartTime > 0.0)
                            ? (cmd.GPUEndTime - cmd.GPUStartTime) * 1000.0
                            : ticket.gpu_ms();
    std::memcpy(output_nhwc4, [out_buf contents], out_bytes);
    return true;
  }
}

}  // namespace netal_nr_fast
