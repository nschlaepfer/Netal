#pragma once

// Portable synthetic-input transport, not a model/output container. All fields
// and texture scalars are little endian; no paths or executable data are stored.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace neuralstudio::fg::offline {
constexpr uint32_t kWidth = 640, kHeight = 360, kTextureBytes = kWidth * kHeight * 4;
constexpr uint32_t kRecordBytes = 16 + 3 * kTextureBytes;
constexpr uint64_t kMaximumBytes = 24ull + 64ull * kRecordBytes;
using Bytes = std::vector<uint8_t>;
struct Frame { uint64_t id; bool reset; Bytes color, motion, depth; };

inline void U32(Bytes& b, uint32_t n) {
  for (unsigned s = 0; s < 32; s += 8) b.push_back(uint8_t(n >> s));
}
inline uint32_t ReadU32(std::span<const uint8_t> b, size_t offset) {
  if (offset > b.size() || b.size() - offset < 4) throw std::runtime_error("truncated FG sequence header");
  uint32_t value = 0;
  for (unsigned i = 0; i < 4; ++i) value |= uint32_t(b[offset + i]) << (8 * i);
  return value;
}
inline Bytes Header(uint32_t count) {
  if (count < 2 || count > 64) throw std::runtime_error("FG sequence needs 2..64 frames");
  Bytes b{'N','F','G','S','E','Q','0','1'};
  U32(b, kWidth); U32(b, kHeight); U32(b, count); U32(b, kRecordBytes);
  return b;
}
inline void Append(Bytes& b, const Frame& frame) {
  static_assert(std::endian::native == std::endian::little);
  if (frame.color.size() != kTextureBytes || frame.motion.size() != kTextureBytes || frame.depth.size() != kTextureBytes)
    throw std::runtime_error("FG sequence texture size mismatch");
  U32(b, uint32_t(frame.id)); U32(b, uint32_t(frame.id >> 32)); U32(b, frame.reset ? 1u : 0u); U32(b, 0);
  for (const auto* texture : {&frame.color, &frame.motion, &frame.depth}) b.insert(b.end(), texture->begin(), texture->end());
}
inline std::vector<Frame> Decode(std::span<const uint8_t> b) {
  static_assert(std::endian::native == std::endian::little);
  constexpr uint8_t magic[] = {'N','F','G','S','E','Q','0','1'};
  if (b.size() < 24 || !std::equal(std::begin(magic), std::end(magic), b.begin()) ||
      ReadU32(b, 8) != kWidth || ReadU32(b, 12) != kHeight || ReadU32(b, 20) != kRecordBytes)
    throw std::runtime_error("unsupported FG sequence format/profile");
  const uint32_t count = ReadU32(b, 16);
  if (count < 2 || count > 64 || b.size() != 24ull + uint64_t(count) * kRecordBytes)
    throw std::runtime_error("FG sequence count/length mismatch");
  std::vector<Frame> frames;
  for (uint32_t index = 0; index < count; ++index) {
    size_t p = 24 + size_t(index) * kRecordBytes;
    uint64_t id = ReadU32(b, p) | (uint64_t(ReadU32(b, p + 4)) << 32);
    uint32_t reset = ReadU32(b, p + 8);
    if (id != index || reset > 1 || (index == 0 && reset != 1) || ReadU32(b, p + 12) != 0)
      throw std::runtime_error("FG sequence frame lineage/reset mismatch");
    Frame frame{id, reset != 0, {}, {}, {}}; p += 16;
    for (auto* texture : {&frame.color, &frame.motion, &frame.depth}) {
      texture->assign(b.begin() + p, b.begin() + p + kTextureBytes); p += kTextureBytes;
    }
    // Reject non-finite half motion and depth outside the documented [0,1]
    // domain before any backend sees the data. Reset motion must be zero.
    for (size_t i = 0; i < kTextureBytes; i += 4) {
      for (unsigned axis = 0; axis < 2; ++axis) {
        uint16_t h = uint16_t(uint16_t(frame.motion[i + axis * 2]) | (uint16_t(frame.motion[i + axis * 2 + 1]) << 8));
        if ((h & 0x7c00u) == 0x7c00u || (frame.reset && (h & 0x7fffu) != 0))
          throw std::runtime_error("FG sequence motion violates finite/reset contract");
      }
      uint32_t bits = ReadU32(frame.depth, i);
      float depth = std::bit_cast<float>(bits);
      if (!(depth >= 0 && depth <= 1)) throw std::runtime_error("FG sequence depth outside [0,1]");
    }
    frames.push_back(std::move(frame));
  }
  return frames;
}
}  // namespace neuralstudio::fg::offline
