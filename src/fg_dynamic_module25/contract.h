#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace fg_dynamic_module25 {
inline constexpr char kModuleSha[] =
    "28fe0e153d1ffc80b54995c022f18395c5556f3c041cc9d724bfe7e70dc4e824";
struct R {
  const char *name;
  std::uint32_t w, h, gx, gy;
  const char *reset_raw;
  const char *generated_raw;
};
inline constexpr std::array<R, 4> rs = {
    {{"720p", 640, 360, 40, 23,
      "e5b3874c2603f4c8ac64fe98796c3a913a1e920139309e7936305b6a2bfbfe10",
      "a644f8bdb9ff0a9e4f76974c4721e072c2376a6649d0c41b15a52ac27d770bb1"},
     {"1080p", 960, 540, 60, 34,
      "1c3e981b92f32f5846d9994f1c09fa1285850325a641493ba7c1f8e8e2351633",
      "6bc26ed86b57cca4afdc55e29867be77bb0af60f0b87ee0ba1124eb4b9137a9c"},
     {"1440p", 1280, 720, 80, 45,
      "1ee4004c69c48cba98476f99c0f87fa06d42301db48a994304a1520dba4aad1f",
      "0840d255f3333717073220349721f3e41d63295ce193ece912205431c6893491"},
     {"4k", 1920, 1080, 120, 68,
      "8895fac3be44519c63305a405aebeb90e7d36f7c509e27b5d42e52eb1b481fde",
      "91c31176eab2339f7077f39e24caf918c97dd758137c568330a2988be99098e3"}}};
inline const R &Find(const std::string &s) {
  for (auto &r : rs)
    if (s == r.name)
      return r;
  throw std::runtime_error("resolution");
}
inline std::uint64_t Mix(std::uint64_t v) {
  v += 0x9e3779b97f4a7c15ull;
  v = (v ^ (v >> 30)) * 0xbf58476d1ce4e5b9ull;
  v = (v ^ (v >> 27)) * 0x94d049bb133111ebull;
  return v ^ (v >> 31);
}
inline void S(std::vector<std::uint8_t> &b, std::size_t o, std::uint16_t v) {
  memcpy(b.data() + o, &v, 2);
}
inline std::vector<std::uint8_t> Input(const R &r, bool g) {
  std::uint16_t v[9] = {0xc400, 0xc000, 0xbc00, 0xb800, 0,
                        0x3800, 0x3c00, 0x4000, 0x4400};
  std::vector<std::uint8_t> b(std::size_t(r.w) * r.h * 4);
  for (std::size_t p = 0; p < b.size() / 4; ++p) {
    auto m = Mix(p + (g ? 19 : 3));
    S(b, p * 4, v[m % 9]);
    S(b, p * 4 + 2, v[(m >> 9) % 9]);
  }
  return b;
}
} // namespace fg_dynamic_module25
