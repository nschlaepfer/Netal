#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>
namespace fg_dynamic_module23 {
inline constexpr char kModuleSha[] =
    "d787cf7b861295d1b4c6a80cb374230de793aa69f288a7e724bf59278bc4a62d";
inline constexpr std::size_t kModuleBytes = 56480;
struct R {
  const char *name;
  uint w, h, gx, gy;
  const char *raw;
};
inline constexpr std::array<R, 4> rs = {
    {{"720p", 320, 180, 20, 12,
      "d19330332b941e43635244e93b624b095742f4e30d8319001da3a505c8f25ba0"},
     {"1080p", 480, 270, 30, 17,
      "63d735411875fe65cc2f71db526c4b2e56338fbb67ec0329866c40ffd5fb7183"},
     {"1440p", 320, 180, 20, 12,
      "bf60c5a6ef4db34c78a7bffb981e5a7cf173211bd8431ee15153f9b6723329e8"},
     {"4k", 480, 270, 30, 17,
      "8d566cfa2cd8cb66b933bbd940677ec89a061703ce794bc29a88c30431c175e3"}}};
inline const R &Find(const std::string &s) {
  for (auto &r : rs)
    if (s == r.name)
      return r;
  throw std::runtime_error("resolution");
}
inline uint C(uint x, uint d) { return (x + d - 1) / d; }
inline std::array<std::size_t, 4> Sizes(const R &r) {
  return {std::size_t(r.w) * r.h * 6, std::size_t(C(r.w, 2)) * C(r.h, 2) * 6,
          std::size_t(C(r.w, 4)) * C(r.h, 4) * 6,
          std::size_t(C(r.w, 8)) * C(r.h, 8) * 6};
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
inline std::array<std::vector<std::uint8_t>, 4> Levels(const R &r, bool g) {
  auto sz = Sizes(r);
  std::array<std::vector<std::uint8_t>, 4> a;
  std::uint16_t v[11] = {0,      0x2800, 0xa800, 0x3000, 0xb000, 0x3400,
                         0xb400, 0x3800, 0xb800, 0x3c00, 0xbc00},
                w[6] = {0, 0x3000, 0x3400, 0x3800, 0x3a00, 0x3c00};
  for (int l = 0; l < 4; ++l) {
    a[l].resize(sz[l]);
    for (std::size_t p = 0; p < sz[l] / 6; ++p) {
      auto m = Mix((g ? 17 : 3) + l * 997 + p);
      S(a[l], p * 6, v[m % 11]);
      S(a[l], p * 6 + 2, v[(m >> 10) % 11]);
      S(a[l], p * 6 + 4, w[(m >> 20) % 6]);
    }
  }
  return a;
}
inline std::vector<std::uint8_t> Mask(const R &r, bool g) {
  std::vector<std::uint8_t> b(std::size_t(r.w) * r.h);
  for (std::size_t p = 0; p < b.size(); ++p)
    b[p] = (Mix(p + (g ? 31 : 7)) & 7) < 5 ? 1 : 0;
  return b;
}
} // namespace fg_dynamic_module23
