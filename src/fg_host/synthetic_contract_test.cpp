#include "synthetic_contract.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>

int main() {
  using neuralstudio::fg::FloatToHalf;
  using neuralstudio::fg::Fnv1a64;
  using neuralstudio::fg::HalfIsFinite;
  using neuralstudio::fg::MakeSyntheticFrame;

  static_assert(sizeof(float) == sizeof(std::uint32_t));
  assert(FloatToHalf(0.0f) == 0x0000u);
  assert(FloatToHalf(-0.0f) == 0x8000u);
  assert(FloatToHalf(1.0f) == 0x3c00u);
  assert(FloatToHalf(-2.0f) == 0xc000u);
  assert(!HalfIsFinite(FloatToHalf(
      std::numeric_limits<float>::infinity())));

  constexpr std::uint32_t kWidth = 64;
  constexpr std::uint32_t kHeight = 36;
  const auto first = MakeSyntheticFrame(kWidth, kHeight, 0);
  const auto first_repeat = MakeSyntheticFrame(kWidth, kHeight, 0);
  const auto second = MakeSyntheticFrame(kWidth, kHeight, 1);
  const std::size_t pixels = static_cast<std::size_t>(kWidth) * kHeight;

  assert(first.backbuffer_rgba8.size() == pixels * 4u);
  assert(first.hudless_rgba8.size() == pixels * 4u);
  assert(first.ui_alpha_r8.size() == pixels);
  assert(first.ui_color_alpha_rgba8.size() == pixels * 4u);
  assert(first.motion_rg16f.size() == pixels * 2u);
  assert(first.depth_r32f.size() == pixels);
  assert(first.bidirectional_distortion_rgba16f.size() == pixels * 4u);
  assert(first.backbuffer_rgba8 == first_repeat.backbuffer_rgba8);
  assert(first.backbuffer_rgba8 != second.backbuffer_rgba8);
  assert(Fnv1a64(first.backbuffer_rgba8) ==
         Fnv1a64(first_repeat.backbuffer_rgba8));
  assert(Fnv1a64(first.backbuffer_rgba8) !=
         Fnv1a64(second.backbuffer_rgba8));
  assert(std::all_of(first.motion_rg16f.begin(), first.motion_rg16f.end(),
                     [](std::uint16_t value) { return value == 0u; }));
  assert(std::any_of(second.motion_rg16f.begin(), second.motion_rg16f.end(),
                     [](std::uint16_t value) { return value != 0u; }));
  assert(std::all_of(
      second.bidirectional_distortion_rgba16f.begin(),
      second.bidirectional_distortion_rgba16f.end(),
      [](std::uint16_t value) { return value == 0u; }));

  std::cout << "synthetic.contract=passed\n";
  std::cout << "synthetic.frame0.fnv1a64="
            << Fnv1a64(first.backbuffer_rgba8) << '\n';
  std::cout << "synthetic.frame1.fnv1a64="
            << Fnv1a64(second.backbuffer_rgba8) << '\n';
  return 0;
}
