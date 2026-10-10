#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <vector>

namespace {

constexpr std::uint32_t kWidth = 160;
constexpr std::uint32_t kHeight = 96;
constexpr std::uint32_t kInputChannels = 48;
constexpr std::uint32_t kOutputChannels = 32;
constexpr std::uint32_t kStages = 27;
constexpr std::uint32_t kSpatialOrder[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};
constexpr std::uint32_t kActiveWeightHalves =
    9 * kInputChannels * kOutputChannels;

std::vector<std::uint8_t> Read(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary | std::ios::ate);
  if (!stream) throw std::runtime_error("unable to read input");
  const auto size = stream.tellg();
  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  stream.seekg(0);
  stream.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!stream) throw std::runtime_error("short input read");
  return bytes;
}

std::uint16_t Load16(const std::vector<std::uint8_t>& bytes,
                     std::size_t index) {
  std::uint16_t value = 0;
  std::memcpy(&value, bytes.data() + index * 2u, sizeof(value));
  return value;
}

float HalfToFloat(std::uint16_t bits) {
  _Float16 value{};
  std::memcpy(&value, &bits, sizeof(bits));
  return static_cast<float>(value);
}

std::uint16_t FloatToHalf(float value) {
  const _Float16 half_value = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half_value, sizeof(bits));
  return bits;
}

std::uint16_t DoubleToHalf(double value) {
  const _Float16 half_value = static_cast<_Float16>(value);
  std::uint16_t bits = 0;
  std::memcpy(&bits, &half_value, sizeof(bits));
  return bits;
}

std::uint64_t RoundShiftRne(std::uint64_t value, std::uint32_t shift) {
  if (shift == 0) return value;
  if (shift >= 64) return 0;
  const auto quotient = value >> shift;
  const auto remainder = value & ((std::uint64_t{1} << shift) - 1);
  const auto halfway = std::uint64_t{1} << (shift - 1);
  return quotient + (remainder > halfway ||
                     (remainder == halfway && (quotient & 1) != 0));
}

std::uint16_t FixedToHalf(std::int64_t value, int scale) {
  if (value == 0) return 0;
  const bool negative = value < 0;
  const auto magnitude = negative ? std::uint64_t(-value)
                                  : std::uint64_t(value);
  const std::uint32_t leading = 63u - std::countl_zero(magnitude);
  int exponent = static_cast<int>(leading) + scale;
  std::uint64_t significand = 0;
  std::uint32_t result = 0;
  if (exponent >= -14) {
    const int shift = static_cast<int>(leading) - 10;
    significand = shift > 0 ? RoundShiftRne(magnitude, shift)
                            : magnitude << -shift;
    if (significand >= 2048) {
      significand >>= 1;
      ++exponent;
    }
    result = exponent > 15
        ? 0x7c00u
        : (std::uint32_t(exponent + 15) << 10u) |
              (std::uint32_t(significand) - 1024u);
  } else {
    const int shift = -(scale + 24);
    significand = shift > 0 ? RoundShiftRne(magnitude, shift)
                            : magnitude << -shift;
    result = significand >= 1024 ? 0x0400u : std::uint32_t(significand);
  }
  if (negative) result |= 0x8000u;
  return static_cast<std::uint16_t>(result);
}

struct StageResult {
  std::uint16_t bits;
  int maximum_exponent;
  std::int64_t total;
  double discarded_common_units;
};

StageResult ExactStage(const std::array<std::uint16_t, 16>& a,
                       const std::array<std::uint16_t, 16>& b,
                       std::uint16_t prior) {
  std::array<std::uint32_t, 16> significands{};
  std::array<int, 16> exponents{};
  std::array<bool, 16> negative{};
  int maximum_exponent = -21;
  for (std::size_t index = 0; index < 16; ++index) {
    const std::uint32_t af = (a[index] >> 10u) & 31u;
    const std::uint32_t bf = (b[index] >> 10u) & 31u;
    std::uint32_t am = a[index] & 1023u;
    std::uint32_t bm = b[index] & 1023u;
    if (af != 0) am += 1024;
    if (bf != 0) bm += 1024;
    significands[index] = am * bm * 8u;
    exponents[index] = (af == 0 ? -14 : int(af) - 15) +
                       (bf == 0 ? -14 : int(bf) - 15);
    negative[index] = ((a[index] ^ b[index]) & 0x8000u) != 0;
    if (significands[index] != 0)
      maximum_exponent = std::max(maximum_exponent, exponents[index]);
  }
  const std::uint32_t cf = (prior >> 10u) & 31u;
  const std::uint32_t fraction = prior & 1023u;
  std::uint32_t c_significand = 0;
  int c_exponent = -1024;
  if (cf != 0) {
    c_significand = (1024u + fraction) << 13u;
    c_exponent = int(cf) - 15;
  } else if (fraction != 0) {
    const std::uint32_t leading = 31u - std::countl_zero(fraction);
    c_significand = fraction << (23u - leading);
    c_exponent = int(leading) - 24;
  }
  if (c_significand != 0)
    maximum_exponent = std::max(maximum_exponent, c_exponent);

  std::int64_t total = 0;
  double discarded = 0.0;
  for (std::size_t index = 0; index < 16; ++index) {
    const int shift = maximum_exponent - exponents[index];
    const std::uint32_t numerator = significands[index] << 2u;
    const std::uint32_t aligned = shift > 31 ? 0u : numerator >> shift;
    const double exact = std::ldexp(double(numerator), -shift);
    const double loss = exact - double(aligned);
    total += negative[index] ? -std::int64_t(aligned)
                             : std::int64_t(aligned);
    discarded += negative[index] ? -loss : loss;
  }
  if (c_significand != 0) {
    const int shift = maximum_exponent - c_exponent;
    const std::uint32_t numerator = c_significand << 2u;
    const std::uint32_t aligned = shift > 31 ? 0u : numerator >> shift;
    const double exact = std::ldexp(double(numerator), -shift);
    const double loss = exact - double(aligned);
    const bool c_negative = (prior & 0x8000u) != 0;
    total += c_negative ? -std::int64_t(aligned) : std::int64_t(aligned);
    discarded += c_negative ? -loss : loss;
  }
  return {FixedToHalf(total, maximum_exponent - 25), maximum_exponent,
          total, discarded};
}

std::uint32_t WeightIndex(std::uint32_t tile, std::uint32_t input_channel,
                          std::uint32_t output_channel) {
  const auto group = output_channel / 8u;
  const auto n = output_channel % 8u;
  const auto stage = input_channel / 16u;
  const auto k = input_channel % 16u;
  const auto lane = n * 4u + (k % 8u) / 2u;
  const auto fragment = (k >= 8u ? 2u : 0u) + (k & 1u);
  return group * (9u * kInputChannels * 8u) +
         tile * (kInputChannels * 8u) + stage * 128u + lane * 4u + fragment;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "usage=fg-analyze-hmma-truncation <weights> <input>\n";
    return 2;
  }
  try {
    const auto weights = Read(argv[1]);
    const auto input = Read(argv[2]);
    if (weights.size() != 56320u ||
        input.size() != std::size_t{kWidth} * kHeight * kInputChannels * 2u)
      throw std::runtime_error("input extent mismatch");
    std::map<std::uint32_t, std::uint64_t> first_stage_histogram;
    std::uint64_t output_divergences = 0;
    std::uint64_t stage_divergences = 0;
    std::uint64_t corrected_output_divergences = 0;
    std::uint64_t corrected_stage_divergences = 0;
    std::uint64_t compensated_output_divergences = 0;
    std::uint64_t compensated_stage_divergences = 0;
    std::uint64_t double_output_divergences = 0;
    std::uint64_t double_stage_divergences = 0;
    double maximum_abs_discard = 0.0;
    for (std::uint32_t y = 0; y < kHeight; ++y) {
      for (std::uint32_t x = 0; x < kWidth; ++x) {
        for (std::uint32_t output_channel = 0;
             output_channel < kOutputChannels; ++output_channel) {
          const auto group = output_channel / 8u;
          const auto lane = output_channel % 8u;
          const auto bias_index = kActiveWeightHalves * 2u + group * 128u + lane;
          std::uint16_t exact = Load16(weights, bias_index);
          std::uint16_t approximate = exact;
          std::uint16_t corrected = exact;
          std::uint16_t compensated = exact;
          std::uint16_t double_corrected = exact;
          std::uint32_t first_stage = 0;
          for (std::uint32_t stage = 0; stage < kStages; ++stage) {
            const auto channel_stage = stage / 9u;
            const auto tile = kSpatialOrder[stage % 9u];
            const int input_x = int(x) + int(tile % 3u) - 1;
            const int input_y = int(y) + int(tile / 3u) - 1;
            const bool in_bounds = input_x >= 0 && input_x < int(kWidth) &&
                                   input_y >= 0 && input_y < int(kHeight);
            std::array<std::uint16_t, 16> a{};
            std::array<std::uint16_t, 16> b{};
            float forward = HalfToFloat(approximate);
            float corrected_forward = HalfToFloat(corrected);
            float compensated_high = HalfToFloat(compensated);
            float compensated_low = 0.0f;
            double double_sum = HalfToFloat(double_corrected);
            for (std::uint32_t inner = 0; inner < 16; ++inner) {
              const auto input_channel = channel_stage * 16u + inner;
              if (in_bounds) {
                const auto input_index =
                    (std::size_t(input_y) * kWidth + input_x) *
                        kInputChannels + input_channel;
                a[inner] = Load16(input, input_index);
              }
              b[inner] = Load16(weights,
                  WeightIndex(tile, input_channel, output_channel));
              forward = std::fma(HalfToFloat(a[inner]), HalfToFloat(b[inner]),
                                 forward);
              corrected_forward = std::fma(
                  HalfToFloat(a[inner]), HalfToFloat(b[inner]),
                  corrected_forward);
              const float term = HalfToFloat(a[inner]) * HalfToFloat(b[inner]);
              const float sum = compensated_high + term;
              const float recovered = sum - compensated_high;
              const float error =
                  (compensated_high - (sum - recovered)) + (term - recovered);
              compensated_high = sum;
              compensated_low += error;
              double_sum += double(HalfToFloat(a[inner])) *
                            double(HalfToFloat(b[inner]));
            }
            const auto exact_stage = ExactStage(a, b, exact);
            exact = exact_stage.bits;
            approximate = FloatToHalf(forward);
            const double correction = std::ldexp(
                exact_stage.discarded_common_units,
                exact_stage.maximum_exponent - 25);
            corrected = FloatToHalf(
                corrected_forward - static_cast<float>(correction));
            compensated = FloatToHalf(
                compensated_high +
                (compensated_low - static_cast<float>(correction)));
            double_corrected = DoubleToHalf(double_sum - correction);
            maximum_abs_discard = std::max(
                maximum_abs_discard, std::abs(exact_stage.discarded_common_units));
            if (exact != approximate) {
              ++stage_divergences;
              if (first_stage == 0) first_stage = stage + 1u;
            }
            corrected_stage_divergences += exact != corrected;
            compensated_stage_divergences += exact != compensated;
            double_stage_divergences += exact != double_corrected;
          }
          if (exact != approximate) {
            ++output_divergences;
            ++first_stage_histogram[first_stage];
          }
          corrected_output_divergences += exact != corrected;
          compensated_output_divergences += exact != compensated;
          double_output_divergences += exact != double_corrected;
        }
      }
    }
    std::cout << "stage_divergences=" << stage_divergences << '\n'
              << "output_divergences=" << output_divergences << '\n'
              << "corrected_stage_divergences="
              << corrected_stage_divergences << '\n'
              << "corrected_output_divergences="
              << corrected_output_divergences << '\n'
              << "compensated_stage_divergences="
              << compensated_stage_divergences << '\n'
              << "compensated_output_divergences="
              << compensated_output_divergences << '\n'
              << "double_stage_divergences=" << double_stage_divergences
              << '\n'
              << "double_output_divergences=" << double_output_divergences
              << '\n'
              << "maximum_abs_discarded_common_units=" << std::setprecision(17)
              << maximum_abs_discard << '\n'
              << "first_divergence_histogram_begin\n";
    for (const auto& [stage, count] : first_stage_histogram)
      std::cout << stage << ' ' << count << '\n';
    std::cout << "first_divergence_histogram_end\nresult=diagnostic\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "error=" << error.what() << '\n';
    return 1;
  }
}
