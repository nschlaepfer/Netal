#define main fg_truncation_analyzer_unused_main
#include "analyze_truncation.cc"
#undef main

namespace {

constexpr std::uint64_t kRandomRecords = 1u << 22u;
constexpr std::uint32_t kMaximumTerm = 134086688u;
constexpr std::uint32_t kMaximumPrior = 67076096u;
constexpr std::uint64_t kProvenMaximumTotal =
    std::uint64_t{16} * kMaximumTerm + kMaximumPrior;

std::uint64_t Mix(std::uint64_t value) {
  value += 0x9e3779b97f4a7c15ull;
  value = (value ^ (value >> 30u)) * 0xbf58476d1ce4e5b9ull;
  value = (value ^ (value >> 27u)) * 0x94d049bb133111ebull;
  return value ^ (value >> 31u);
}

std::uint16_t FiniteHalf(std::uint64_t value) {
  std::uint16_t bits = static_cast<std::uint16_t>(value);
  if ((bits & 0x7c00u) == 0x7c00u) bits ^= 0x0400u;
  return bits;
}

std::uint16_t FixedU32ToHalf(std::uint32_t magnitude, bool negative,
                             int scale) {
  if (magnitude == 0) return 0;
  const std::uint32_t leading = 31u - std::countl_zero(magnitude);
  int exponent = int(leading) + scale;
  std::uint32_t significand = 0;
  std::uint32_t result = 0;
  if (exponent >= -14) {
    const int shift = int(leading) - 10;
    significand = shift > 0 ? std::uint32_t(RoundShiftRne(magnitude, shift))
                            : magnitude << -shift;
    if (significand >= 2048u) {
      significand >>= 1u;
      ++exponent;
    }
    result = exponent > 15
        ? 0x7c00u
        : (std::uint32_t(exponent + 15) << 10u) |
              (significand - 1024u);
  } else {
    const int shift = -(scale + 24);
    significand = shift > 0 ? std::uint32_t(RoundShiftRne(magnitude, shift))
                            : magnitude << -shift;
    result = significand >= 1024u ? 0x0400u : significand;
  }
  if (negative) result |= 0x8000u;
  return static_cast<std::uint16_t>(result);
}

struct U32Result {
  std::uint16_t bits;
  std::uint32_t positive;
  std::uint32_t negative;
};

U32Result U32Stage(const std::array<std::uint16_t, 16>& a,
                   const std::array<std::uint16_t, 16>& b,
                   std::uint16_t prior) {
  std::array<std::uint32_t, 16> significands{};
  std::array<int, 16> exponents{};
  std::array<bool, 16> signs{};
  int maximum_exponent = -21;
  for (std::size_t index = 0; index < 16; ++index) {
    const std::uint32_t af = (a[index] >> 10u) & 31u;
    const std::uint32_t bf = (b[index] >> 10u) & 31u;
    std::uint32_t am = a[index] & 1023u;
    std::uint32_t bm = b[index] & 1023u;
    if (af != 0) am += 1024u;
    if (bf != 0) bm += 1024u;
    significands[index] = am * bm * 8u;
    exponents[index] = (af == 0 ? -14 : int(af) - 15) +
                       (bf == 0 ? -14 : int(bf) - 15);
    signs[index] = ((a[index] ^ b[index]) & 0x8000u) != 0;
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
  std::uint32_t positive = 0;
  std::uint32_t negative = 0;
  for (std::size_t index = 0; index < 16; ++index) {
    const int shift = maximum_exponent - exponents[index];
    const std::uint32_t aligned = shift > 31
        ? 0u : (significands[index] << 2u) >> shift;
    (signs[index] ? negative : positive) += aligned;
  }
  if (c_significand != 0) {
    const int shift = maximum_exponent - c_exponent;
    const std::uint32_t aligned = shift > 31
        ? 0u : (c_significand << 2u) >> shift;
    ((prior & 0x8000u) != 0 ? negative : positive) += aligned;
  }
  const bool result_negative = negative > positive;
  const std::uint32_t magnitude = result_negative ? negative - positive
                                                  : positive - negative;
  return {FixedU32ToHalf(magnitude, result_negative,
                         maximum_exponent - 25), positive, negative};
}

}  // namespace

int main() {
  static_assert(kProvenMaximumTotal == 2212463104ull);
  std::uint64_t mismatches = 0;
  std::uint32_t observed_positive = 0;
  std::uint32_t observed_negative = 0;
  for (std::uint64_t record = 0; record < kRandomRecords; ++record) {
    std::array<std::uint16_t, 16> a{};
    std::array<std::uint16_t, 16> b{};
    std::uint64_t state = Mix(0x243f6a8885a308d3ull ^ record);
    for (std::size_t index = 0; index < 16; ++index) {
      state = Mix(state + index);
      a[index] = FiniteHalf(state);
      b[index] = FiniteHalf(state >> 16u);
    }
    const auto prior = FiniteHalf(Mix(state));
    const auto reference = ExactStage(a, b, prior);
    const auto candidate = U32Stage(a, b, prior);
    mismatches += reference.bits != candidate.bits;
    observed_positive = std::max(observed_positive, candidate.positive);
    observed_negative = std::max(observed_negative, candidate.negative);
  }
  std::cout << "finite_random_records=" << kRandomRecords << '\n'
            << "mismatches=" << mismatches << '\n'
            << "proven_maximum_unsigned_total=" << kProvenMaximumTotal << '\n'
            << "observed_maximum_positive=" << observed_positive << '\n'
            << "observed_maximum_negative=" << observed_negative << '\n'
            << "result=" << (mismatches == 0 ? "pass" : "fail") << '\n';
  return mismatches == 0 ? 0 : 3;
}
