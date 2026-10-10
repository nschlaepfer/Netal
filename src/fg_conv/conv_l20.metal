#include <metal_stdlib>
using namespace metal;

struct ConvL20Parameters {
  uint width;
  uint height;
  uint padded_width;
  uint padded_height;
  uint output_width;
  uint output_height;
};

constant uint kInputChannels = 48;
constant uint kOutputChannels = 32;
constant uint kBiasHalfOffset = 55296 / 2;
constant uint kSpatialOrder[9] = {0, 3, 6, 1, 4, 7, 2, 5, 8};

inline uint packed_weight_half_index(uint raw_spatial_tile,
                                     uint input_channel,
                                     uint output_channel) {
  const uint output_group = output_channel / 8;
  const uint n = output_channel % 8;
  const uint channel_stage = input_channel / 16;
  const uint k = input_channel % 16;
  const uint thread_in_group = (k % 8) / 2;
  const uint fragment_half = (k >= 8 ? 2 : 0) + (k & 1);
  const uint lane = n * 4 + thread_in_group;
  const uint byte_offset = output_group * 6912 + raw_spatial_tile * 768 +
                           channel_stage * 256 + lane * 8 + fragment_half * 2;
  return byte_offset / 2;
}

inline ulong round_shift_rne(ulong value, uint shift) {
  if (shift == 0u) return value;
  if (shift >= 64u) return 0ul;
  const ulong quotient = value >> shift;
  const ulong remainder = value & ((1ul << shift) - 1ul);
  const ulong halfway = 1ul << (shift - 1u);
  return quotient +
         ulong(remainder > halfway ||
               (remainder == halfway && (quotient & 1ul) != 0ul));
}

inline half fixed_to_half_rne(long value, int scale) {
  if (value == 0l) return as_type<half>(ushort(0u));
  const bool negative = value < 0l;
  const ulong magnitude = negative ? ulong(-value) : ulong(value);
  uint leading = 0u;
  for (ulong scan = magnitude; scan > 1ul; scan >>= 1u) ++leading;
  int exponent = int(leading) + scale;
  ulong significand = 0ul;
  uint result = 0u;
  if (exponent >= -14) {
    const int shift = int(leading) - 10;
    significand = shift > 0 ? round_shift_rne(magnitude, uint(shift))
                            : magnitude << uint(-shift);
    if (significand >= 2048ul) {
      significand >>= 1u;
      ++exponent;
    }
    if (exponent > 15) {
      result = 0x7c00u;
    } else {
      result = (uint(exponent + 15) << 10u) |
               (uint(significand) - 1024u);
    }
  } else {
    const int shift = -(scale + 24);
    significand = shift > 0 ? round_shift_rne(magnitude, uint(shift))
                            : magnitude << uint(-shift);
    result = significand >= 1024ul ? 0x0400u : uint(significand);
  }
  if (negative) result |= 0x8000u;
  return as_type<half>(ushort(result));
}

// RTX Blackwell HMMA.16816.F16 aligns the 16 exact binary16 products and the
// prior half accumulator to the largest exponent, retains two alignment bits,
// truncates discarded alignment bits, then performs one RNE conversion.
inline half hmma_f16_accumulate_16(
    device const half* packed_weights,
    device const half* input,
    uint input_pixel,
    bool input_in_bounds,
    uint raw_spatial_tile,
    uint channel_stage,
    uint output_channel,
    half prior) {
  uint product_significands[16];
  int product_exponents[16];
  bool product_negative[16];
  int maximum_exponent = -21;
  for (uint inner = 0u; inner < 16u; ++inner) {
    const uint input_channel = channel_stage * 16u + inner;
    const ushort a_bits = input_in_bounds
        ? as_type<ushort>(input[input_pixel + input_channel])
        : ushort(0u);
    const ushort b_bits = as_type<ushort>(packed_weights[
        packed_weight_half_index(raw_spatial_tile, input_channel,
                                 output_channel)]);
    const uint a_field = (uint(a_bits) >> 10u) & 31u;
    const uint b_field = (uint(b_bits) >> 10u) & 31u;
    uint a_mantissa = uint(a_bits) & 1023u;
    uint b_mantissa = uint(b_bits) & 1023u;
    if (a_field != 0u) a_mantissa += 1024u;
    if (b_field != 0u) b_mantissa += 1024u;
    const int a_exponent = a_field == 0u ? -14 : int(a_field) - 15;
    const int b_exponent = b_field == 0u ? -14 : int(b_field) - 15;
    product_significands[inner] = a_mantissa * b_mantissa * 8u;
    product_exponents[inner] = a_exponent + b_exponent;
    product_negative[inner] = ((a_bits ^ b_bits) & 0x8000u) != 0u;
    if (product_significands[inner] != 0u) {
      maximum_exponent = max(maximum_exponent, product_exponents[inner]);
    }
  }

  const ushort c_bits = as_type<ushort>(prior);
  const uint c_field = (uint(c_bits) >> 10u) & 31u;
  const uint c_fraction = uint(c_bits) & 1023u;
  uint c_significand = 0u;
  int c_exponent = -1024;
  if (c_field != 0u) {
    c_significand = (1024u + c_fraction) << 13u;
    c_exponent = int(c_field) - 15;
  } else if (c_fraction != 0u) {
    uint leading = 0u;
    for (uint scan = c_fraction; scan > 1u; scan >>= 1u) ++leading;
    c_significand = c_fraction << (23u - leading);
    c_exponent = int(leading) - 24;
  }
  if (c_significand != 0u) {
    maximum_exponent = max(maximum_exponent, c_exponent);
  }

  long total = 0l;
  for (uint inner = 0u; inner < 16u; ++inner) {
    const int shift = maximum_exponent - product_exponents[inner];
    const uint aligned = shift > 31
        ? 0u
        : (product_significands[inner] << 2u) >> uint(shift);
    total += product_negative[inner] ? -long(aligned) : long(aligned);
  }
  if (c_significand != 0u) {
    const int shift = maximum_exponent - c_exponent;
    const uint aligned = shift > 31
        ? 0u
        : (c_significand << 2u) >> uint(shift);
    total += (c_bits & 0x8000u) != 0u ? -long(aligned) : long(aligned);
  }
  return fixed_to_half_rne(total, maximum_exponent - 25);
}

kernel void fg_conv_l20(
    device const half* packed_weights [[buffer(0)]],
    device const half* input [[buffer(1)]],
    device half* output [[buffer(2)]],
    constant ConvL20Parameters& parameters [[buffer(3)]],
    uint3 gid [[thread_position_in_grid]]) {
  if (gid.x >= parameters.output_width ||
      gid.y >= parameters.output_height || gid.z >= kOutputChannels) {
    return;
  }

  const uint output_channel = gid.z;
  const uint output_group = output_channel / 8;
  const uint output_in_group = output_channel % 8;
  half accumulator = packed_weights[
      kBiasHalfOffset + output_group * 128 + output_in_group];

  // The PTX advances the three channel stages outermost. Within each stage a
  // scalar output accumulator sees spatial tiles in this HMMA chain order.
  for (uint channel_stage = 0; channel_stage < 3; ++channel_stage) {
    for (uint spatial_step = 0; spatial_step < 9; ++spatial_step) {
      const uint raw_spatial_tile = kSpatialOrder[spatial_step];
      const int ky = int(raw_spatial_tile / 3);
      const int kx = int(raw_spatial_tile % 3);
      const int input_y = int(gid.y) + ky - 1;
      const int input_x = int(gid.x) + kx - 1;
      const bool input_in_bounds =
          input_y >= 0 && input_y < int(parameters.height) &&
          input_x >= 0 && input_x < int(parameters.width);
      const uint input_pixel = input_in_bounds
          ? (uint(input_y) * parameters.width + uint(input_x)) * kInputChannels
          : 0u;
      accumulator = hmma_f16_accumulate_16(
          packed_weights, input, input_pixel, input_in_bounds,
          raw_spatial_tile, channel_stage, output_channel, accumulator);
    }
  }

  const half leaky_slope = half(0.01f);
  accumulator = max(accumulator, half(accumulator * leaky_slope));
  const uint output_index =
      (gid.y * parameters.output_width + gid.x) * kOutputChannels +
      output_channel;
  output[output_index] = accumulator;
}
