#include <metal_stdlib>

using namespace metal;

struct PoolingParameters {
  uint input_width;
  uint input_height;
  uint output_width;
  uint output_height;
  uint channels;
};

// Finite-domain bit-exact form of the selected PTX sequence:
//   set.gt.f16x2; mul.f16x2; set.le.f16x2; mul.f16x2; add.f16x2.
// Integer ordering prevents Metal fast-math from canonicalizing the selected
// device's negative-zero result while retaining every finite FP16 payload bit.
inline ushort ptx_finite_max(ushort left, ushort right) {
  const ushort left_magnitude = left & 0x7FFFu;
  const ushort right_magnitude = right & 0x7FFFu;
  if (left_magnitude == 0u && right_magnitude == 0u) {
    return left & right;
  }
  const bool left_negative = (left & 0x8000u) != 0u;
  const bool right_negative = (right & 0x8000u) != 0u;
  if (left_negative != right_negative) {
    return left_negative ? right : left;
  }
  if (left_magnitude == right_magnitude) {
    return right;
  }
  if (left_negative) {
    return left_magnitude < right_magnitude ? left : right;
  }
  return left_magnitude > right_magnitude ? left : right;
}

inline uint ptx_pair_max(uint left_bits, uint right_bits) {
  const ushort2 left = as_type<ushort2>(left_bits);
  const ushort2 right = as_type<ushort2>(right_bits);
  const ushort2 result = ushort2(ptx_finite_max(left.x, right.x),
                                 ptx_finite_max(left.y, right.y));
  return as_type<uint>(result);
}

kernel void fg_pooling(
    device uint* output_pairs [[buffer(0)]],
    device const uint* input_pairs [[buffer(1)]],
    constant PoolingParameters& parameters [[buffer(2)]],
    uint3 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.output_width ||
      position.y >= parameters.output_height) {
    return;
  }

  const uint channel_pairs = parameters.channels / 2u;
  const uint input_x = position.x * 2u;
  const uint input_y = position.y * 2u;
  for (uint channel_pair = position.z; channel_pair < channel_pairs;
       channel_pair += 16u) {
    uint current = input_pairs[
        (input_y * parameters.input_width + input_x) * channel_pairs +
        channel_pair];
    if (input_x + 1u < parameters.input_width) {
      current = ptx_pair_max(
          current,
          input_pairs[(input_y * parameters.input_width + input_x + 1u) *
                          channel_pairs +
                      channel_pair]);
    }
    if (input_y + 1u < parameters.input_height) {
      current = ptx_pair_max(
          current,
          input_pairs[((input_y + 1u) * parameters.input_width + input_x) *
                          channel_pairs +
                      channel_pair]);
      if (input_x + 1u < parameters.input_width) {
        current = ptx_pair_max(
            current,
            input_pairs[((input_y + 1u) * parameters.input_width +
                         input_x + 1u) *
                            channel_pairs +
                        channel_pair]);
      }
    }
    output_pairs[(position.y * parameters.output_width + position.x) *
                     channel_pairs +
                 channel_pair] = current;
  }
}
