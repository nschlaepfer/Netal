#include "../fg_dynamic_module5/dynamic_module5.metal"

inline DynamicModule5Features connected_dynamic_compute_features(
    texture2d<half, access::read> color_reset,
    texture2d<half, access::read> color_generated,
    texture2d<float, access::read> mask_generated,
    texture2d<float, access::read> mask_reset,
    texture2d<half, access::read> projection,
    texture2d<half, access::read> scalar_shared,
    texture2d<half, access::read> scalar_persistent,
    texture2d<float, access::read> vector_current,
    texture2d<half, access::read> vector_repaired,
    texture2d<float, access::read> mask_working,
    int2 position, constant DynamicModule5Parameters& parameters,
    device const uchar* correction_rom) {
  DynamicModule5Features result{};
  const uint2 color_extent = uint2(parameters.work_width,
                                   parameters.work_height);
  const uint2 motion_extent = uint2(parameters.motion_width,
                                    parameters.motion_height);
  if (any(position < 0) || position.x >= int(color_extent.x) ||
      position.y >= int(color_extent.y)) return result;
  const float2 normalized = float2(
      selected_div(float(position.x) + 0.5f, float(color_extent.x),
                   correction_rom),
      selected_div(float(position.y) + 0.5f, float(color_extent.y),
                   correction_rom));
  const float working_mask = sample_unorm(
      mask_working, normalized, motion_extent);
  const float reset_mask = sample_unorm(
      mask_reset, normalized, motion_extent);
  const float generated_mask = sample_unorm(
      mask_generated, normalized, motion_extent);
  result.mask_output = clamp(
      (working_mask * reset_mask) * generated_mask, 0.0f, 1.0f);

  const float4 reset_base = read_half4(color_reset, position, color_extent);
  const float4 generated_base =
      read_half4(color_generated, position, color_extent);
  const float2 repaired = sample_half4(
      vector_repaired, normalized, motion_extent).xy;
  const float2 repaired_normalized = float2(
      selected_div(repaired.x, float(motion_extent.x), correction_rom),
      selected_div(repaired.y, float(motion_extent.y), correction_rom));
  const float2 current = sample_float2(
      vector_current, normalized, motion_extent);
  const float4 reset_shifted = sample_half4(
      color_reset, normalized + current, color_extent);
  const float4 generated_shifted = sample_half4(
      color_generated, normalized + repaired_normalized, color_extent);
  const float generated_difference = selected_mean_absolute_rgb(
      generated_shifted, reset_base, correction_rom);
  const float reset_difference = selected_mean_absolute_rgb(
      reset_shifted, generated_base, correction_rom);
  result.vector_output = float2(reset_difference, generated_difference);

  const float persistent = read_half4(
      scalar_persistent, position, color_extent).x;
  result.value[0] = read_half4(scalar_shared, position, color_extent).x;
  result.value[1] = selected_luma(reset_base);
  result.value[2] = generated_difference;
  result.value[3] = read_half4(projection, position, color_extent).x;
  result.value[4] = working_mask;
  result.value[5] = selected_luma(generated_base);
  result.value[6] = reset_difference;
  result.value[7] = generated_mask;
  result.value[8] = persistent;
  result.value[9] = 1.0f - persistent;
  return result;
}

kernel void fg_connected_dynamic_module5_surfaces(
    texture2d<half, access::read> color_reset [[texture(0)]],
    texture2d<half, access::read> color_generated [[texture(1)]],
    texture2d<float, access::read> mask_generated [[texture(2)]],
    texture2d<float, access::read> mask_reset [[texture(3)]],
    texture2d<half, access::read> projection [[texture(4)]],
    texture2d<half, access::read> scalar_shared [[texture(5)]],
    texture2d<half, access::read> scalar_persistent [[texture(6)]],
    texture2d<float, access::read> vector_current [[texture(7)]],
    texture2d<half, access::read> vector_repaired [[texture(8)]],
    texture2d<float, access::read> mask_working [[texture(9)]],
    texture2d<float, access::write> mask_output [[texture(10)]],
    texture2d<half, access::write> vector_output [[texture(11)]],
    constant DynamicModule5Parameters& parameters [[buffer(0)]],
    device const uchar* correction_rom [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.work_width ||
      position.y >= parameters.work_height) return;
  const auto features = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(position), parameters, correction_rom);
  mask_output.write(float4(features.mask_output, 0.0f, 0.0f, 0.0f), position);
  vector_output.write(
      half4(as_type<half>(float_to_half_rz_bits(features.vector_output.x)),
            as_type<half>(float_to_half_rz_bits(features.vector_output.y)),
            0.0h, 0.0h), position);
}

kernel void fg_connected_dynamic_module5_tensor(
    texture2d<half, access::read> color_reset [[texture(0)]],
    texture2d<half, access::read> color_generated [[texture(1)]],
    texture2d<float, access::read> mask_generated [[texture(2)]],
    texture2d<float, access::read> mask_reset [[texture(3)]],
    texture2d<half, access::read> projection [[texture(4)]],
    texture2d<half, access::read> scalar_shared [[texture(5)]],
    texture2d<half, access::read> scalar_persistent [[texture(6)]],
    texture2d<float, access::read> vector_current [[texture(7)]],
    texture2d<half, access::read> vector_repaired [[texture(8)]],
    texture2d<float, access::read> mask_working [[texture(9)]],
    device ushort* tensor_output [[buffer(0)]],
    constant DynamicModule5Parameters& parameters [[buffer(1)]],
    device const uchar* correction_rom [[buffer(2)]],
    uint2 position [[thread_position_in_grid]]) {
  if (position.x >= parameters.tensor_width ||
      position.y >= parameters.tensor_height) return;
  const int base_x = int(position.x * 2u) + parameters.input_x_offset;
  const int base_y = int(position.y * 2u) + parameters.input_y_offset;
  DynamicModule5Features subpixels[4];
  subpixels[0] = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y), parameters, correction_rom);
  subpixels[1] = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y), parameters, correction_rom);
  subpixels[2] = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x, base_y + 1), parameters, correction_rom);
  subpixels[3] = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(base_x + 1, base_y + 1), parameters, correction_rom);
  const uint base =
      (position.y * parameters.tensor_width + position.x) * 48u;
  for (uint feature = 0; feature < 10u; ++feature)
    for (uint subpixel = 0; subpixel < 4u; ++subpixel)
      tensor_output[base + feature * 4u + subpixel] =
          as_type<ushort>(half(subpixels[subpixel].value[feature]));
  for (uint channel = 40u; channel < 48u; ++channel)
    tensor_output[base + channel] = 0u;
}
