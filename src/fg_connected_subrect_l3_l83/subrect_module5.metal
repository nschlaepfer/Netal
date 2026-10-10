#include "../fg_connected_dynamic_early/module5_mixed.metal"

kernel void fg_connected_subrect_module5_surfaces(
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
  constexpr uint kActiveWorkHeight = 120u;
  if (position.x >= parameters.work_width ||
      position.y >= kActiveWorkHeight) return;
  const auto features = connected_dynamic_compute_features(
      color_reset, color_generated, mask_generated, mask_reset, projection,
      scalar_shared, scalar_persistent, vector_current, vector_repaired,
      mask_working, int2(position), parameters, correction_rom);
  mask_output.write(float4(features.mask_output, 0.0f, 0.0f, 0.0f),
                    position);
  vector_output.write(
      half4(as_type<half>(float_to_half_rz_bits(features.vector_output.x)),
            as_type<half>(float_to_half_rz_bits(features.vector_output.y)),
            0.0h, 0.0h), position);
}

kernel void fg_connected_subrect_module5_tensor(
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
    uint2 local_position [[thread_position_in_grid]]) {
  constexpr uint kTensorOriginY = 12u;
  constexpr uint kActiveTensorHeight = 60u;
  if (local_position.x >= parameters.tensor_width ||
      local_position.y >= kActiveTensorHeight) return;
  const int base_x = int(local_position.x * 2u);
  const int base_y = int(local_position.y * 2u);
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
  const uint output_y = local_position.y + kTensorOriginY;
  const uint base =
      (output_y * parameters.tensor_width + local_position.x) * 48u;
  for (uint feature = 0; feature < 10u; ++feature)
    for (uint subpixel = 0; subpixel < 4u; ++subpixel)
      tensor_output[base + feature * 4u + subpixel] =
          as_type<ushort>(half(subpixels[subpixel].value[feature]));
  for (uint channel = 40u; channel < 48u; ++channel)
    tensor_output[base + channel] = 0u;
}
