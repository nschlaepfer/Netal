// Colour import/export variants for games whose MetalFX textures are not RGB9E5.
//
// The session's colour and output buffers both hold one half4 per pixel (kColorRowBytes and
// kOutputRowBytes are width * 8). The original converter unpacks RGB9E5 into that layout and packs
// it back out again; every format here is one Metal already converts on read/write, so these are
// straight copies.
//
// This lives in its own library rather than being added to dlss_sr_live_texture_convert.metallib,
// which belongs to the RE9 runtime bundle whose output is verified against RTX receipts and stays
// byte-for-byte unchanged.
//
// Formats covered, and why these:
//   RGBA16Float   (115) plain HDR half4 - Lies of P's colour texture
//   RG11B10Float  ( 92) packed HDR, no alpha - Lies of P's output texture
//   RGBA8Unorm    ( 70) / BGRA8Unorm (80) LDR, for games that upscale before tonemapping
//   RGB10A2Unorm  ( 90) / BGR10A2Unorm (94) 10-bit - Death Stranding's colour and output textures
//
// A note on the 10-bit pair, because it is not just another copy: unorm formats are normalised to
// [0,1], so whatever the game stores there is display-referred rather than the linear HDR radiance
// the network was trained on. The transfer round-trips cleanly - half carries eleven bits of
// mantissa, more than the ten a channel holds - but a correct *image* additionally depends on the
// game's colour encoding matching what DLSS expects, which no receipt here establishes.
//
// Metal performs the pack/unpack in texture read/write, so each kernel is the same shape and the
// format only appears in the texture type. Alpha is carried through where the format has one:
// DLSS super-resolution consumes RGB, an RGB9E5 source has no alpha at all, so nothing downstream
// depends on a particular value.
#include <metal_stdlib>
using namespace metal;

#define NETAL_IMPORT(NAME)                                                             \
  kernel void NAME(texture2d<half, access::read> source [[texture(0)]],                \
                   device half4* destination [[buffer(0)]],                            \
                   constant uint2& content_size [[buffer(1)]],                         \
                   uint2 position [[thread_position_in_grid]]) {                       \
    if (position.x >= content_size.x || position.y >= content_size.y) return;          \
    destination[position.y * content_size.x + position.x] = source.read(position);     \
  }

#define NETAL_EXPORT(NAME)                                                             \
  kernel void NAME(device const half4* source [[buffer(0)]],                           \
                   texture2d<half, access::write> destination [[texture(0)]],          \
                   constant uint2& output_size [[buffer(1)]],                          \
                   uint2 position [[thread_position_in_grid]]) {                       \
    if (position.x >= output_size.x || position.y >= output_size.y) return;            \
    destination.write(source[position.y * output_size.x + position.x], position);      \
  }

// One kernel per direction is enough: `texture2d<half>` reads and writes any of these formats,
// with Metal doing the conversion. They are named per format so the adapter's lookup stays
// explicit and a future format that needs real work can diverge without disturbing the others.
NETAL_IMPORT(neuralstudio_sr_import_rgba16f)
NETAL_IMPORT(neuralstudio_sr_import_rg11b10f)
NETAL_IMPORT(neuralstudio_sr_import_rgba8unorm)
NETAL_IMPORT(neuralstudio_sr_import_rgb10a2)

NETAL_EXPORT(neuralstudio_sr_export_rgba16f)
NETAL_EXPORT(neuralstudio_sr_export_rg11b10f)
NETAL_EXPORT(neuralstudio_sr_export_rgba8unorm)
NETAL_EXPORT(neuralstudio_sr_export_rgb10a2)
