#!/usr/bin/env python3
# SPDX-License-Identifier: MIT AND Apache-2.0
"""Model-free GPU integration suite for Netal's continuous DLSS 5 NR game bridge.

Exercises:
1. Atomic build (`scripts/build-netal-nr-fast-bridge.py`), SHA-256 receipts, and collision refusal.
2. Strict opt-in guard: default `off` mode and non-matching `NEURALSTUDIO_TARGET_EXE` install zero hooks.
3. `MTLFXSpatialScaler` continuous inline mode (`NETAL_NR_SYNTHETIC=small`) across 4 moving frames,
   verifying optical flow engages, output differs from MetalFX passthrough, and all pixels are finite.
4. Simulated Metal 4 `MTL4CommandQueue` `signalDrawable:` with strict `RenderTarget`-only (`ShaderWrite == 0`)
   textures in both `inline` (`NETAL_NR_ASYNC=0`) and `async` (`NETAL_NR_ASYNC=1`) modes across 6 frames,
   plus opt-in diagnostic `raw_<frame>.png` / `nr_<frame>.png` capture verification.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / 'scripts' / 'build-netal-nr-fast-bridge.py'


HARNESS_MM = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalFX/MetalFX.h>
#import <QuartzCore/CAMetalLayer.h>
#import <dlfcn.h>
#import <objc/runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unistd.h>
#include <vector>

using HookMaskFn = uint32_t (*)();
using PublishedCountFn = uint64_t (*)();

@interface NetalMockDrawable : NSObject <CAMetalDrawable>
@property(nonatomic, strong) id<MTLTexture> texture;
@property(nonatomic, strong) CAMetalLayer* layer;
@property(nonatomic, readonly) CFTimeInterval presentedTime;
@property(nonatomic, readonly) NSUInteger drawableID;
@end

@implementation NetalMockDrawable
@synthesize presentedTime = _presentedTime;
@synthesize drawableID = _drawableID;
- (void)present {}
- (void)presentAtTime:(CFTimeInterval)presentationTime { (void)presentationTime; }
- (void)presentAfterMinimumDuration:(CFTimeInterval)duration { (void)duration; }
- (void)addPresentedHandler:(MTLDrawablePresentedHandler)block { (void)block; }
- (void)signalOnCommandQueue:(id)queue { (void)queue; }
@end

static void FillMovingPatternRGBA8(
    std::vector<uint8_t>& rgba,
    uint32_t w,
    uint32_t h,
    int shift_x,
    int shift_y) {
  rgba.resize(static_cast<size_t>(w) * h * 4u);
  for (uint32_t y = 0; y < h; ++y) {
    for (uint32_t x = 0; x < w; ++x) {
      const float sx = static_cast<float>(static_cast<int>(x) - shift_x);
      const float sy = static_cast<float>(static_cast<int>(y) - shift_y);
      const float fx = sx / static_cast<float>(w);
      const float fy = sy / static_cast<float>(h);
      const int cx = ((static_cast<int>(x) - shift_x) >> 4);
      const int cy = ((static_cast<int>(y) - shift_y) >> 4);
      const float chk = ((cx + cy) & 1) ? 0.18f : -0.12f;
      const float r = std::clamp(0.42f + 0.30f * std::sin(fx * 18.0f + fy * 6.0f) + chk, 0.05f, 0.95f);
      const float g = std::clamp(0.46f + 0.28f * std::cos(fx * 13.0f - fy * 15.0f) + 0.5f * chk, 0.05f, 0.95f);
      const float b = std::clamp(0.40f + 0.32f * std::sin((fx + fy) * 21.0f) - chk, 0.05f, 0.95f);
      const size_t idx = (static_cast<size_t>(y) * w + x) * 4u;
      rgba[idx + 0u] = static_cast<uint8_t>(std::lround(r * 255.0f));
      rgba[idx + 1u] = static_cast<uint8_t>(std::lround(g * 255.0f));
      rgba[idx + 2u] = static_cast<uint8_t>(std::lround(b * 255.0f));
      rgba[idx + 3u] = 255u;
    }
  }
}

static id<MTLTexture> MakeSharedRGBA8(id<MTLDevice> dev, uint32_t w, uint32_t h, MTLTextureUsage usage) {
  MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatRGBA8Unorm
                                                         width:w
                                                        height:h
                                                     mipmapped:NO];
  td.storageMode = MTLStorageModeShared;
  td.usage = usage;
  return [dev newTextureWithDescriptor:td];
}

static id<MTLTexture> MakePrivateRenderTargetOnlyBGRA8(id<MTLDevice> dev, uint32_t w, uint32_t h) {
  MTLTextureDescriptor* td =
      [MTLTextureDescriptor texture2DDescriptorWithPixelFormat:MTLPixelFormatBGRA8Unorm
                                                         width:w
                                                        height:h
                                                     mipmapped:NO];
  td.storageMode = MTLStorageModePrivate;
  // Strictly RenderTarget ONLY (0x4): NO ShaderRead (0x1) and NO ShaderWrite (0x2)!
  td.usage = MTLTextureUsageRenderTarget;
  return [dev newTextureWithDescriptor:td];
}

static void ReadTextureToRGBA8(
    id<MTLDevice> dev,
    id<MTLCommandQueue> q,
    id<MTLTexture> tex,
    std::vector<uint8_t>& out_bytes) {
  const uint32_t w = static_cast<uint32_t>(tex.width);
  const uint32_t h = static_cast<uint32_t>(tex.height);
  const size_t row_bytes = ((static_cast<size_t>(w) * 4u + 255u) & ~size_t(255u));
  id<MTLBuffer> rbuf = [dev newBufferWithLength:row_bytes * h options:MTLResourceStorageModeShared];
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLBlitCommandEncoder> blit = [cb blitCommandEncoder];
  [blit copyFromTexture:tex
            sourceSlice:0
            sourceLevel:0
           sourceOrigin:MTLOriginMake(0, 0, 0)
             sourceSize:MTLSizeMake(w, h, 1)
               toBuffer:rbuf
      destinationOffset:0
 destinationBytesPerRow:row_bytes
destinationBytesPerImage:row_bytes * h];
  [blit endEncoding];
  [cb commit];
  [cb waitUntilCompleted];

  out_bytes.resize(static_cast<size_t>(w) * h * 4u);
  const uint8_t* src = static_cast<const uint8_t*>(rbuf.contents);
  const bool is_bgra = (tex.pixelFormat == MTLPixelFormatBGRA8Unorm || tex.pixelFormat == MTLPixelFormatBGRA8Unorm_sRGB);
  for (uint32_t y = 0; y < h; ++y) {
    const uint8_t* rptr = src + static_cast<size_t>(y) * row_bytes;
    for (uint32_t x = 0; x < w; ++x) {
      size_t di = (static_cast<size_t>(y) * w + x) * 4u;
      if (is_bgra) {
        out_bytes[di + 0u] = rptr[x * 4u + 2u];
        out_bytes[di + 1u] = rptr[x * 4u + 1u];
        out_bytes[di + 2u] = rptr[x * 4u + 0u];
        out_bytes[di + 3u] = rptr[x * 4u + 3u];
      } else {
        std::memcpy(&out_bytes[di], &rptr[x * 4u], 4u);
      }
    }
  }
}

static double ComputeMeanL1(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b, uint32_t w, uint32_t h) {
  double sum = 0.0;
  const size_t n = static_cast<size_t>(w) * h;
  for (size_t i = 0; i < n; ++i) {
    sum += std::abs(int(a[i * 4u + 0u]) - int(b[i * 4u + 0u])) / 255.0;
    sum += std::abs(int(a[i * 4u + 1u]) - int(b[i * 4u + 1u])) / 255.0;
    sum += std::abs(int(a[i * 4u + 2u]) - int(b[i * 4u + 2u])) / 255.0;
  }
  return sum / static_cast<double>(n * 3u);
}

int main(int argc, char** argv) {
  @autoreleasepool {
    if (argc < 3) {
      std::fprintf(stderr, "usage: bridge_harness <mode> <bridge_dylib_path>\n");
      return 2;
    }
    const std::string mode = argv[1];
    const char* dylib_path = argv[2];

    Method m_temp = class_getInstanceMethod([MTLFXTemporalScalerDescriptor class], @selector(newTemporalScalerWithDevice:));
    Method m_spat = class_getInstanceMethod([MTLFXSpatialScalerDescriptor class], @selector(newSpatialScalerWithDevice:));
    Method m_layer = class_getInstanceMethod([CAMetalLayer class], @selector(setDevice:));
    IMP imp_temp_before = m_temp ? method_getImplementation(m_temp) : nullptr;
    IMP imp_spat_before = m_spat ? method_getImplementation(m_spat) : nullptr;
    IMP imp_layer_before = m_layer ? method_getImplementation(m_layer) : nullptr;

    void* handle = dlopen(dylib_path, RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
      std::fprintf(stderr, "dlopen failed: %s\n", dlerror());
      return 3;
    }
    auto get_mask = reinterpret_cast<HookMaskFn>(dlsym(handle, "NetalNRFastBridgeInstalledHookMask"));
    auto get_pub = reinterpret_cast<PublishedCountFn>(dlsym(handle, "NetalNRFastBridgePublishedCount"));
    auto get_comp = reinterpret_cast<PublishedCountFn>(dlsym(handle, "NetalNRFastBridgeCompletedCount"));
    if (!get_mask || !get_pub || !get_comp) {
      std::fprintf(stderr, "missing bridge C symbols\n");
      return 4;
    }

    if (mode == "check_unhooked") {
      IMP imp_temp_after = m_temp ? method_getImplementation(m_temp) : nullptr;
      IMP imp_spat_after = m_spat ? method_getImplementation(m_spat) : nullptr;
      IMP imp_layer_after = m_layer ? method_getImplementation(m_layer) : nullptr;
      const bool untouched = (imp_temp_before == imp_temp_after) &&
                             (imp_spat_before == imp_spat_after) &&
                             (imp_layer_before == imp_layer_after) &&
                             (get_mask() == 0u);
      std::printf("{\"hook_mask\":%u,\"imps_untouched\":%s}\n",
                  get_mask(), untouched ? "true" : "false");
      return untouched ? 0 : 1;
    }

    id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
    if (!dev || ![dev supportsFamily:MTLGPUFamilyApple7]) {
      std::printf("{\"skipped\":true,\"reason\":\"Apple7+ GPU required\"}\n");
      return 0;
    }
    id<MTLCommandQueue> q = [dev newCommandQueue];

    if (mode == "spatial_baseline" || mode == "spatial_nr_inline") {
      if (![MTLFXSpatialScalerDescriptor supportsDevice:dev]) {
        std::printf("{\"skipped\":true,\"reason\":\"MTLFXSpatialScaler unsupported on device\"}\n");
        return 0;
      }
      const uint32_t in_w = 256, in_h = 128;
      const uint32_t out_w = 256, out_h = 128;
      MTLFXSpatialScalerDescriptor* desc = [[MTLFXSpatialScalerDescriptor alloc] init];
      desc.inputWidth = in_w;
      desc.inputHeight = in_h;
      desc.outputWidth = out_w;
      desc.outputHeight = out_h;
      desc.colorTextureFormat = MTLPixelFormatRGBA8Unorm;
      desc.outputTextureFormat = MTLPixelFormatRGBA8Unorm;
      desc.colorProcessingMode = MTLFXSpatialScalerColorProcessingModePerceptual;

      id<MTLFXSpatialScaler> scaler = [desc newSpatialScalerWithDevice:dev];
      if (!scaler) {
        std::fprintf(stderr, "newSpatialScalerWithDevice returned nil\n");
        return 6;
      }

      id<MTLTexture> in_tex = MakeSharedRGBA8(dev, in_w, in_h, scaler.colorTextureUsage | MTLTextureUsageShaderRead);
      id<MTLTexture> out_tex = MakeSharedRGBA8(
          dev, out_w, out_h, scaler.outputTextureUsage | MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite);

      std::vector<uint8_t> frame0_out, frame3_out;
      for (int f = 0; f < 4; ++f) {
        std::vector<uint8_t> src_rgba;
        FillMovingPatternRGBA8(src_rgba, in_w, in_h, f * 6, -f * 4);
        [in_tex replaceRegion:MTLRegionMake2D(0, 0, in_w, in_h)
                  mipmapLevel:0
                    withBytes:src_rgba.data()
                  bytesPerRow:in_w * 4u];

        scaler.colorTexture = in_tex;
        scaler.outputTexture = out_tex;
        id<MTLCommandBuffer> cb = [q commandBuffer];
        [scaler encodeToCommandBuffer:cb];
        [cb commit];
        [cb waitUntilCompleted];
        usleep(3000);  // Ensure async completion block records telemetry cleanly

        if (f == 0) ReadTextureToRGBA8(dev, q, out_tex, frame0_out);
        if (f == 3) ReadTextureToRGBA8(dev, q, out_tex, frame3_out);
      }

      if (const char* dump_bin = std::getenv("HARNESS_DUMP_OUT"); dump_bin && *dump_bin) {
        FILE* fp = std::fopen(dump_bin, "wb");
        if (fp) {
          std::fwrite(frame3_out.data(), 1, frame3_out.size(), fp);
          std::fclose(fp);
        }
      }

      const double temporal_motion_l1 = ComputeMeanL1(frame0_out, frame3_out, out_w, out_h);
      std::printf("{\"hook_mask\":%u,\"published\":%llu,\"temporal_motion_l1\":%.6f}\n",
                  get_mask(),
                  static_cast<unsigned long long>(get_pub()),
                  temporal_motion_l1);
      return 0;
    }

    if (mode == "drawable_mtl4") {
      if (@available(macOS 26.0, *)) {
        CAMetalLayer* layer = [CAMetalLayer layer];
        layer.framebufferOnly = YES;
        [layer setDevice:dev];
        const bool fb_only_cleared = (layer.framebufferOnly == NO);

        NSError* err = nil;
        MTL4CommandQueueDescriptor* qd = [MTL4CommandQueueDescriptor new];
        id<MTL4CommandQueue> q4 = [dev newMTL4CommandQueueWithDescriptor:qd error:&err];
        if (!q4) {
          std::printf("{\"skipped\":true,\"reason\":\"MTL4CommandQueue unsupported on device\"}\n");
          return 0;
        }
        id<MTLSharedEvent> fence_ev = [dev newSharedEvent];
        uint64_t fence_val = 0;

        const uint32_t w = 256, h = 128;
        // Strict RenderTarget-only texture (usage == MTLTextureUsageRenderTarget, ShaderWrite == 0)!
        id<MTLTexture> rt_tex = MakePrivateRenderTargetOnlyBGRA8(dev, w, h);
        const bool lacks_shader_write = ((rt_tex.usage & MTLTextureUsageShaderWrite) == 0);

        id<MTLTexture> upload_tex = MakeSharedRGBA8(dev, w, h, MTLTextureUsageShaderRead);
        // Build a tiny helper render pipeline to draw moving frames into `rt_tex` just like a real game!
        NSString* game_src =
            @"#include <metal_stdlib>\n"
             "using namespace metal;\n"
             "struct V { float4 p [[position]]; };\n"
             "vertex V vs(uint vid [[vertex_id]]) {\n"
             "  V o; float2 uv = float2((vid << 1u) & 2u, vid & 2u);\n"
             "  o.p = float4(uv * float2(2.0f, -2.0f) + float2(-1.0f, 1.0f), 0.0f, 1.0f); return o;\n"
             "}\n"
             "fragment float4 fs(V in [[stage_in]], texture2d<float, access::read> t [[texture(0)]]) {\n"
             "  return float4(t.read(uint2(in.p.xy)).rgb, 1.0f);\n"
             "}\n";
        id<MTLLibrary> glib = [dev newLibraryWithSource:game_src options:[[MTLCompileOptions alloc] init] error:&err];
        MTLRenderPipelineDescriptor* grpd = [[MTLRenderPipelineDescriptor alloc] init];
        grpd.vertexFunction = [glib newFunctionWithName:@"vs"];
        grpd.fragmentFunction = [glib newFunctionWithName:@"fs"];
        grpd.colorAttachments[0].pixelFormat = rt_tex.pixelFormat;
        id<MTLRenderPipelineState> game_pso = [dev newRenderPipelineStateWithDescriptor:grpd error:&err];

        NetalMockDrawable* drawable = [[NetalMockDrawable alloc] init];
        drawable.texture = rt_tex;
        drawable.layer = layer;

        const bool is_async = (std::getenv("NETAL_NR_ASYNC") && std::strcmp(std::getenv("NETAL_NR_ASYNC"), "1") == 0);
        int modified_frames = 0;
        double last_diff_vs_raw = 0.0;

        for (int f = 0; f < 6; ++f) {
          std::vector<uint8_t> raw_rgba;
          FillMovingPatternRGBA8(raw_rgba, w, h, f * 4, -f * 2);
          [upload_tex replaceRegion:MTLRegionMake2D(0, 0, w, h)
                        mipmapLevel:0
                          withBytes:raw_rgba.data()
                        bytesPerRow:w * 4u];

          // Render simulated game frame into RenderTarget-only `rt_tex` and order before `q4` via event.
          id<MTLCommandBuffer> gcb = [q commandBuffer];
          MTLRenderPassDescriptor* pass = [MTLRenderPassDescriptor renderPassDescriptor];
          pass.colorAttachments[0].texture = rt_tex;
          pass.colorAttachments[0].loadAction = MTLLoadActionDontCare;
          pass.colorAttachments[0].storeAction = MTLStoreActionStore;
          id<MTLRenderCommandEncoder> renc = [gcb renderCommandEncoderWithDescriptor:pass];
          [renc setRenderPipelineState:game_pso];
          [renc setFragmentTexture:upload_tex atIndex:0];
          [renc drawPrimitives:MTLPrimitiveTypeTriangle vertexStart:0 vertexCount:3];
          [renc endEncoding];

          const uint64_t pre_v = ++fence_val;
          [gcb encodeSignalEvent:fence_ev value:pre_v];
          [gcb commit];

          [q4 waitForEvent:fence_ev value:pre_v];
          [q4 signalDrawable:drawable];

          // Wait for `q4` (which waits on our bridge's `v2` signal!) so `rt_tex` writeback is complete.
          const uint64_t post_v = ++fence_val;
          [q4 signalEvent:fence_ev value:post_v];
          id<MTLCommandBuffer> wait_cb = [q commandBuffer];
          [wait_cb encodeWaitForEvent:fence_ev value:post_v];
          [wait_cb commit];
          [wait_cb waitUntilCompleted];

          if (is_async && f == 0) {
            // Deterministically poll until background worker slot 0 retires (fast ~4ms on M4 Max,
            // up to 2s ceiling on smaller GPUs) so frames 1..5 exercise encode_compose_delta!
            for (int spin = 0; spin < 1000 && get_comp() == 0; ++spin) {
              usleep(2000);
            }
          } else {
            usleep(3000);
          }

          std::vector<uint8_t> post_rgba;
          ReadTextureToRGBA8(dev, q, rt_tex, post_rgba);
          double diff = ComputeMeanL1(raw_rgba, post_rgba, w, h);
          if (diff > 1e-4) {
            ++modified_frames;
          }
          last_diff_vs_raw = diff;
        }

        std::printf(
            "{\"hook_mask\":%u,\"framebuffer_only_cleared\":%s,\"lacks_shader_write\":%s,"
            "\"published\":%llu,\"modified_frames\":%d,\"last_diff_vs_raw\":%.6f}\n",
            get_mask(),
            fb_only_cleared ? "true" : "false",
            lacks_shader_write ? "true" : "false",
            static_cast<unsigned long long>(get_pub()),
            modified_frames,
            last_diff_vs_raw);
        return 0;
      }
      return 8;
    }

    return 9;
  }
}
'''


class NetalNrFastBridgeTestSuite(unittest.TestCase):
    _tmpdir: tempfile.TemporaryDirectory[str]
    _bridge_dir: Path
    _dylib: Path
    _harness_bin: Path

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmpdir = tempfile.TemporaryDirectory(prefix='netal-nr-bridge-ci-')
        root = Path(cls._tmpdir.name)
        cls._bridge_dir = root / 'bridge_build'
        proc = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), '--output', str(cls._bridge_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f'build-netal-nr-fast-bridge.py failed:\n{proc.stdout}\n{proc.stderr}')
        cls._dylib = cls._bridge_dir / 'libNetalNRFastBridge.dylib'

        harness_mm = root / 'bridge_harness.mm'
        harness_mm.write_text(HARNESS_MM, encoding='utf-8')
        cls._harness_bin = root / 'nxnative_launcher_probe'
        subprocess.check_call(
            [
                'xcrun', '--sdk', 'macosx', 'clang++',
                '-std=c++20', '-arch', 'arm64', '-fobjc-arc',
                '-O2', '-Wall', '-Wextra', '-Werror',
                '-framework', 'Foundation',
                '-framework', 'Metal',
                '-framework', 'MetalFX',
                '-framework', 'QuartzCore',
                str(harness_mm),
                '-o', str(cls._harness_bin),
            ]
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmpdir.cleanup()

    def _base_env(self) -> dict[str, str]:
        env = dict(os.environ)
        for k in list(env.keys()):
            if k.startswith('NETAL_NR_') or k == 'NEURALSTUDIO_TARGET_EXE' or k == 'HARNESS_DUMP_OUT':
                env.pop(k, None)
        return env

    def test_01_build_receipt_and_atomic_refusal(self) -> None:
        receipt_path = self._bridge_dir / 'build-receipt.json'
        self.assertTrue(receipt_path.is_file())
        receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
        self.assertEqual(receipt['schema'], 'netal.nr.fast-bridge-build.v1')
        expected = {
            'nr_fast_kernels.metallib',
            'netal_nr_optical_flow.metallib',
            'libnetal_nr_fast.a',
            'libNetalNRFastBridge.dylib',
        }
        found = set()
        for art in receipt['artifacts']:
            p = self._bridge_dir / art['name']
            self.assertTrue(p.is_file(), f"missing artifact {art['name']}")
            raw = p.read_bytes()
            self.assertEqual(len(raw), art['bytes'])
            self.assertEqual(hashlib.sha256(raw).hexdigest(), art['sha256'])
            found.add(art['name'])
        self.assertEqual(found, expected)

        collide = subprocess.run(
            [sys.executable, str(BUILD_SCRIPT), '--output', str(self._bridge_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(collide.returncode, 0)

    def test_02_default_off_and_non_target_process_install_zero_hooks(self) -> None:
        # 1. Default unset NETAL_NR_ADAPTER_MODE -> Off -> zero hooks, no log created.
        log_off = Path(self._tmpdir.name) / 'should_not_exist.log'
        env_off = self._base_env()
        env_off['NETAL_NR_LOG'] = str(log_off)
        res_off = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'check_unhooked', str(self._dylib)],
                env=env_off,
                text=True,
            )
        )
        self.assertEqual(res_off['hook_mask'], 0)
        self.assertTrue(res_off['imps_untouched'])
        self.assertFalse(log_off.exists())

        # 2. NETAL_NR_ADAPTER_MODE=continuous with non-matching NEURALSTUDIO_TARGET_EXE=dmhost,re9.exe
        #    while binary is named `nxnative_launcher_probe` -> zero hooks!
        env_skip = self._base_env()
        env_skip['NETAL_NR_ADAPTER_MODE'] = 'continuous'
        env_skip['NETAL_NR_HOOK_SPATIAL'] = '1'
        env_skip['NETAL_NR_HOOK_DRAWABLE'] = '1'
        env_skip['NEURALSTUDIO_TARGET_EXE'] = 'dmhost,re9.exe'
        env_skip['NETAL_NR_LOG'] = str(log_off)
        res_skip = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'check_unhooked', str(self._dylib)],
                env=env_skip,
                text=True,
            )
        )
        self.assertEqual(res_skip['hook_mask'], 0)
        self.assertTrue(res_skip['imps_untouched'])
        self.assertFalse(log_off.exists())

    def test_03_spatial_scaler_continuous_inline_moving_frames(self) -> None:
        work = Path(self._tmpdir.name) / 'spatial'
        work.mkdir(exist_ok=True)
        base_out = work / 'baseline_f3.rgba'
        nr_out = work / 'nr_f3.rgba'
        nr_log = work / 'spatial_nr.log'

        # Baseline unhooked MetalFX SpatialScaler run:
        env_base = self._base_env()
        env_base['HARNESS_DUMP_OUT'] = str(base_out)
        r_base = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'spatial_baseline', str(self._dylib)],
                env=env_base,
                text=True,
            )
        )
        if r_base.get('skipped'):
            self.skipTest(r_base.get('reason', 'GPU unsupported'))
        self.assertEqual(r_base['hook_mask'], 0)
        self.assertEqual(r_base['published'], 0)

        # Hooked continuous inline NR run with synthetic Small31 + GPU optical flow:
        env_nr = self._base_env()
        env_nr['NETAL_NR_ADAPTER_MODE'] = 'continuous'
        env_nr['NETAL_NR_HOOK_SPATIAL'] = '1'
        env_nr['NETAL_NR_SYNTHETIC'] = 'small'
        env_nr['NETAL_NR_SYNC_PREPARE'] = '1'
        env_nr['NETAL_NR_ASYNC'] = '0'
        env_nr['NEURALSTUDIO_TARGET_EXE'] = 'dmhost,nxnative_launcher_probe'
        env_nr['NETAL_NR_LOG'] = str(nr_log)
        env_nr['HARNESS_DUMP_OUT'] = str(nr_out)
        r_nr = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'spatial_nr_inline', str(self._dylib)],
                env=env_nr,
                text=True,
            )
        )
        self.assertEqual(r_nr['published'], 4)
        self.assertGreater(r_nr['temporal_motion_l1'], 1e-3)

        b_bytes = base_out.read_bytes()
        n_bytes = nr_out.read_bytes()
        self.assertEqual(len(b_bytes), len(n_bytes))
        diff_vs_metalfx = sum(abs(a - b) for a, b in zip(b_bytes, n_bytes)) / (len(b_bytes) * 255.0)
        self.assertGreater(diff_vs_metalfx, 1e-3)

        log_text = nr_log.read_text(encoding='utf-8')
        self.assertIn('event=bridge_ready', log_text)
        self.assertIn('event=nr_prepare status=ok topology=small blocks=31 synthetic=1', log_text)
        self.assertIn('flow_ready=1', log_text)
        self.assertIn('event=nr_frame_published', log_text)
        self.assertIn('seam=spatial mode=inline', log_text)
        self.assertIn('synth_motion=1', log_text)
        self.assertIn('event=nr_pacing', log_text)

    def test_04_mtl4_signal_drawable_inline_and_async_rendertarget_only(self) -> None:
        work = Path(self._tmpdir.name) / 'drawable'
        cap_dir = work / 'pair_caps'
        work.mkdir(exist_ok=True)

        # 1. Inline mode (6 frames on RenderTarget-only texture without ShaderWrite + pair capture on frame 2):
        inline_log = work / 'drawable_inline.log'
        env_in = self._base_env()
        env_in['NETAL_NR_ADAPTER_MODE'] = 'continuous'
        env_in['NETAL_NR_HOOK_DRAWABLE'] = '1'
        env_in['NETAL_NR_SYNTHETIC'] = 'small'
        env_in['NETAL_NR_SYNC_PREPARE'] = '1'
        env_in['NETAL_NR_ASYNC'] = '0'
        env_in['NETAL_NR_CAPTURE_PAIR_DIR'] = str(cap_dir)
        env_in['NETAL_NR_CAPTURE_FRAMES'] = '2'
        env_in['NETAL_NR_LOG'] = str(inline_log)
        r_in = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'drawable_mtl4', str(self._dylib)],
                env=env_in,
                text=True,
            )
        )
        if r_in.get('skipped'):
            self.skipTest(r_in.get('reason', 'GPU unsupported'))
        self.assertTrue(r_in['framebuffer_only_cleared'])
        self.assertTrue(r_in['lacks_shader_write'])
        self.assertEqual(r_in['published'], 6)
        self.assertEqual(r_in['modified_frames'], 6)
        self.assertGreater(r_in['last_diff_vs_raw'], 1e-3)
        self.assertTrue((cap_dir / 'raw_00002.png').is_file())
        self.assertTrue((cap_dir / 'nr_00002.png').is_file())
        self.assertNotEqual((cap_dir / 'raw_00002.png').read_bytes(), (cap_dir / 'nr_00002.png').read_bytes())

        # 2. Async 60 Hz non-blocking mode (`NETAL_NR_ASYNC=1`, detail-split blur active):
        async_log = work / 'drawable_async.log'
        env_as = self._base_env()
        env_as['NETAL_NR_ADAPTER_MODE'] = 'continuous'
        env_as['NETAL_NR_HOOK_DRAWABLE'] = '1'
        env_as['NETAL_NR_SYNTHETIC'] = 'small'
        env_as['NETAL_NR_SYNC_PREPARE'] = '1'
        env_as['NETAL_NR_ASYNC'] = '1'
        env_as['NETAL_NR_DETAIL'] = '2.0'
        env_as['NETAL_NR_COLOUR'] = '0.8'
        env_as['NETAL_NR_LOG'] = str(async_log)
        r_as = json.loads(
            subprocess.check_output(
                [str(self._harness_bin), 'drawable_mtl4', str(self._dylib)],
                env=env_as,
                text=True,
            )
        )
        self.assertTrue(r_as['framebuffer_only_cleared'])
        self.assertTrue(r_as['lacks_shader_write'])
        # Frame 0 dispatches background worker; frames 1..5 compose cached deltas via encode_compose_delta!
        self.assertGreaterEqual(r_as['published'], 4)
        self.assertGreaterEqual(r_as['modified_frames'], 4)
        self.assertGreater(r_as['last_diff_vs_raw'], 1e-3)

        alog_text = async_log.read_text(encoding='utf-8')
        self.assertIn('seam=drawable mode=async_compose', alog_text)
        self.assertIn('event=nr_pacing mode=async', alog_text)


if __name__ == '__main__':
    unittest.main(verbosity=2)
