#!/usr/bin/env python3
"""Model-free fast-output opt-in guards; --gpu also runs native Metal helper fixtures.

The GPU fixtures check native math bounds, quantized/clamped bilinear sampling,
CUDA gather order and ignored/null contract tables. They do not test the model,
history quality, NVIDIA parity, iPhone execution or a performance claim.
"""
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('boundary', ROOT / 'scripts/build-netal-sr-boundary-kernels.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)
SHADER = ROOT / builder.FAST_OUTPUT_SOURCE[0]
RUN_GPU = '--gpu' in sys.argv
if RUN_GPU:
    sys.argv.remove('--gpu')

METAL_FIXTURE = r'''
kernel void netal_fast_output_fixture(device const float4* inputs [[buffer(0)]],
    device const CmTexture2DDescriptor* descriptors [[buffer(1)]],
    device float4* results [[buffer(2)]], uint id [[thread_position_in_grid]]) {
    float4 a = inputs[id * 2], uv = inputs[id * 2 + 1];
    results[id * 4] = float4(cm_lg2_approx_ftz_f32(a.x, nullptr, nullptr, nullptr),
        cm_ex2_approx_ftz_f32(a.y, nullptr, nullptr, nullptr),
        cm_precise_rsqrt(a.x, nullptr), cm_rcp_approx_ftz_f32(a.w, nullptr));
    half2 h = as_type<half2>(cm_half2_ex2(as_type<uint>(half2(a.y, a.z)), nullptr));
    results[id * 4 + 1] = float4(cm_div_approx_ftz_f32(a.z, a.w, nullptr), float2(h), 0);
    results[id * 4 + 2] = cm_tex_level_2d_f32(ulong(&descriptors[id]), uv.x, uv.y, 0, nullptr);
    results[id * 4 + 3] = cm_tld4_r_2d_f32(ulong(&descriptors[id]), uv.x, uv.y);
}
'''

HOST_FIXTURE = r'''
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
struct Descriptor { uint64_t data; uint32_t width, height, format, channels; };
struct V4 { float x, y, z, w; };
static bool near(float got, float expected, float relative = 0.00004f) {
    return std::isfinite(got) && std::abs(got-expected) <= relative*std::max(1.0f,std::abs(expected));
}
int main(int argc, char** argv) { @autoreleasepool {
    if (argc != 2) return 2;
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { std::fprintf(stderr,"no Metal device\n"); return 3; }
    NSError* error = nil;
    id<MTLLibrary> library = [device newLibraryWithURL:[NSURL fileURLWithPath:@(argv[1])] error:&error];
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:[library newFunctionWithName:@"netal_fast_output_fixture"] error:&error];
    if (!pipeline) { std::fprintf(stderr,"pipeline: %s\n",error.description.UTF8String); return 4; }
    // Distinct channels/rows expose gather permutations; one-texel edges and zero
    // channels exercise clamping/zero-fill. Every sample uses null arithmetic ROMs.
    constexpr uint32_t count = 60, stride = 128;
    id<MTLBuffer> data = [device newBufferWithLength:count*stride options:MTLResourceStorageModeShared];
    id<MTLBuffer> desc = [device newBufferWithLength:count*sizeof(Descriptor) options:MTLResourceStorageModeShared];
    id<MTLBuffer> input = [device newBufferWithLength:count*2*sizeof(V4) options:MTLResourceStorageModeShared];
    id<MTLBuffer> output = [device newBufferWithLength:count*4*sizeof(V4) options:MTLResourceStorageModeShared];
    auto* descriptors = static_cast<Descriptor*>(desc.contents);
    auto* values = static_cast<V4*>(input.contents);
    const float coords[5][2] = {{0.25f,0.25f},{0.5f,0.5f},{-0.5f,0.75f},{1.8f,-0.25f},{0.251f,0.749f}};
    for (uint32_t n=0; n<count; ++n) {
        uint32_t format = (n/5)%2 ? 16 : 32, channels = (n/10)%3 == 0 ? 4 : (n/10)%3 == 1 ? 1 : 0;
        uint32_t width = n>=30 ? 1 : 2, height = n>=30 ? 1 : 2;
        descriptors[n] = {data.gpuAddress+n*stride,width,height,format,channels};
        auto* bytes = static_cast<uint8_t*>(data.contents)+n*stride;
        for (uint32_t j=0; j<width*height*channels; ++j) {
            float value = float(int(j/channels)*10+int(j%channels)*2-15)*0.25f;
            if (format==16) { _Float16 h=value; std::memcpy(bytes+j*2,&h,2); }
            else std::memcpy(bytes+j*4,&value,4);
        }
        values[n*2] = {std::exp2(float(int(n%15)-7)*0.5f),float(int(n%13)-6)*0.5f,
                       float(int(n%9)-4)*0.5f,float(n%2 ? -1 : 1)*(0.25f+float(n%7)*0.5f)};
        values[n*2+1] = {coords[n%5][0],coords[n%5][1],0,0};
    }
    id<MTLCommandBuffer> command = [[device newCommandQueue] commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [command computeCommandEncoder];
    [encoder setComputePipelineState:pipeline]; [encoder setBuffer:input offset:0 atIndex:0];
    [encoder setBuffer:desc offset:0 atIndex:1]; [encoder setBuffer:output offset:0 atIndex:2];
    [encoder useResource:data usage:MTLResourceUsageRead];
    [encoder dispatchThreads:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(1,1,1)];
    [encoder endEncoding]; [command commit]; [command waitUntilCompleted];
    if (command.status != MTLCommandBufferStatusCompleted) { std::fprintf(stderr,"GPU: %s\n",command.error.description.UTF8String); return 5; }
    auto* got = reinterpret_cast<float*>(output.contents);
    uint32_t checks = 0, failures = 0;
    auto check = [&](uint32_t n,uint32_t lane,float expected,float tolerance=0.00004f) {
        ++checks; if (!near(got[n*16+lane],expected,tolerance)) {
            if (failures++<8) std::fprintf(stderr,"case=%u lane=%u got=%.9g expected=%.9g\n",n,lane,got[n*16+lane],expected);
        }
    };
    for (uint32_t n=0; n<count; ++n) {
        V4 a=values[n*2], uv=values[n*2+1]; Descriptor d=descriptors[n];
        check(n,0,std::log2(a.x)); check(n,1,std::exp2(a.y));
        check(n,2,1/std::sqrt(a.x)); check(n,3,1/a.w); check(n,4,a.z/a.w);
        check(n,5,float(_Float16(std::exp2(float(_Float16(a.y))))),0.001f);
        check(n,6,float(_Float16(std::exp2(float(_Float16(a.z))))),0.001f);
        check(n,7,0);
        float x=std::nearbyint((uv.x*d.width-0.5f)*256)/256;
        float y=std::nearbyint((uv.y*d.height-0.5f)*256)/256;
        int x0=int(std::floor(x)), y0=int(std::floor(y));
        auto texel = [&](int tx,int ty,uint32_t channel) {
            if (channel>=d.channels) return 0.0f;
            tx=std::clamp(tx,0,int(d.width)-1); ty=std::clamp(ty,0,int(d.height)-1);
            uint32_t index=(ty*d.width+tx)*d.channels+channel;
            const auto* bytes=static_cast<const uint8_t*>(data.contents)+n*stride;
            if (d.format==16) { _Float16 h; std::memcpy(&h,bytes+index*2,2); return float(h); }
            float f; std::memcpy(&f,bytes+index*4,4); return f;
        };
        for (uint32_t c=0; c<4; ++c) {
            // Four explicit weights are independent of the shader's nested mix.
            float tx=x-x0,ty=y-y0;
            float expected=(1-tx)*(1-ty)*texel(x0,y0,c)+tx*(1-ty)*texel(x0+1,y0,c)+
                           (1-tx)*ty*texel(x0,y0+1,c)+tx*ty*texel(x0+1,y0+1,c);
            check(n,8+c,expected);
        }
        check(n,12,texel(x0,y0+1,0)); check(n,13,texel(x0+1,y0+1,0));
        check(n,14,texel(x0+1,y0,0)); check(n,15,texel(x0,y0,0));
    }
    std::printf("native_helpers checks=%u failures=%u model_inputs=0 iphone_validated=false\n",checks,failures);
    return failures ? 6 : 0;
} }
'''


class FastOutputTests(unittest.TestCase):
    def run_builder(self, fast):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'output'
            calls = []

            def run(command, **_):
                calls.append(command)
                Path(command[command.index('-o') + 1]).write_bytes(b'procedural fixture')

            argv = ['boundary', '--only', 'hiluma_output_depthinv_exact', '--output', str(output)]
            if fast:
                argv.append('--fast-output')
            with patch.object(sys, 'argv', argv), patch.object(builder, 'metal_toolchain', return_value=(
                    Path('/tools/metal'), Path('/tools/metallib'), [], {'sdk': 'macosx'})), \
                    patch.object(builder.subprocess, 'run', side_effect=run), \
                    patch.object(builder.subprocess, 'check_output', return_value='fixture compiler'), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(builder.main(), 0)
            return json.loads((output / 'build-receipt.json').read_text()), calls

    def test_default_build_does_not_enable_approximation(self):
        receipt, calls = self.run_builder(False)
        self.assertEqual([k['library'] for k in receipt['kernels']], ['hiluma_output_depthinv_exact.metallib'])
        self.assertFalse(any('NETAL_SR_FAST_OUTPUT' in arg for call in calls for arg in call))

    def test_opt_in_retains_reference_and_labels_approximation(self):
        receipt, calls = self.run_builder(True)
        exact, fast = receipt['kernels']
        self.assertEqual(exact['source'], fast['source'])
        self.assertEqual(fast['library'], 'hiluma_output_depthinv_fast.metallib')
        self.assertIn('-DNETAL_SR_FAST_OUTPUT=1', fast['flags'])
        self.assertNotIn('-ffast-math', fast['flags'])
        self.assertIn('approximate', fast['arithmetic'])
        self.assertFalse(receipt['inference_validated'])

    def test_kernel_abi_control_flow_history_and_gather_retained(self):
        source = SHADER.read_text()
        kernel = source[source.index('kernel void '):]
        gather = source[source.index('static inline float4 cm_tld4_r_2d_f32('):]
        gather = gather[:gather.index('\nstatic inline void cm_sust_b_2d_b32_zero')]
        # Frozen retained sections: this approximation must not change algorithm,
        # history, stores, parameter offsets, bounds or the load-bearing gather order.
        self.assertEqual(hashlib.sha256(kernel.encode()).hexdigest(),
                         '8b53479c0965a4fafd40d13f1763a4f21efbc75e9b771b07886f1173022d1487')
        self.assertEqual(hashlib.sha256(gather.encode()).hexdigest(),
                         'edb07f0257534764d15e28a51af119a96d69f3b3e4fe4a70ca770ec838896559')

    @unittest.skipUnless(RUN_GPU and sys.platform == 'darwin' and shutil.which('xcrun'),
                         'pass --gpu on a Mac for native Metal helper fixtures')
    def test_gpu_math_sampling_and_ignored_roms(self):
        with tempfile.TemporaryDirectory(prefix='netal-fast-output-fixtures-') as temporary:
            directory = Path(temporary)
            metal = directory / 'fixture.metal'
            metal.write_text('#include "' + str(SHADER) + '"\n' + METAL_FIXTURE)
            host = directory / 'fixture.mm'
            host.write_text(HOST_FIXTURE)
            subprocess.run(['xcrun', '--sdk', 'macosx', 'metal', '-std=metal4.0', '-fmetal-math-mode=safe',
                            '-DNETAL_SR_FAST_OUTPUT=1', '-c', str(metal), '-o', str(directory / 'fixture.air')],
                           check=True, capture_output=True, text=True)
            subprocess.run(['xcrun', '--sdk', 'macosx', 'metallib', str(directory / 'fixture.air'),
                            '-o', str(directory / 'fixture.metallib')], check=True)
            subprocess.run(['xcrun', '--sdk', 'macosx', 'clang++', '-std=c++17', '-fobjc-arc', str(host),
                            '-framework', 'Foundation', '-framework', 'Metal', '-o', str(directory / 'fixture')],
                           check=True, capture_output=True, text=True)
            result = subprocess.run([str(directory / 'fixture'), str(directory / 'fixture.metallib')],
                                    check=True, capture_output=True, text=True)
            self.assertIn('failures=0', result.stdout)
            print(result.stdout.strip())


if __name__ == '__main__':
    unittest.main()
