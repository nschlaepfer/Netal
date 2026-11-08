#!/usr/bin/env python3
"""Exercise unavailable timing, percentiles and unsigned/physical-only build contracts."""
import importlib.util
import hashlib
import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('probe_builder', ROOT / 'scripts/build-netal-ios-sr-probe.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class ProbeTests(unittest.TestCase):
    def test_mixed_fp8_descriptor_rejections(self):
        compiler = shutil.which('clang++')
        if not compiler or not Path('/System/Library/Frameworks/Foundation.framework').exists():
            self.skipTest('Apple Foundation compiler unavailable')
        with tempfile.TemporaryDirectory() as temporary:
            binary = Path(temporary) / 'fp8-audit'
            subprocess.run([compiler, '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                            str(ROOT / 'src/ios_probe/NetalProbeFP8AuditTest.mm'),
                            '-framework', 'Foundation', '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)
    def test_unsigned_project_and_no_embedded_identity(self):
        settings = builder.xcode_spec('org.example.probe', '26.0')['targets']['NetalSRProbe']['settings']['base']
        self.assertEqual(settings['CODE_SIGNING_ALLOWED'], 'NO')
        self.assertNotIn('DEVELOPMENT_TEAM', settings)
        self.assertEqual(settings['SUPPORTED_PLATFORMS'], 'iphoneos')
        self.assertEqual(settings['LIBRARY_SEARCH_PATHS'], '$(SRCROOT)/Library/lib')
        self.assertNotIn('model', json.dumps(builder.info_plist('org.example.probe', '26.0')).lower())

    def test_scene_manifest_matches_programmatic_delegate(self):
        manifest = builder.info_plist('org.example.probe', '26.0')['UIApplicationSceneManifest']
        self.assertFalse(manifest['UIApplicationSupportsMultipleScenes'])
        scene = manifest['UISceneConfigurations']['UIWindowSceneSessionRoleApplication']
        self.assertEqual(len(scene), 1)
        self.assertEqual(scene[0]['UISceneConfigurationName'], 'Default Configuration')
        self.assertEqual(scene[0]['UISceneDelegateClassName'], 'NetalProbeSceneDelegate')

    def test_plist_identifies_actual_probe_sources(self):
        hashes = builder.info_plist('org.example.probe', '26.0')['NetalProbeSourceSHA256']
        self.assertEqual(set(hashes), set(builder.SOURCES))
        for name in builder.SOURCES:
            self.assertEqual(hashes[name], hashlib.sha256((ROOT / 'src/ios_probe' / name).read_bytes()).hexdigest())

    def test_optional_lab_is_separate_and_requires_ios27(self):
        normal = builder.xcode_spec('org.example.probe', '26.0')['targets']['NetalSRProbe']
        self.assertNotIn('NETAL_IOS_FP8_LAB=1', normal['settings']['base']['GCC_PREPROCESSOR_DEFINITIONS'])
        self.assertNotIn('NetalFP8LabIdentity', builder.info_plist('org.example.probe', '26.0'))
        with self.assertRaises(ValueError):
            builder.xcode_spec('org.example.probe', '26.0', fp8_lab=True)
        with self.assertRaises(ValueError):
            builder.info_plist('org.example.probe', '26.0', {'enabled': True})
        with self.assertRaises(ValueError):
            builder.load_fp8_lab(Path('/nonexistent-lab'), '26.0')
        lab = builder.xcode_spec('org.example.probe', '27.0', fp8_lab=True)['targets']['NetalSRProbe']
        self.assertIn('NETAL_IOS_FP8_LAB=1', lab['settings']['base']['GCC_PREPROCESSOR_DEFINITIONS'])
        self.assertIn({'path': 'Resources', 'buildPhase': 'resources'}, lab['sources'])
        self.assertIn({'framework': 'FP8Lab/lib/libnetal_ios_fp8_lab.a', 'embed': False}, lab['dependencies'])

    def test_optional_lab_rejects_mismatched_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ('libnetal_ios_fp8_lab.a', 'NetalFP8MatmulLab.h', 'netal_fp8_lab.metallib'):
                (root / name).write_text('synthetic build identity fixture ' + name)
            sha = lambda name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            receipt = {'schema': 'netal.fp8.matmul-lab.build.v1', 'sdk': 'iphoneos',
                       'deployment_target': '27.0', 'models_included': False,
                       'archive': 'libnetal_ios_fp8_lab.a', 'archive_sha256': sha('libnetal_ios_fp8_lab.a'),
                       'library': 'netal_fp8_lab.metallib', 'library_sha256': sha('netal_fp8_lab.metallib'),
                       'sources': [{'path': 'src/ios_fp8_lab/NetalFP8MatmulLab.h', 'sha256': sha('NetalFP8MatmulLab.h')}]}
            (root / 'build-receipt.json').write_text(json.dumps(receipt))
            found, identity = builder.load_fp8_lab(root, '27.0')
            self.assertEqual(found, root)
            self.assertEqual(identity['metallib_sha256'], receipt['library_sha256'])
            self.assertFalse(identity['models_included'])
            (root / 'netal_fp8_lab.metallib').write_text('altered fixture')
            with self.assertRaises(ValueError):
                builder.load_fp8_lab(root, '27.0')

    def test_metalfx_linkage_and_capture_are_explicit(self):
        normal = builder.info_plist('org.example.probe', '26.0')
        self.assertIs(normal['MetalCaptureEnabled'], False)
        with self.assertRaises(ValueError):
            builder.info_plist('org.example.probe', '26.0', enable_metal_capture=True)
        capture = builder.info_plist('org.example.probe', '26.0', metalfx_identity={'fixture': True}, enable_metal_capture=True)
        self.assertIs(capture['MetalCaptureEnabled'], True)
        target = builder.xcode_spec('org.example.probe', '26.0', metalfx_lab=True)['targets']['NetalSRProbe']
        self.assertIn({'sdk': 'MetalFX.framework'}, target['dependencies'])
        definitions = target['settings']['base']['GCC_PREPROCESSOR_DEFINITIONS']
        self.assertIn('NETAL_IOS_METALFX_LAB=1', definitions)
        self.assertNotIn('NETAL_IOS_FP8_LAB=1', definitions)
        combined = builder.xcode_spec('org.example.probe', '27.0', fp8_lab=True, metalfx_lab=True)['targets']['NetalSRProbe']
        self.assertIn('$(SRCROOT)/FP8Lab/include', combined['settings']['base']['HEADER_SEARCH_PATHS'])
        self.assertIn('$(SRCROOT)/MetalFXLab/include', combined['settings']['base']['HEADER_SEARCH_PATHS'])

    def test_metalfx_rejects_stimulus_drift_or_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'libnetal_ios_metalfx_lab.a').write_text('synthetic archive identity fixture')
            (root / 'NetalMetalFXLab.h').write_text('synthetic header identity fixture')
            sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
            receipt = {'schema': 'netal.metalfx.lab.build.v1', 'sdk': 'iphoneos',
                       'deployment_target': '26.0', 'models_included': False,
                       'archive': 'libnetal_ios_metalfx_lab.a', 'archive_sha256': sha(root / 'libnetal_ios_metalfx_lab.a'),
                       'sources': [{'path': 'src/ios_metalfx_lab/NetalMetalFXLab.h', 'sha256': sha(root / 'NetalMetalFXLab.h')},
                                   {'path': 'src/ios_probe/NetalProbeMetrics.h', 'sha256': sha(ROOT / 'src/ios_probe/NetalProbeMetrics.h')}]}
            path = root / 'build-receipt.json'
            path.write_text(json.dumps(receipt))
            _, identity = builder.load_metalfx_lab(root, '26.0')
            self.assertFalse(identity['models_included'])
            receipt['sources'][1]['sha256'] = '0' * 64
            path.write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):
                builder.load_metalfx_lab(root, '26.0')
            receipt['sources'][1]['sha256'] = sha(ROOT / 'src/ios_probe/NetalProbeMetrics.h')
            path.write_text(json.dumps(receipt))
            (root / 'libnetal_ios_metalfx_lab.a').write_text('altered fixture')
            with self.assertRaises(ValueError):
                builder.load_metalfx_lab(root, '26.0')

    def test_caller_team_only(self):
        settings = builder.xcode_spec('org.example.probe', '26.0', 'ABCDEFGHIJ')['targets']['NetalSRProbe']['settings']['base']
        self.assertEqual(settings['DEVELOPMENT_TEAM'], 'ABCDEFGHIJ')
        self.assertEqual(settings['CODE_SIGN_STYLE'], 'Automatic')

    def test_invalid_inputs_fail_before_compiler(self):
        with tempfile.TemporaryDirectory() as temp:
            library = Path(temp) / 'library'
            (library / 'lib').mkdir(parents=True)
            (library / 'include/NetalIOSSR').mkdir(parents=True)
            (library / 'lib/libNetalIOSSR.a').touch()
            (library / 'include/NetalIOSSR/NetalIOSSR.h').touch()
            output = Path(temp) / 'new'
            for sdk, target, bundle, team in [
                ('iphonesimulator', '26.0', 'org.example.probe', None),
                ('iphoneos', '25.0', 'org.example.probe', None),
                ('iphoneos', '27.0;false', 'org.example.probe', None),
                ('iphoneos', '26.0', '../escape', None),
                ('iphoneos', '26.0', 'org.example.probe', 'bad')]:
                with self.assertRaises(ValueError):
                    builder.validate(library, output, sdk, target, bundle, team)
            output.mkdir()
            with self.assertRaises(ValueError):
                builder.validate(library, output, 'iphoneos', '26.0', 'org.example.probe')

    def test_driver_timing_and_quantile_contract(self):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        source = r'''
#include "NetalProbeMetrics.h"
#include <cassert>
#include <limits>
int main() {
  using namespace netal_probe;
  assert(std::abs(GPUMilliseconds(1.0, 1.003, true, false)-3.0)<1e-8);
  assert(std::isnan(GPUMilliseconds(0.0, 0.0, true, false)));
  assert(std::isnan(GPUMilliseconds(1.0, 1.0, true, false)));
  assert(std::isnan(GPUMilliseconds(2.0, 1.0, true, false)));
  assert(std::isnan(GPUMilliseconds(1.0, 2.0, false, false)));
  assert(std::isnan(GPUMilliseconds(1.0, 2.0, true, true)));
  assert(std::isnan(GPUMilliseconds(1.0, std::numeric_limits<double>::infinity(), true, false)));
  assert(std::isnan(Percentile({}, 0.5)));
  assert(std::isnan(Percentile({1.0, std::numeric_limits<double>::quiet_NaN()}, 0.5)));
  assert(std::isnan(Percentile({1.0,-1.0}, 0.5)));
  assert(Percentile({4,1,3,2}, 0.5)==2);
  assert(Percentile({4,1,3,2}, 0.95)==4);
  assert(Percentile({4,1,3,2}, 0.99)==4);
  assert(StopBeforeNextFrame(false,true,false,10,100)==StopReason::None);
  assert(StopBeforeNextFrame(true,true,false,10,100)==StopReason::Cancelled);
  assert(StopBeforeNextFrame(false,false,false,10,100)==StopReason::Inactive);
  assert(StopBeforeNextFrame(false,true,true,10,100)==StopReason::CriticalThermal);
  assert(StopBeforeNextFrame(false,true,false,100,100)==StopReason::Deadline);
  assert(StopBeforeNextFrame(false,true,false,std::numeric_limits<double>::quiet_NaN(),100)==StopReason::Deadline);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'metrics.cpp'
            path.write_text(source)
            binary = Path(temp) / 'metrics'
            subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'src/ios_probe'), str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_output_persistence_requires_json_boolean(self):
        compiler = shutil.which('clang++')
        if not compiler or not Path('/System/Library/Frameworks/Foundation.framework').exists():
            self.skipTest('Apple Foundation compiler unavailable')
        source = r'''
#include "NetalProbeMetrics.h"
#include <cassert>
int main() { @autoreleasepool {
  bool result=true;
  assert(netal_probe::ReadOptionalBoolean(nil,false,&result) && !result);
  for (NSString* text in @[@"true",@"false"]) {
    NSData* data=[text dataUsingEncoding:NSUTF8StringEncoding];
    id value=[NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:nil];
    assert(netal_probe::ReadOptionalBoolean(value,false,&result));
    assert(result==[text isEqual:@"true"]);
  }
  for (NSString* text in @[@"0",@"1",@"null",@"\"true\"",@"[]",@"{}"]) {
    NSData* data=[text dataUsingEncoding:NSUTF8StringEncoding];
    id value=[NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingFragmentsAllowed error:nil];
    assert(!netal_probe::ReadOptionalBoolean(value,false,&result));
  }
} }
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'boolean.mm'
            path.write_text(source)
            binary = Path(temp) / 'boolean'
            subprocess.run([compiler, '-std=c++20', '-fobjc-arc', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'src/ios_probe'), str(path), '-framework', 'Foundation',
                            '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)

    def test_render_geometry_and_normalized_stimulus(self):
        compiler = shutil.which('clang++') or shutil.which('c++')
        if not compiler:
            self.skipTest('C++ compiler unavailable')
        source = r'''
#include "NetalProbeMetrics.h"
#include <cassert>
#include <string_view>
int main() {
  using namespace netal_probe;
  Dimensions parsed;
  assert(ParseSize("640x360", &parsed) && parsed.width==640 && parsed.height==360);
  assert(ParseSize("960x540", &parsed) && parsed.width==960 && parsed.height==540);
  for (auto text : {"", "0x360", "640x0", "0640x360", "640X360", "640x360\n",
       " 640x360", "640x360x1", "640x360junk", "-640x360", "4294967296x360",
       "18446744073709551616x360"}) assert(!ParseSize(text, &parsed));
  const char embeddedNull[]="640x360\0ignored";
  assert(!ParseSize(std::string_view(embeddedNull,sizeof(embeddedNull)-1), &parsed));
  assert(RenderFitsOutput({640,360},{1280,720}));
  assert(RenderFitsOutput({960,540},{1920,1080}));
  assert(RenderFitsOutput({640,360},{1920,1080}));
  assert(RenderFitsOutput({854,480},{1280,720})); // ordinary integer rounding, within 1%.
  assert(RenderFitsOutput({640,363},{1280,720}));
  assert(!RenderFitsOutput({640,364},{1280,720}));
  assert(!RenderFitsOutput({1281,720},{1280,720}));
  assert(!RenderFitsOutput({640,721},{1280,720}));
  assert(!RenderFitsOutput({640,400},{1280,720}));
  assert(!RenderFitsOutput({0,360},{1280,720}));
  assert(!RenderFitsOutput({640,360},{0,720}));
  const Dimensions a{640,360},b{960,540};
  constexpr double x=37,y=19,jx=0.1875,jy=-0.0185185075;
  const auto p=PlaneCoordinates(x,y,jx,jy,a,17);
  const double bx=(x+0.5+jx)*b.width/a.width-0.5-jx;
  const double by=(y+0.5+jy)*b.height/a.height-0.5-jy;
  const auto q=PlaneCoordinates(bx,by,jx,jy,b,17);
  assert(std::abs(p.u-q.u)<1e-12 && std::abs(p.v-q.v)<1e-12);
  const auto next=PlaneCoordinates(x,y,jx,jy,a,18);
  assert(std::abs((next.u-p.u)+kPlaneVelocityU)<1e-12);
  assert(next.v==p.v);
  assert(std::abs(-kPlaneVelocityU*a.width-(-0.5))<1e-12);
  assert(std::abs(-kPlaneVelocityU*b.width-(-0.75))<1e-12);
  assert(kPlaneCyclesU==12 && kPlaneCyclesV==7);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'geometry.cpp'
            path.write_text(source)
            binary = Path(temp) / 'geometry'
            subprocess.run([compiler, '-std=c++20', '-Wall', '-Wextra', '-Werror',
                            '-I', str(ROOT / 'src/ios_probe'), str(path), '-o', str(binary)], check=True)
            subprocess.run([str(binary)], check=True)


if __name__ == '__main__':
    unittest.main()
