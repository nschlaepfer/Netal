#!/usr/bin/env python3
"""Exercise unavailable timing, percentiles and unsigned/physical-only build contracts."""
import importlib.util
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


if __name__ == '__main__':
    unittest.main()
