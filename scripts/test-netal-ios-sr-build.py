#!/usr/bin/env python3
"""Model-free target/ABI regression checks for the iOS build entry point."""
import importlib.util
import json
import struct
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from netal_apple_build import check_macho_platform, metal_target, version_tuple

spec = importlib.util.spec_from_file_location('ios_build', Path(__file__).with_name('build-netal-ios-sr.py'))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assets_spec = importlib.util.spec_from_file_location('sr_assets', Path(__file__).with_name('build-netal-sr-assets.py'))
assets = importlib.util.module_from_spec(assets_spec)
assets_spec.loader.exec_module(assets)


def macho(platform, minimum=(26 << 16), length=24, cpu=0x100000c, kind=1):
    header = struct.pack('<IIIIIIII', 0xfeedfacf, cpu, 0, kind, 1, length, 0, 0)
    return header + struct.pack('<IIIIII', 0x32, length, platform, minimum, 0, 0)


class BuildTargetTests(unittest.TestCase):
    def test_existing_mac_default_not_retargeted(self):
        self.assertEqual(metal_target('macosx'), (None, None))

    def test_device_and_simulator_are_distinct(self):
        self.assertEqual(metal_target('iphoneos'), ('26.0', 'air64-apple-ios26.0'))
        self.assertEqual(metal_target('iphonesimulator', '27.0'), ('27.0', 'air64-apple-ios27.0-simulator'))

    def test_unsupported_or_malformed_targets_denied(self):
        for sdk, value in [('iphoneos', '25.0'), ('iphoneos', '26'), ('iphoneos', 'nan'),
                           ('iphoneos', '26.0 -c other'), ('iphoneos', '300.0'), ('windows', '26.0')]:
            with self.assertRaises(ValueError):
                metal_target(sdk, value)

    def test_version_order(self):
        self.assertLess(version_tuple('26.9'), version_tuple('27.0'))
        self.assertEqual(version_tuple('26.0'), version_tuple('26.0.0'))

    def test_archive_object_admission(self):
        with tempfile.TemporaryDirectory() as t:
            obj = Path(t) / 'test.o'
            obj.write_bytes(macho(2))
            check_macho_platform(obj, 2, '26.0')
            for raw in [macho(1), macho(7), macho(2, 27 << 16), macho(2, cpu=0x1000007), macho(2, kind=6),
                        macho(2)[:38], b'not an object']:
                obj.write_bytes(raw)
                with self.assertRaises(ValueError):
                    check_macho_platform(obj, 2, '26.0')

    def test_simulator_object_cannot_be_used_as_device(self):
        with tempfile.TemporaryDirectory() as t:
            obj = Path(t) / 'test.o'
            obj.write_bytes(macho(7))
            check_macho_platform(obj, 7, '26.0')
            with self.assertRaises(ValueError):
                check_macho_platform(obj, 2, '26.0')

    def test_existing_output_preserved_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as t:
            out = Path(t) / 'output'
            out.mkdir()
            (out / 'keep').write_text('keep')
            with patch.object(module.subprocess, 'check_output', side_effect=AssertionError('tool called')):
                with self.assertRaises(ValueError):
                    module.build(out)
            self.assertEqual((out / 'keep').read_text(), 'keep')

    def test_simulator_engine_denied_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(module.subprocess, 'check_output', side_effect=AssertionError('tool called')):
                with self.assertRaisesRegex(ValueError, 'physical iphoneos'):
                    module.build(Path(t) / 'new', sdk='iphonesimulator')

    def test_old_deployment_denied_before_tool_execution(self):
        with tempfile.TemporaryDirectory() as t:
            with patch.object(module.subprocess, 'check_output', side_effect=AssertionError('tool called')):
                with self.assertRaises(ValueError):
                    module.build(Path(t) / 'new', deployment_target='25.0')

    def check_asset_target_routing(self, sdk, minimum, triple):
        """Tiny procedural inputs exercise every library-builder command, without a model."""
        with tempfile.TemporaryDirectory() as t, ExitStack() as stack:
            root = Path(t)
            model = root / 'model'
            model.mkdir()
            data = b'synthetic routing fixture'
            (model / 'weights.bin').write_bytes(data)
            abi = SimpleNamespace(validate_package=lambda _: ({}, {}),
                                  for_geometry=lambda *_: {}, profile_manifest=lambda *_: {})
            extractor = SimpleNamespace(KNOWN={'weights.bin': len(data)},
                                        REFERENCE={'files': {'weights.bin': assets.digest(data)}},
                                        abi_module=lambda: abi)
            stack.enter_context(patch.object(assets, 'load', side_effect=[extractor, SimpleNamespace()]))
            stack.enter_context(patch.object(assets, 'STAGES', []))
            stack.enter_context(patch.object(assets, 'profile', return_value=({'pwin/enc0/arguments-template.bin': b'fixed'}, {})))
            stack.enter_context(patch.object(assets, 'generate_math', return_value={}))
            stack.enter_context(patch.object(assets, 'load_calibration', return_value={}))
            flags = [] if triple is None else ['-target', triple, '-isysroot', '/synthetic-sdk']
            info = {'sdk': sdk, 'deployment_target': minimum, 'metal_target': triple}
            toolchain = stack.enter_context(patch.object(assets, 'metal_toolchain',
                                                       return_value=(Path('/tools/metal'), Path('/tools/metallib'), flags, info)))
            stack.enter_context(patch.object(assets.subprocess, 'check_output', return_value='fixture compiler'))
            calls = []

            def run(command, **_):
                calls.append(command)
                if len(command) > 1 and command[1].endswith('-kernels.py'):
                    output = Path(command[command.index('--output') + 1])
                    output.mkdir()
                    name = Path(command[1]).stem + '.metallib'
                    (output / name).write_bytes(b'synthetic library')
                    kernel = {'source': 'tools/pwin_pipeline/pwin_gemm.metal', 'library': name}
                    (output / 'build-receipt.json').write_text(json.dumps({'kernels': [kernel]}))
                elif '-o' in command:
                    Path(command[command.index('-o') + 1]).write_bytes(b'synthetic compiler output')
                return SimpleNamespace(stdout='')

            stack.enter_context(patch.object(assets.subprocess, 'run', side_effect=run))
            receipt = assets.build(model, root / 'output', ['1920x1080'], sdk=sdk, deployment_target=minimum)
            toolchain.assert_called_once_with(sdk, minimum)
            compiler = next(c for c in calls if c[0] == '/tools/metal')
            self.assertEqual(compiler[1:1 + len(flags)], flags)
            if triple is None:
                self.assertNotIn('-target', compiler)
            children = [c for c in calls if len(c) > 1 and c[1].endswith('-kernels.py')]
            self.assertEqual(len(children), 3)
            for command in children:
                self.assertEqual(command[command.index('--sdk') + 1], sdk)
                if minimum is None:
                    self.assertNotIn('--deployment-target', command)
                else:
                    self.assertEqual(command[command.index('--deployment-target') + 1], minimum)
            self.assertEqual(receipt['metal_target'], triple)
            self.assertFalse(receipt['inference_validated'])

    def test_all_asset_libraries_receive_iphone_target(self):
        self.check_asset_target_routing('iphoneos', '26.0', 'air64-apple-ios26.0')

    def test_asset_mac_defaults_preserved(self):
        self.check_asset_target_routing('macosx', None, None)

    def test_asset_bad_sdk_denied_before_model_read(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaisesRegex(ValueError, 'unsupported Apple SDK'):
                assets.build(Path(t) / 'missing-model', Path(t) / 'output', [], sdk='windows')


if __name__ == '__main__':
    unittest.main()
