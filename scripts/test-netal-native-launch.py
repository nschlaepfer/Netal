#!/usr/bin/env python3
"""Native eligibility and launcher contracts; fixtures never execute a game."""
import importlib.util
import json
import os
from pathlib import Path
import plistlib
import shutil
import struct
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("native_inspect", ROOT / "scripts/netal-native-inspect.py")
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)


def macho(metalfx=False):
    text = b"/System/Library/Frameworks/MetalFX.framework/Versions/A/MetalFX\0" if metalfx else b"/usr/lib/libSystem.B.dylib\0"
    length = (24 + len(text) + 7) & ~7
    command = struct.pack("<IIIIII", 0xc, length, 24, 0, 0, 0) + text
    command = command.ljust(length, b"\0")
    return struct.pack("<IIIIIIII", 0xfeedfacf, 0x100000c, 0, 2, 1, len(command), 0, 0) + command


class InspectionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="netal-native-test-")
        self.root = Path(self.temp.name)
        self.app = self.root / "Fixture game.app"
        self.binary = self.app / "Contents/MacOS/Fixture"
        self.binary.parent.mkdir(parents=True)
        self.binary.write_bytes(macho(True))
        self.binary.chmod(0o755)
        self.plist = {"CFBundleIdentifier": "com.example.fixture", "CFBundleExecutable": "Fixture"}
        self.write_plist()
        self.entitlements = {"com.apple.security.cs.allow-dyld-environment-variables": True,
                             "com.apple.security.cs.disable-library-validation": True}
        self.arch = "arm64 x86_64"

    def tearDown(self):
        self.temp.cleanup()

    def write_plist(self):
        (self.app / "Contents/Info.plist").write_bytes(plistlib.dumps(self.plist))

    def command(self, arguments):
        if arguments[0].endswith("lipo"):
            return subprocess.CompletedProcess(arguments, 0, self.arch.encode(), b"")
        if "--entitlements" in arguments:
            return subprocess.CompletedProcess(arguments, 0, plistlib.dumps(self.entitlements), b"")
        return subprocess.CompletedProcess(arguments, 0, b"", b"CodeDirectory flags=0x10000(runtime)")

    def inspect(self):
        with patch.object(native, "run", self.command):
            return native.inspect(self.app)

    def test_arm64_candidate_is_not_gameplay_proof(self):
        result = self.inspect()
        self.assertTrue(result["canLaunchModels"])
        self.assertIn("unverified", result["summary"])

    def test_unknown_seam_allows_only_observation(self):
        self.binary.write_bytes(macho(False))
        result = self.inspect()
        self.assertTrue(result["canObserve"])
        self.assertFalse(result["canLaunchModels"])
        self.assertEqual(result["summary"], "Use Window Capture")

    def test_framework_link_detected(self):
        self.binary.write_bytes(macho(False))
        engine = self.app / "Contents/Frameworks/UnityPlayer.dylib"
        engine.parent.mkdir()
        engine.write_bytes(macho(True))
        result = self.inspect()
        self.assertTrue(result["canLaunchModels"])
        self.assertEqual(result["evidencePaths"], ["Contents/Frameworks/UnityPlayer.dylib"])

    def test_false_entitlement_does_not_authorize_injection(self):
        self.entitlements["com.apple.security.cs.allow-dyld-environment-variables"] = False
        self.assertFalse(self.inspect()["canObserve"])

    def test_library_validation_blocks_injection(self):
        self.entitlements["com.apple.security.cs.disable-library-validation"] = False
        self.assertFalse(self.inspect()["canObserve"])

    def test_intel_only_is_explicit(self):
        self.arch = "x86_64"
        result = self.inspect()
        self.assertFalse(result["canObserve"])
        self.assertIn("Intel", result["summary"])

    def test_identifier_cannot_escape_session_root(self):
        self.plist["CFBundleIdentifier"] = "../../outside"
        self.write_plist()
        self.assertFalse(self.inspect()["canObserve"])

    def test_symlinked_external_libraries_not_scanned(self):
        self.binary.write_bytes(macho(False))
        engine = self.root / "external.dylib"
        engine.write_bytes(macho(True))
        frameworks = self.app / "Contents/Frameworks"
        frameworks.mkdir()
        (frameworks / "engine.dylib").symlink_to(engine)
        self.assertFalse(self.inspect()["hasMetalFXEvidence"])

    def test_fat_second_slice_beyond_first_four_megabytes(self):
        first, second = macho(False), macho(True)
        offset = 5 << 20
        header = struct.pack(">II", 0xcafebabe, 2)
        header += struct.pack(">IIIII", 0x1000007, 3, 4096, len(first), 12)
        header += struct.pack(">IIIII", 0x100000c, 0, offset, len(second), 12)
        with self.binary.open("wb") as output:
            output.write(header)
            output.seek(4096); output.write(first)
            output.seek(offset); output.write(second)
        self.assertTrue(native.metalfx_evidence(self.binary))

    def test_malformed_fat_binary_is_bounded(self):
        self.binary.write_bytes(struct.pack(">II", 0xcafebabe, 0xffffffff))
        self.assertFalse(native.metalfx_evidence(self.binary))


class LauncherContractTests(InspectionTests):
    def setUp(self):
        super().setUp()
        self.repo = self.root / "runtime"
        (self.repo / "scripts").mkdir(parents=True)
        for name in ("netal-native.sh", "netal-native-preflight.sh", "netal-native-inspect.py"):
            shutil.copy2(ROOT / "scripts" / name, self.repo / "scripts" / name)
        self.bridge = self.repo / "build/netal-native/libNetalNative.dylib"
        self.bridge.parent.mkdir(parents=True)
        self.bridge.write_bytes(macho())
        for feature in ("sr", "fg"):
            folder = self.repo / f"work/re9-{feature}-runtime-bundle"
            folder.mkdir(parents=True)
            (folder / "payload.bin").write_bytes(b"test")
            (folder / "manifest.json").write_text(json.dumps({"files": [{"path": "payload.bin", "bytes": 4}]}))

    def run_check(self, *arguments):
        env = dict(os.environ, HOME=str(self.root / "fake-home"))
        # --check exits before mkdir, defaults, pipeline generation, open, or any game launch.
        return subprocess.run(["/bin/zsh", str(self.repo / "scripts/netal-native.sh"), str(self.app),
                               "--check", *arguments], capture_output=True, text=True, env=env, timeout=30)

    def test_unsandboxed_check_needs_no_container_and_writes_nothing(self):
        result = self.run_check()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("result=pass", result.stdout)
        self.assertFalse((self.root / "fake-home").exists())

    def test_sr_only_does_not_require_fg_bundle(self):
        shutil.rmtree(self.repo / "work/re9-fg-runtime-bundle")
        result = self.run_check("--no-fg")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unsandboxed_no_stage_uses_prepared_runtime(self):
        result = self.run_check("--no-stage")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_fg_only_does_not_require_sr_bundle(self):
        shutil.rmtree(self.repo / "work/re9-sr-runtime-bundle")
        result = self.run_check("--no-sr")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_requested_missing_runtime_fails_before_mutation(self):
        shutil.rmtree(self.repo / "work/re9-fg-runtime-bundle")
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Frame Generation runtime is not ready", result.stderr)
        self.assertFalse((self.root / "fake-home").exists())

    def test_observe_does_not_require_any_model(self):
        shutil.rmtree(self.repo / "work")
        self.binary.write_bytes(macho(False))
        result = self.run_check("--observe")
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_unknown_seam_cannot_arm_models(self):
        self.binary.write_bytes(macho(False))
        result = self.run_check()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Window Capture", result.stderr)

    def test_empty_choices_fail_before_mutation(self):
        result = self.run_check("--no-sr", "--no-fg")
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((self.root / "fake-home").exists())

    def test_missing_option_value_reports_usage(self):
        result = self.run_check("--output")
        self.assertEqual(result.returncode, 64)
        self.assertIn("Missing value", result.stderr)


if __name__ == "__main__":
    unittest.main()
