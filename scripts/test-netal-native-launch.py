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
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True

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
        self.assertEqual(result["executable"], "Fixture")
        self.assertEqual(result["targetExecutables"], ["Fixture"])

    def test_two_binary_helper_executable_detected(self):
        macos = self.app / "Contents/MacOS"
        launcher = macos / "Launcher"
        render_host = macos / "RenderHost"
        crashpad = macos / "crashpad_handler"
        support_dylib = macos / "libSupport.dylib"
        self.binary.unlink()
        launcher.write_bytes(macho(False))
        launcher.chmod(0o755)
        render_host.write_bytes(macho(True))
        render_host.chmod(0o755)
        crashpad.write_bytes(macho(False))
        crashpad.chmod(0o755)
        support_dylib.write_bytes(macho(True))
        support_dylib.chmod(0o755)
        self.plist["CFBundleExecutable"] = "Launcher"
        self.write_plist()
        result = self.inspect()
        self.assertTrue(result["canLaunchModels"])
        self.assertEqual(result["executable"], "Launcher")
        self.assertEqual(result["targetExecutables"], ["Launcher", "RenderHost"])
        self.assertIn("Contents/MacOS/RenderHost", result["evidencePaths"])
        self.assertNotIn("crashpad_handler", result["targetExecutables"])
        self.assertNotIn("libSupport.dylib", result["targetExecutables"])

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

    def test_two_binary_bundle_check_and_target_routing(self):
        macos = self.app / "Contents/MacOS"
        launcher = macos / "Launcher"
        render_host = macos / "RenderHost"
        crashpad = macos / "crashpad_handler"
        self.binary.unlink()
        launcher.write_bytes(macho(False))
        launcher.chmod(0o755)
        render_host.write_bytes(macho(True))
        render_host.chmod(0o755)
        crashpad.write_bytes(macho(False))
        crashpad.chmod(0o755)
        self.plist["CFBundleExecutable"] = "Launcher"
        self.write_plist()

        check_result = self.run_check()
        self.assertEqual(check_result.returncode, 0, check_result.stdout + check_result.stderr)
        self.assertIn("result=pass", check_result.stdout)
        self.assertFalse((self.root / "fake-home").exists())

        # Verify the launcher script checks both binaries for running state and forwards
        # comma-separated NEURALSTUDIO_TARGET_EXE without executing any real game or Launch Services.
        mock_bin = self.root / "mock-bin"
        mock_bin.mkdir()
        ps_file = self.root / "mock-ps.txt"
        open_log = self.root / "mock-open.json"
        mock_ps = mock_bin / "ps"
        mock_ps.write_text("#!/bin/sh\ncat \"$NETAL_MOCK_PS_FILE\"\n")
        mock_ps.chmod(0o755)
        mock_open = mock_bin / "open"
        mock_open.write_text(
            "#!/usr/bin/env python3\nimport json, os, sys\n"
            "open(os.environ['NETAL_MOCK_OPEN_LOG'], 'w').write(json.dumps(sys.argv[1:]))\n"
        )
        mock_open.chmod(0o755)
        mock_defaults = mock_bin / "defaults"
        mock_defaults.write_text("#!/bin/sh\nexit 0\n")
        mock_defaults.chmod(0o755)

        patched_script = (self.repo / "scripts/netal-native.sh").read_text()
        patched_script = patched_script.replace("/bin/ps ", f'"{mock_ps}" ')
        patched_script = patched_script.replace("/usr/bin/open ", f'"{mock_open}" ')
        patched_script = patched_script.replace("defaults write ", f'"{mock_defaults}" write ')
        test_launcher = self.repo / "scripts/netal-native-mocked.sh"
        test_launcher.write_text(patched_script)
        test_launcher.chmod(0o755)

        base_env = dict(os.environ, HOME=str(self.root / "launch-home"),
                        NETAL_MOCK_PS_FILE=str(ps_file), NETAL_MOCK_OPEN_LOG=str(open_log))

        # If RenderHost is still running (even after Launcher exited), launch must refuse cleanly.
        ps_file.write_text(f"/sbin/launchd\n{render_host.resolve()}\n")
        running = subprocess.run(["/bin/zsh", str(test_launcher), str(self.app)],
                                 capture_output=True, text=True, env=base_env, timeout=30)
        self.assertNotEqual(running.returncode, 0)
        self.assertIn("still running", running.stderr)
        self.assertFalse(open_log.exists())

        # Once neither binary is running, launch forwards NEURALSTUDIO_TARGET_EXE=Launcher,RenderHost.
        ps_file.write_text(f"/sbin/launchd\n{crashpad.resolve()}\n")
        launched = subprocess.run(["/bin/zsh", str(test_launcher), str(self.app), "--no-sr"],
                                  capture_output=True, text=True, env=base_env, timeout=30)
        self.assertEqual(launched.returncode, 0, launched.stdout + launched.stderr)
        open_args = json.loads(open_log.read_text())
        self.assertIn("NEURALSTUDIO_TARGET_EXE=Launcher,RenderHost", open_args)

    def test_bridge_is_target_process_comma_separated_exact_basename(self):
        bridge_src = (ROOT / "src/macos_host/re9_metalfx_noop_bridge.mm").read_text()
        start = bridge_src.index("bool IsTargetProcess() {")
        end = bridge_src.index("\nvoid LogEncode(", start)
        fn_src = bridge_src[start:end]
        probe_mm = self.root / "target_probe.mm"
        probe_mm.write_text(
            "#import <Foundation/Foundation.h>\n#include <cstdlib>\n"
            + fn_src
            + "\nint main() { @autoreleasepool { return IsTargetProcess() ? 0 : 1; } }\n"
        )
        probe_bin = self.root / "target_probe"
        subprocess.run(["xcrun", "--sdk", "macosx", "clang++", "-std=c++20", "-fobjc-arc",
                        "-Wall", "-Wextra", "-Werror", "-framework", "Foundation",
                        str(probe_mm), "-o", str(probe_bin)], check=True)

        def check_proc(exe_name, target_env, extra_args=()):
            exe_copy = self.root / exe_name
            shutil.copy2(probe_bin, exe_copy)
            env = dict(os.environ)
            if target_env is None:
                env.pop("NEURALSTUDIO_TARGET_EXE", None)
            else:
                env["NEURALSTUDIO_TARGET_EXE"] = target_env
            res = subprocess.run([str(exe_copy), *extra_args], env=env, timeout=10)
            return res.returncode == 0

        self.assertTrue(check_proc("nxnative", "nxnative,dmhost"))
        self.assertTrue(check_proc("dmhost", "nxnative,dmhost"))
        self.assertTrue(check_proc("dmhost", " nxnative , dmhost "))
        self.assertFalse(check_proc("crashpad_handler", "nxnative,dmhost"))
        self.assertFalse(check_proc("Finder", "nxnative,dmhost"))
        self.assertFalse(check_proc("dmhost_helper", "nxnative,dmhost"))
        self.assertFalse(check_proc("mydmhost", "nxnative,dmhost"))
        self.assertFalse(check_proc("crashpad_handler", "nxnative,dmhost",
                                    ["--client=/Applications/Game.app/Contents/MacOS/dmhost"]))
        self.assertTrue(check_proc("wine64-preloader", None, ["C:\\Games\\RE9\\re9.exe"]))

    def test_nr_experimental_opt_in_check_and_launch_env(self):
        # Missing --nr-bridge-dir / --nr-model must fail cleanly before mutation.
        missing_args = self.run_check("--no-sr", "--no-fg", "--nr-experimental")
        self.assertEqual(missing_args.returncode, 64)
        self.assertFalse((self.root / "fake-home").exists())

        # Create synthetic NR bridge artifacts and model file; also remove MetalFX evidence to verify
        # NR-only (--no-sr --no-fg --nr-experimental) works on drawable-only native Metal 4 bundles!
        self.binary.write_bytes(macho(False))
        nr_dir = self.root / "nr-bridge"
        nr_dir.mkdir()
        (nr_dir / "libNetalNRFastBridge.dylib").write_bytes(macho(False))
        (nr_dir / "nr_fast_kernels.metallib").write_bytes(b"MTLB")
        (nr_dir / "netal_nr_optical_flow.metallib").write_bytes(b"MTLB")
        nr_model = self.root / "fixture_nr.safetensors"
        nr_model.write_bytes(b"stub")

        chk = self.run_check(
            "--no-sr", "--no-fg", "--nr-experimental",
            "--nr-bridge-dir", str(nr_dir),
            "--nr-model", str(nr_model),
        )
        self.assertEqual(chk.returncode, 0, chk.stdout + chk.stderr)
        self.assertIn("result=pass", chk.stdout)

        mock_bin = self.root / "mock-nr-bin"
        mock_bin.mkdir()
        ps_file = self.root / "mock-nr-ps.txt"
        ps_file.write_text("/sbin/launchd\n")
        open_log = self.root / "mock-nr-open.json"
        for name, body in (
            ("ps", '#!/bin/sh\ncat "$NETAL_MOCK_PS_FILE"\n'),
            ("open", "#!/usr/bin/env python3\nimport json, os, sys\nopen(os.environ['NETAL_MOCK_OPEN_LOG'], 'w').write(json.dumps(sys.argv[1:]))\n"),
            ("defaults", "#!/bin/sh\nexit 0\n"),
        ):
            p = mock_bin / name
            p.write_text(body)
            p.chmod(0o755)

        patched_script = (self.repo / "scripts/netal-native.sh").read_text()
        patched_script = patched_script.replace("/bin/ps ", f'"{mock_bin / "ps"}" ')
        patched_script = patched_script.replace("/usr/bin/open ", f'"{mock_bin / "open"}" ')
        patched_script = patched_script.replace("defaults write ", f'"{mock_bin / "defaults"}" write ')
        test_launcher = self.repo / "scripts/netal-native-nr-mocked.sh"
        test_launcher.write_text(patched_script)
        test_launcher.chmod(0o755)

        env = dict(os.environ, HOME=str(self.root / "nr-home"),
                   NETAL_MOCK_PS_FILE=str(ps_file), NETAL_MOCK_OPEN_LOG=str(open_log))
        launched = subprocess.run(
            [
                "/bin/zsh", str(test_launcher), str(self.app),
                "--no-sr", "--no-fg", "--nr-experimental",
                "--nr-bridge-dir", str(nr_dir),
                "--nr-model", str(nr_model),
                "--nr-scale", "0.5",
            ],
            capture_output=True, text=True, env=env, timeout=30,
        )
        self.assertEqual(launched.returncode, 0, launched.stdout + launched.stderr)
        open_args = json.loads(open_log.read_text())
        self.assertIn(f"DYLD_INSERT_LIBRARIES={nr_dir / 'libNetalNRFastBridge.dylib'}", open_args)
        self.assertIn("NETAL_NR_ADAPTER_MODE=continuous", open_args)
        self.assertIn(f"NETAL_NR_MODEL={nr_model}", open_args)
        self.assertIn("NETAL_NR_HOOK_DRAWABLE=1", open_args)
        self.assertIn("NETAL_NR_ASYNC=1", open_args)
        self.assertIn("NETAL_NR_SCALE=0.5", open_args)


if __name__ == "__main__":
    unittest.main()
