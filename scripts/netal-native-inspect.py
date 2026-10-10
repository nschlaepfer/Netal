#!/usr/bin/env python3
"""Read-only native launch eligibility. Static MetalFX evidence is never gameplay proof."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import plistlib
import re
import struct
import subprocess
import sys

MAX_BINARIES = 96
MAX_HEADER = 1 << 20
MAX_STRING_SCAN = 4 << 20
MACH_MAGICS = {b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe", b"\xfe\xed\xfa\xcf", b"\xfe\xed\xfa\xce"}


def run(arguments: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(arguments, capture_output=True, timeout=15)


def macho_slices(path: Path) -> list[tuple[int, int]]:
    """Read each fat-binary slice independently; never assume it is in the first 4 MiB."""
    with path.open("rb") as handle:
        head = handle.read(8)
        if head[:4] in MACH_MAGICS:
            return [(0, path.stat().st_size)]
        formats = {b"\xca\xfe\xba\xbe": (">", False), b"\xbe\xba\xfe\xca": ("<", False),
                   b"\xca\xfe\xba\xbf": (">", True), b"\xbf\xba\xfe\xca": ("<", True)}
        if head[:4] not in formats or len(head) != 8:
            return []
        endian, wide = formats[head[:4]]
        count = struct.unpack(endian + "I", head[4:])[0]
        if count > 32:
            return []
        size = 32 if wide else 20
        rows = handle.read(count * size)
        if len(rows) != count * size:
            return []
        result = []
        length = path.stat().st_size
        for index in range(count):
            row = rows[index * size:(index + 1) * size]
            offset, extent = struct.unpack_from(endian + ("QQ" if wide else "II"), row, 8)
            if extent >= 28 and offset <= length and extent <= length - offset:
                result.append((offset, extent))
        return result


def metalfx_evidence(path: Path) -> bool:
    try:
        slices = macho_slices(path)
        with path.open("rb") as handle:
            for offset, size in slices:
                handle.seek(offset)
                header = handle.read(min(size, MAX_HEADER))
                if header[:4] not in MACH_MAGICS or len(header) < 28:
                    continue
                endian = "<" if header[:4] in (b"\xcf\xfa\xed\xfe", b"\xce\xfa\xed\xfe") else ">"
                ncmds, sizeofcmds = struct.unpack_from(endian + "II", header, 16)
                cursor = 32 if header[:4] in (b"\xcf\xfa\xed\xfe", b"\xfe\xed\xfa\xcf") else 28
                if ncmds > 65536 or sizeofcmds > MAX_HEADER - cursor:
                    continue
                for _ in range(ncmds):
                    if cursor + 8 > len(header):
                        break
                    command, length = struct.unpack_from(endian + "II", header, cursor)
                    if length < 8 or cursor + length > len(header):
                        break
                    if command & 0x7fffffff in (0xc, 0x18, 0x1f, 0x20, 0x23):
                        if b"metalfx.framework" in header[cursor:cursor + length].lower():
                            return True
                    cursor += length
                # dlopen users need not have a load command. This remains a candidate only.
                handle.seek(offset)
                strings = handle.read(min(size, MAX_STRING_SCAN))
                if b"MetalFX.framework" in strings or b"newTemporalScalerWithDevice:" in strings or b"newFrameInterpolatorWithDevice:" in strings:
                    return True
    except (OSError, ValueError, struct.error):
        pass
    return False


def bundled_binaries(app: Path, executable: Path) -> list[Path]:
    result = [executable]
    visited = 0
    # Only bundle code directories. Do not recursively read user data or follow symlinks.
    for base in (app / "Contents/Frameworks", app / "Contents/PlugIns", app / "Contents/MacOS"):
        if not base.is_dir() or base.is_symlink():
            continue
        for root, directories, files in os.walk(base, followlinks=False):
            directories[:] = sorted(d for d in directories if not (Path(root) / d).is_symlink()
                                    and len((Path(root) / d).relative_to(base).parts) <= 7)
            for name in sorted(files):
                visited += 1
                if visited > 4096:
                    return result
                path = Path(root) / name
                if path == executable or path.is_symlink():
                    continue
                try:
                    if macho_slices(path):
                        result.append(path)
                except OSError:
                    continue
                if len(result) >= MAX_BINARIES:
                    return result
    return result


def inspect(app: Path) -> dict:
    result = dict(app=str(app), bundleID="", executable="", architectures=[], sandboxed=False,
                  hardened=False, allowDyld=False, disableLibraryValidation=False,
                  hasMetalFXEvidence=False, evidencePaths=[], canObserve=False, canLaunchModels=False,
                  summary="Cannot inspect this game", detail="Choose a downloaded macOS .app bundle.")
    try:
        app = app.resolve(strict=True)
        if app.suffix.lower() != ".app":
            raise ValueError("Choose a macOS .app bundle.")
        with (app / "Contents/Info.plist").open("rb") as handle:
            plist = plistlib.load(handle)
        bundle_id = plist.get("CFBundleIdentifier", "")
        executable = plist.get("CFBundleExecutable", "")
        if not isinstance(bundle_id, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,254}", bundle_id):
            raise ValueError("The app has no valid bundle identifier.")
        if not isinstance(executable, str) or not executable or Path(executable).name != executable or any(ord(c) < 32 for c in executable):
            raise ValueError("The app has no valid executable name.")
        binary = app / "Contents/MacOS" / executable
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise ValueError("The game executable is missing or not executable. Finish downloading it first.")
        if not binary.resolve().is_relative_to(app):
            raise ValueError("The game executable points outside its app bundle.")
        result.update(app=str(app), bundleID=bundle_id, executable=executable)
        arch = run(["/usr/bin/lipo", "-archs", str(binary)])
        if arch.returncode:
            raise ValueError("Cannot inspect the executable architecture.")
        result["architectures"] = arch.stdout.decode(errors="replace").strip().split()
        signature = run(["/usr/bin/codesign", "-dvvv", str(binary)])
        signing_text = (signature.stdout + signature.stderr).decode(errors="replace")
        if signature.returncode and "code object is not signed at all" not in signing_text:
            raise ValueError("Cannot inspect the game's code signature.")
        flags = re.search(r"flags=0x([0-9a-fA-F]+)", signing_text)
        if signature.returncode == 0 and not flags:
            raise ValueError("The code signature did not report its runtime flags.")
        result["hardened"] = bool(flags and int(flags.group(1), 16) & 0x10000)
        entitlements = {}
        if signature.returncode == 0:
            signed = run(["/usr/bin/codesign", "-d", "--entitlements", "-", "--xml", str(binary)])
            if signed.returncode:
                raise ValueError("Cannot inspect the game's signing entitlements.")
            if signed.stdout.strip():
                entitlements = plistlib.loads(signed.stdout)
        result["sandboxed"] = entitlements.get("com.apple.security.app-sandbox") is True
        result["allowDyld"] = entitlements.get("com.apple.security.cs.allow-dyld-environment-variables") is True
        result["disableLibraryValidation"] = entitlements.get("com.apple.security.cs.disable-library-validation") is True
        binaries = bundled_binaries(app, binary)
        result["evidencePaths"] = [str(p.relative_to(app)) for p in binaries if metalfx_evidence(p)]
        result["hasMetalFXEvidence"] = bool(result["evidencePaths"])
        if "arm64" not in result["architectures"]:
            result.update(summary="Intel game: use Window Capture", detail="The native Netal bridge requires an Apple silicon game. Window Capture can work with Intel games running through Rosetta.")
        elif result["hardened"] and not result["allowDyld"]:
            result.update(summary="Native attachment blocked", detail="This game's code signature does not allow the Netal bridge. Use Window Capture; no game files or system security settings need changing.")
        elif result["hardened"] and not result["disableLibraryValidation"]:
            result.update(summary="Native attachment blocked", detail="This game only accepts libraries signed by its developer. Use Window Capture instead.")
        else:
            result["canObserve"] = True
            result["canLaunchModels"] = result["hasMetalFXEvidence"]
            if result["hasMetalFXEvidence"]:
                result.update(summary="Native candidate · gameplay unverified", detail="MetalFX code was found. Netal can try the game's existing SR and FG integrations; each feature is confirmed only after the game submits frames.")
            else:
                result.update(summary="Use Window Capture", detail="No MetalFX integration was found in the inspected game code. Native model acceleration is not enabled by this check; an observation run can check for dynamically loaded integrations.")
    except (OSError, ValueError, plistlib.InvalidFileException, subprocess.SubprocessError) as error:
        result["detail"] = str(error)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("app", type=Path)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--observe", action="store_true")
    args = parser.parse_args()
    result = inspect(args.app)
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        for label, key in (("app", "app"), ("bundle id", "bundleID"), ("executable", "executable"),
                           ("architectures", "architectures"), ("sandboxed", "sandboxed"),
                           ("hardened", "hardened"), ("dyld env", "allowDyld"),
                           ("unsigned libs", "disableLibraryValidation")):
            print(f"{label:16} {result[key]}")
        for path in result["evidencePaths"]:
            print(f"MetalFX evidence {path}")
        print(f"\n{result['summary']}\n{result['detail']}")
    return 0 if result["canObserve" if args.observe else "canLaunchModels"] else 1


if __name__ == "__main__":
    sys.exit(main())
