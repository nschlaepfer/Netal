#!/usr/bin/env bash
set -euo pipefail

repository="$(cd "$(dirname "$0")/.." && pwd)"
attempt_root="${1:-}"
if [[ -z "$attempt_root" ]]; then
  attempt_root="$repository/build/fg_postprocess_l86/macos/attempt-$(date -u +%Y%m%dT%H%M%SZ)"
fi
if [[ -e "$attempt_root" ]]; then
  echo "Attempt directory already exists; preserve it and choose a new path: $attempt_root" >&2
  exit 2
fi

source_root="$repository/src/fg_postprocess_l86"
build_root="$attempt_root/build"
archive="$repository/evidence/checkpoints/2026-08-31-sm120-rcp-rom/external-evidence-sm120-rcp-rom.tar.gz"
rom="$build_root/sm120-rcp-correction-rom.bin"
started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
mkdir -p "$build_root"

tar -xOf "$archive" \
  work/g23-rcp-architecture/m3-rcp-rom/correction-rom.bin >"$rom"
rom_hash="$(shasum -a 256 "$rom" | awk '{print $1}')"
if [[ "$rom_hash" != "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750" ]]; then
  echo "SM120 RCP architecture ROM identity mismatch: $rom_hash" >&2
  exit 3
fi

xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -c "$source_root/postprocess_l86_mode0.metal" \
  -o "$build_root/postprocess_l86_mode0.air" \
  >"$build_root/exact-metal-compile.log" 2>&1
xcrun -sdk macosx metallib \
  "$build_root/postprocess_l86_mode0.air" \
  -o "$build_root/postprocess_l86_mode0.metallib" \
  >"$build_root/exact-metallib-link.log" 2>&1
xcrun -sdk macosx metal -std=metal3.1 -fno-fast-math -ffp-contract=off \
  -DFG_L86_NEGATIVE_PRECISE_DIVISION=1 \
  -c "$source_root/postprocess_l86_mode0.metal" \
  -o "$build_root/postprocess_l86_precise_negative.air" \
  >"$build_root/negative-metal-compile.log" 2>&1
xcrun -sdk macosx metallib \
  "$build_root/postprocess_l86_precise_negative.air" \
  -o "$build_root/postprocess_l86_precise_negative.metallib" \
  >"$build_root/negative-metallib-link.log" 2>&1
xcrun -sdk macosx clang++ \
  -std=c++20 -fobjc-arc -Wall -Wextra -Werror -Wno-deprecated-declarations \
  "$source_root/macos_probe.mm" \
  -framework Foundation -framework Metal \
  -o "$build_root/fg-postprocess-l86-mac" \
  >"$build_root/host-compile.log" 2>&1

run_case() {
  local run_id="$1"
  local fixture="$2"
  local poison="$3"
  local metallib="$4"
  local run_root="$attempt_root/$run_id"
  mkdir -p "$run_root"
  "$build_root/fg-postprocess-l86-mac" \
    "$metallib" "$rom" "$run_root" "$run_id" "$fixture" "$poison" 0 \
    fg_postprocess_l86_mode0 \
    >"$run_root/stdout.log" 2>"$run_root/stderr.log"
}

run_case pyramid-a-poison-a5 pyramid-a a5 "$build_root/postprocess_l86_mode0.metallib"
run_case pyramid-b-poison-a5 pyramid-b a5 "$build_root/postprocess_l86_mode0.metallib"
run_case pyramid-a-poison-5a pyramid-a 5a "$build_root/postprocess_l86_mode0.metallib"
run_case pyramid-b-poison-5a pyramid-b 5a "$build_root/postprocess_l86_mode0.metallib"
run_case negative-precise pyramid-a a5 "$build_root/postprocess_l86_precise_negative.metallib"

negative_mode_root="$attempt_root/reject-mode-1"
mkdir -p "$negative_mode_root"
set +e
"$build_root/fg-postprocess-l86-mac" \
  "$build_root/postprocess_l86_mode0.metallib" "$rom" \
  "$negative_mode_root" reject-mode-1 pyramid-a a5 1 \
  fg_postprocess_l86_mode0 \
  >"$negative_mode_root/stdout.log" 2>"$negative_mode_root/stderr.log"
negative_mode_exit=$?
set -e
if [[ "$negative_mode_exit" -ne 4 ]]; then
  echo "Mode-1 rejection gate failed: exit=$negative_mode_exit" >&2
  exit 4
fi
if [[ -e "$negative_mode_root/run-receipt.json" ]]; then
  echo "Rejected mode unexpectedly produced a live-execution receipt" >&2
  exit 5
fi

python3 - "$attempt_root" "$source_root" "$build_root" "$started" <<'PY'
import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

attempt = Path(sys.argv[1])
source = Path(sys.argv[2])
build = Path(sys.argv[3])
started = sys.argv[4]

def artifact(path: Path) -> dict[str, object]:
    data = path.read_bytes()
    return {
        "relative_path": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

gpu_lines = subprocess.run(
    ["system_profiler", "SPDisplaysDataType"],
    check=True,
    capture_output=True,
    text=True,
).stdout.splitlines()
gpu = next(
    (line.split(":", 1)[1].strip() for line in gpu_lines if "Chipset Model:" in line),
    "unknown",
)
receipt = {
    "schema": "neuralstudio.fg.postprocess-l86-mac-build.v1",
    "started_utc": started,
    "ended_utc": subprocess.run(
        ["date", "-u", "+%Y-%m-%dT%H:%M:%SZ"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip(),
    "host": {"machine": platform.machine(), "macos": platform.mac_ver()[0], "gpu": gpu},
    "selected_launch_token": 86,
    "source_module_token": 19,
    "source_module_sha256": "143c6712c99db4a5792df697d68178afeea3657f4f0eeb12f905c90abf38a1b0",
    "raw_parameter_sha256": "faea46ab664868a3ad3bcdcf66dbb7350fddd82fb8d65031b2ea2cec12eb8925",
    "address_free_parameter_sha256": "8f2b991ee0a0288d0f79777881720354e691b920393d921ff53a30b7f27d5fc9",
    "mode": 0,
    "mode_1_rejected": True,
    "architecture_rom_sha256": "7cf4f8cece6b18b07605edc9cb601a153346a79b10bf3b93c7ec058d57d29750",
    "source_artifacts": [
        artifact(source / "postprocess_l86_mode0.metal"),
        artifact(source / "macos_probe.mm"),
    ],
    "build_artifacts": [
        artifact(build / "postprocess_l86_mode0.metallib"),
        artifact(build / "postprocess_l86_precise_negative.metallib"),
        artifact(build / "fg-postprocess-l86-mac"),
    ],
    "external_expected_tensors_consumed": False,
    "archived_intermediates_consumed": False,
    "address_fields_recorded": False,
    "result": "pass",
}
path = attempt / "mac-build-receipt.json"
path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
(attempt / "mac-build-receipt.sha256").write_text(
    f"{digest}  mac-build-receipt.json\n"
)
PY

python3 "$repository/scripts/verify-fg-postprocess-l86.py" backend \
  --backend mac --root "$attempt_root" \
  --output "$attempt_root/backend-receipt.json" \
  >"$attempt_root/backend-verification.log"
shasum -a 256 "$attempt_root/backend-receipt.json" \
  >"$attempt_root/backend-receipt.sha256"

echo "attempt_root=$attempt_root"
echo "rom_sha256=$rom_hash"
echo "backend_receipt_sha256=$(awk '{print $1}' "$attempt_root/backend-receipt.sha256")"
echo "result=pass"
