#!/usr/bin/env python3
"""Compile, verify, and benchmark Netal's pure-GPU multi-scale optical-flow synthesizer."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout + proc.stderr)
        proc.check_returncode()
    return proc


def main() -> int:
    repo = Path(__file__).resolve().parents[1]
    metal_src = repo / "src/macos_host/netal_nr_optical_flow.metal"
    host_src = repo / "src/macos_host/netal_nr_optical_flow_test.mm"

    with tempfile.TemporaryDirectory(prefix="netal-nr-optical-flow-") as tmp:
        out = Path(tmp)
        air = out / "netal_nr_optical_flow.air"
        metallib = out / "netal_nr_optical_flow.metallib"
        binary = out / "netal_nr_optical_flow_test"

        run(["xcrun", "--sdk", "macosx", "metal", "-std=metal3.1", "-O2",
             "-Wall", "-Wextra", "-Werror", "-c", str(metal_src), "-o", str(air)])
        run(["xcrun", "--sdk", "macosx", "metallib", str(air), "-o", str(metallib)])
        run(["xcrun", "--sdk", "macosx", "clang++", "-arch", "arm64", "-std=c++20",
             "-fobjc-arc", "-O2", "-Wall", "-Wextra", "-Werror",
             "-framework", "Foundation", "-framework", "Metal",
             str(host_src), "-o", str(binary)])

        proc = run([str(binary), str(metallib)])
        report = json.loads(proc.stdout)
        print(json.dumps(report, indent=2))
        if report.get("skipped"):
            print(f"SKIP optical flow GPU run: {report.get('reason')}")
            return 0

        static = report["static"]
        assert not static["scene_cut"], "static pair falsely flagged scene cut"
        assert static["mae_dx"] < 0.01 and static["mae_dy"] < 0.01, f"static drift: {static}"

        s1 = report["shift_8_neg4"]
        assert not s1["scene_cut"], "shift (+8, -4) falsely flagged scene cut"
        assert abs(s1["mean_dx"] - 8.0) < 0.35 and abs(s1["mean_dy"] - (-4.0)) < 0.35, f"shift (+8, -4) inaccurate: {s1}"
        assert s1["mae_dx"] < 0.5 and s1["mae_dy"] < 0.5, f"shift (+8, -4) high MAE: {s1}"

        s2 = report["shift_neg12_6"]
        assert not s2["scene_cut"], "shift (-12, +6) falsely flagged scene cut"
        assert abs(s2["mean_dx"] - (-12.0)) < 0.35 and abs(s2["mean_dy"] - 6.0) < 0.35, f"shift (-12, +6) inaccurate: {s2}"
        assert s2["mae_dx"] < 0.5 and s2["mae_dy"] < 0.5, f"shift (-12, +6) high MAE: {s2}"

        s3 = report["shift_48_neg32"]
        assert not s3["scene_cut"], "large shift (+48, -32) falsely flagged scene cut"
        assert abs(s3["mean_dx"] - 48.0) < 0.6 and abs(s3["mean_dy"] - (-32.0)) < 0.6, f"shift (+48, -32) inaccurate: {s3}"

        s_odd = report["shift_5_neg3_rgba16f"]
        assert not s_odd["scene_cut"], "odd RGBA16F shift (+5, -3) falsely flagged scene cut"
        assert abs(s_odd["mean_dx"] - 5.0) < 0.5 and abs(s_odd["mean_dy"] - (-3.0)) < 0.5, f"shift (+5, -3) RGBA16F inaccurate: {s_odd}"
        assert s_odd["mae_dx"] < 0.6 and s_odd["mae_dy"] < 0.6, f"shift (+5, -3) RGBA16F high MAE: {s_odd}"

        cut = report["scene_cut"]
        assert cut["scene_cut"] is True and cut["gpu_flag"] == 1, f"scene cut not detected: {cut}"
        assert cut["mean_luma_diff"] > 0.18, f"scene cut luma diff too low: {cut}"
        assert cut["mae_dx"] == 0.0 and cut["mae_dy"] == 0.0, f"scene cut did not zero flow: {cut}"

        bench = report["benchmark_1080p"]
        if bench["samples"] > 0:
            assert 0.0 < bench["median_gpu_ms"] < 20.0, f"1080p median GPU time {bench['median_gpu_ms']:.4f} ms out of range"

        print(f"PASS optical flow verified on {report['device']} at {report['resolution']}: "
              f"median_gpu_ms={bench['median_gpu_ms']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
