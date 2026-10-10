#!/usr/bin/env python3
"""Read a Netal session log and say whether the frame-time decay is the machine or the runtime.

The decay is the open question: over a long Death Stranding session the SR pass went 8.24 -> 36.97 ms
while frame rate halved. `command_gpu_ms` alone cannot answer why, because it is GPU wall-clock for
our command buffer and inflates identically whether our code got slower or the GPU got slower
underneath it. `sr_pacing` now also carries `thermal=`, so the two can be lined up.

    tools/netal-decay-report.py [log]

Reads the live Netal log by default. It reports the curve, and then the one comparison that
discriminates: whether the SR pass degrades in step with the machine's thermal state.
"""
import glob, os, re, sys

DEFAULT = os.path.expanduser("~/Library/Containers/*/Data/netal/netal-live.log")
THERMAL_ORDER = {"nominal": 0, "fair": 1, "serious": 2, "critical": 3}


def newest_log() -> str | None:
    logs = glob.glob(DEFAULT)
    return max(logs, key=os.path.getmtime) if logs else None


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else newest_log()
    if not path or not os.path.isfile(path):
        print("no log found", file=sys.stderr)
        return 2

    pacing, gpu = {}, {}
    for line in open(path, errors="ignore"):
        m = re.search(r"sr_pacing frame=(\d+) .*?mean_ms=([0-9.]+) fps=([0-9.]+) min_ms=([0-9.]+) "
                      r".*?spikes_over_25ms=(\d+) worst_ms_session=[0-9.]+ rss_mb=(\d+)", line)
        if m:
            frame = int(m.group(1))
            thermal = re.search(r"thermal=(\w+)", line)
            pacing[frame] = (float(m.group(2)), float(m.group(3)), float(m.group(4)),
                             int(m.group(5)), int(m.group(6)),
                             thermal.group(1) if thermal else None)
        m = re.search(r"sr_exact_completed .*?frame=(\d+) .*?command_gpu_ms=([0-9.]+)", line)
        if m:
            gpu[int(m.group(1))] = float(m.group(2))

    if not pacing:
        print("no pacing samples in %s" % path, file=sys.stderr)
        return 2
    frames = sorted(pacing)
    print("log     %s" % path)
    print("window  frames %d..%d  (%d samples)" % (frames[0], frames[-1], len(frames)))
    has_thermal = any(pacing[f][5] for f in frames)
    if not has_thermal:
        print("thermal not recorded — this session predates the instrumentation, so the question")
        print("        it was added to answer cannot be answered from it.")

    print()
    print("  %-9s %8s %8s %8s %9s %8s %9s" %
          ("frame", "fps", "frame ms", "min ms", "SR ms", "rss MB", "thermal"))
    step = max(1, len(frames) // 14)
    for f in frames[::step]:
        mean, fps, low, _, rss, thermal = pacing[f]
        near = min(gpu, key=lambda x: abs(x - f)) if gpu else None
        print("  %-9d %8.1f %8.2f %8.2f %9s %8d %9s"
              % (f, fps, mean, low,
                 ("%.2f" % gpu[near]) if near is not None else "-", rss, thermal or "-"))

    # The discriminator: did the SR pass degrade while the machine stayed cool?
    if gpu and len(frames) > 10:
        k = max(1, len(frames) // 5)
        early = [f for f in frames[:k]], [f for f in frames[-k:]]
        def sr(group):
            vals = [gpu[min(gpu, key=lambda x: abs(x - f))] for f in group]
            return sum(vals) / len(vals)
        first, last = sr(early[0]), sr(early[1])
        print()
        print("  SR pass   %.2f ms  ->  %.2f ms   (%+.0f%%)" % (first, last, (last - first) / first * 100))
        if has_thermal:
            worst_early = max((THERMAL_ORDER.get(pacing[f][5] or "nominal", 0) for f in early[0]))
            worst_late = max((THERMAL_ORDER.get(pacing[f][5] or "nominal", 0) for f in early[1]))
            names = {v: k for k, v in THERMAL_ORDER.items()}
            print("  thermal   %-9s ->  %-9s" % (names[worst_early], names[worst_late]))
            print()
            if last > first * 1.5 and worst_late <= worst_early:
                print("  VERDICT: the SR pass degraded while the machine stayed cool — this is ours.")
            elif last > first * 1.5:
                print("  VERDICT: the SR pass degraded as the machine heated — this is the machine.")
            else:
                print("  VERDICT: no material degradation in this session.")
        else:
            print("  no verdict: thermal was not recorded.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
