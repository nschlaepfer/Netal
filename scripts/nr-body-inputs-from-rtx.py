#!/usr/bin/env python3
"""Extract the six body-boundary inputs (entry view, four skip exports, Block0 skip) from an RTX pool state after launch 24 into the encoder-run
layout consumed by nr-body-resident.py / nr-case-validate.py. Inputs: either --pool-after24 FILE (full 212,519,936-byte pool image after launch 24)
or --oracle DIR (block0.pool.bin + capture/pool-snapshots/snapshot-NNNN.delta, replayed through --upto launch). Every extracted region is hashed and
the provenance (source files with SHA-256, launch, offsets) is written to encoder/native_exports.json and block0/run.json. Optional --compare-run
checks the regions byte for byte against an existing encoder-run directory (self-test on the pinned frame: RTX vs the accepted Mac encoder)."""
import argparse, hashlib, json, os, struct, sys
POOL_BYTES = 212519936; HDR = struct.Struct("<8sIIQII")
REGIONS = [("encoder/block22-middle.e4m3", 86649856, 1105920, "entry view (launch 24 entry layout)"), ("encoder/block22-native.e4m3", 90827776, 2211840, "skip export block 22"),
           ("encoder/block14-native.e4m3", 93285376, 4423680, "skip export block 14"), ("encoder/block8-native.e4m3", 97709056, 8847360, "skip export block 8"),
           ("encoder/block4-native.e4m3", 141945856, 17694720, "skip export block 4"), ("block0/skip.e4m3", 388096, 70778880, "Block0 full-resolution output (launch 2); pool range 388096..71166976")]
def sha(b): return hashlib.sha256(b).hexdigest()
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def apply_delta(path, launch, state):
    b = open(path, "rb").read(); magic, l, alloc, abytes, pbytes, n = HDR.unpack_from(b)
    if not (magic == b"NSDLTA1\0" and l == launch and abytes == len(state)): raise SystemExit(f"unexpected delta header in {path}: {magic} launch {l} bytes {abytes}")
    pos = HDR.size; changed = 0
    for _ in range(n):
        page, cnt = struct.unpack_from("<II", b, pos); pos += 8; off = page * pbytes; new = b[pos:pos + cnt]
        if state[off:off + cnt] != new: changed += cnt
        state[off:off + cnt] = new; pos += cnt
    return changed
def main():
    p = argparse.ArgumentParser(); p.add_argument("--out", required=True); g = p.add_mutually_exclusive_group(required=True); g.add_argument("--pool-after24"); g.add_argument("--oracle")
    p.add_argument("--upto", type=int, default=24, help="last launch applied from the oracle deltas"); p.add_argument("--compare-run", default=""); a = p.parse_args()
    if os.path.exists(a.out): raise SystemExit("preserve previous output: " + a.out)
    prov = {"schema": "neuralstudio.nr.body-inputs-from-rtx.v1", "sources": []}
    if a.pool_after24:
        state = bytearray(open(a.pool_after24, "rb").read()); prov["sources"].append({"file": os.path.abspath(a.pool_after24), "sha256": sha(state), "role": "full pool image after launch 24"})
    else:
        base = os.path.join(a.oracle, "block0.pool.bin"); state = bytearray(open(base, "rb").read()); prov["sources"].append({"file": os.path.abspath(base), "sha256": sha(state), "role": "initial pool image"})
        probe = bytearray(state); pre = {n: apply_delta(os.path.join(a.oracle, f"capture/pool-snapshots/snapshot-{n:04d}.delta"), n, probe) for n in (1, 2)}
        first = 1 if (pre[1] or pre[2]) else 3   # block0.pool.bin may already contain launches 1-2 (see nr-vit-oracle-extract.py)
        prov["initial_image_contains_launches_1_2"] = first == 3
        for n in range(first, a.upto + 1):
            f = os.path.join(a.oracle, f"capture/pool-snapshots/snapshot-{n:04d}.delta"); ch = apply_delta(f, n, state); prov["sources"].append({"file": os.path.abspath(f), "sha256": sha_file(f), "launch": n, "changed_bytes": ch})
    if len(state) != POOL_BYTES: raise SystemExit(f"pool image is {len(state)} bytes, expected {POOL_BYTES}")
    outputs = {}; mism = []
    for rel, off, n, role in REGIONS:
        b = bytes(state[off:off + n]); d = os.path.join(a.out, rel); os.makedirs(os.path.dirname(d), exist_ok=True); open(d, "wb").write(b)
        outputs[os.path.basename(rel)] = {"bytes": n, "sha256": sha(b), "pool_offset": off, "role": role, "path": rel}
        if a.compare_run:
            ref = os.path.join(a.compare_run, rel)
            if not os.path.exists(ref): mism.append(f"{rel}: missing in {a.compare_run}")
            elif sha_file(ref) != outputs[os.path.basename(rel)]["sha256"]: mism.append(f"{rel}: differs from {ref}")
    ne = {"schema": "neuralstudio.nr.native-exports.rtx-extract.v1", "producer": "RTX pool state after launch %d (extracted, not Mac-produced)" % a.upto, "outputs": {k: v for k, v in outputs.items() if k != "skip.e4m3"},
          "expected_input_opened": False, "live_device_inputs": False, "provenance": prov}
    json.dump(ne, open(os.path.join(a.out, "encoder/native_exports.json"), "w"), indent=1)
    json.dump({"schema": "neuralstudio.nr.block0-skip.rtx-extract.v1", "skip": outputs["skip.e4m3"], "provenance": prov}, open(os.path.join(a.out, "block0/run.json"), "w"), indent=1)
    if a.compare_run: json.dump({"compare_run": os.path.abspath(a.compare_run), "mismatches": mism, "result": "identical" if not mism else "differs"}, open(os.path.join(a.out, "compare.json"), "w"), indent=1)
    for k, v in outputs.items(): print(f"{k:24s} {v['bytes']:>10d} bytes  {v['sha256'][:16]}")
    print("compare:", "identical to " + a.compare_run if a.compare_run and not mism else (mism if mism else "n/a")); sys.exit(1 if mism else 0)
if __name__ == "__main__": main()
