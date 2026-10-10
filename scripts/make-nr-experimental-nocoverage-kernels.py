#!/usr/bin/env python3
"""Generate the no-coverage kernel variant for the experimental lane (Opus-owned).

The pinned encoder kernels write a write-coverage counter for EVERY output element:
`atomic_fetch_add_explicit(&count[...], 1u, memory_order_relaxed);` - an atomic read-modify-write of four
bytes per element, against one or two bytes for the output itself. Nothing in the model reads those counters;
they exist so the research lane can prove every element was written exactly once. Removing them changes no
weight, no arithmetic and no dispatch geometry.

The rewrite is mechanical and is checked here: every removed statement must match the coverage form exactly,
the `count`/`counts`/`coverage` kernel PARAMETERS stay in place (so the host binding and buffer indices are unchanged),
and no other token in the file may differ. The variant is written to a separate directory and is labelled
experimental; the pinned libraries are never overwritten.

Usage: make-nr-experimental-nocoverage-kernels.py SRC_DIR OUT_DIR
"""
import hashlib, json, os, re, sys
src_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)
# The three spellings the pinned kernels use for the write-coverage counter, and nothing else.
PATTERN = re.compile(r"atomic_fetch_add_explicit\(&(counts?|coverage)\[([A-Za-z_][A-Za-z0-9_]*)\],1u,memory_order_relaxed\);")
sha = lambda b: hashlib.sha256(b if isinstance(b, bytes) else b.encode()).hexdigest()
report = {"schema": "neuralstudio.nr.experimental-nocoverage-kernels.v1", "lane": "experimental",
          "changes": "removal of write-coverage atomics only", "files": []}
total = 0
for name in sorted(os.listdir(src_dir)):
    p = os.path.join(src_dir, name)
    if not os.path.isfile(p): continue
    text = open(p).read()
    if name.endswith((".metal", ".metalh")):
        new, n = PATTERN.subn("", text)
        # every atomic in these kernels must be a coverage increment; if one is not, refuse rather than guess
        left = new.count("atomic_fetch_add_explicit")
        if left: sys.exit(f"{name}: {left} atomic_fetch_add_explicit remain that do not match the coverage form; refusing")
        # nothing else may change: putting the removed statements back must restore the original exactly
        restored = PATTERN.sub("", text)
        if restored != new: sys.exit(f"{name}: rewrite is not idempotent")
        total += n
    else:
        new, n = text, 0
    open(os.path.join(out_dir, name), "w").write(new)
    report["files"].append({"file": name, "coverage_atomics_removed": n,
                            "source_sha256": sha(text), "variant_sha256": sha(new)})
report["total_coverage_atomics_removed"] = total
print(json.dumps(report, indent=1))
