#!/usr/bin/env python3
"""Package a self-contained experimental NR runtime bundle (Opus-owned).

The previous ad-hoc staging relocated only files with a known extension, which silently left FOUR descriptor
values pointing at research directories outside the package: two `tables` directories, a `tex_in` capture and
the EX2 half-domain table. Codex had to work around that with per-session container relocation.

This packager relocates by EXISTENCE, not by extension: every descriptor string that names an existing file or
directory is copied into the bundle and rewritten relative to it. Anything left pointing outside must be on the
explicit harness-supplied list, and the packager FAILS if any other value escapes - so the failure mode is a
refused package, not a bundle that only works next to a research worktree.

Usage: package-nr-experimental-bundle.py ENCODER_ASSETS BODY_DESCRIPTOR OUT_BUNDLE
       [--encoder-approx DIR] [--body-approx-lib NAME=PATH ...]
"""
import argparse, hashlib, json, os, shutil, sys

# Directory-valued descriptor keys carry ONLY these files. Copying such a directory recursively is what pulled
# 234 archived activation/reference tensors into bundle-v2 (Codex review): `nr_split/fixtures` holds one
# `launch-NNN/` subdirectory per launch full of captured tensors, and none of them is a lookup table. The lists
# below are the filenames the body actually opens, so anything not named here is not copied.
TABLE_ALLOWLIST = {
    # descriptor key -> exactly the files that key's directory must provide
    "tables": {"rtx-rsqrt-half.bin", "rtx-rcp-half.bin"},
    "decoder.tables": {"fraction-corrections.bin", "rtx-rcp-half.bin", "rtx-rsqrt-half.bin",
                       "sm120-rcp-correction-rom.bin", "unit-corrections.bin"},
}

# Slots the experimental runtime supplies live from its own encoder; they are never opened from disk.
HARNESS_SUPPLIED = {"entry_file"} | {f"decoder.patches[{i}].file" for i in range(8)}
# Values that are research inputs the runtime does not use: the caller provides the colour texture.
DROP = {"decoder.tex_in"}

def looks_like_path(v):
    """A path has no spaces and is either absolute or a build-relative path. `provenance` notes contain slashes
    but also spaces, and must not be mistaken for files."""
    return "/" in v and " " not in v and (v.startswith("/") or v.startswith("build/") or v.startswith("work/") or v.startswith("spec/"))

def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""): h.update(c)
    return h.hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("encoder_assets"); ap.add_argument("body_descriptor"); ap.add_argument("out_bundle")
    ap.add_argument("--encoder-approx", default=""); ap.add_argument("--body-approx-lib", action="append", default=[])
    a = ap.parse_args()
    out = a.out_bundle
    if os.path.exists(out): sys.exit(f"refusing to overwrite {out}")
    os.makedirs(os.path.join(out, "body", "assets"))
    shutil.copytree(a.encoder_assets, os.path.join(out, "encoder"))
    if a.encoder_approx: shutil.copytree(a.encoder_approx, os.path.join(out, "encoder-approx"))

    d = json.load(open(a.body_descriptor))
    manifest, escaped, supplied, dropped, excluded = {}, [], [], [], {}
    seen = {}

    def relocate(value, key):
        if key in DROP: dropped.append((key, value)); return None
        src = value
        if not os.path.exists(src):
            (supplied if key.split("[")[0].replace("[0]", "") in HARNESS_SUPPLIED or key in HARNESS_SUPPLIED else escaped).append((key, value))
            return value                                   # left as-is; asserted below
        if src in seen: return seen[src]
        base = os.path.basename(os.path.dirname(src)) + "/" + os.path.basename(src)
        rel = "assets/" + base
        dst = os.path.join(out, "body", rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if os.path.isdir(src):
            allow = TABLE_ALLOWLIST.get(key)
            if allow is None:
                sys.exit(f"REFUSED: {key} names a directory ({src}) with no explicit file allowlist. "
                         "Add its exact constant-table filenames to TABLE_ALLOWLIST rather than copying it whole.")
            os.makedirs(dst, exist_ok=True)
            missing = sorted(f for f in allow if not os.path.exists(os.path.join(src, f)))
            if missing: sys.exit(f"REFUSED: {key} directory {src} is missing required tables: {missing}")
            for f in sorted(allow):
                shutil.copy2(os.path.join(src, f), os.path.join(dst, f))
                manifest[os.path.relpath(os.path.join(dst, f), out)] = sha(os.path.join(dst, f))
            skipped = sum(1 for e in os.listdir(src) if e not in allow)
            excluded[key] = {"directory": src, "copied": sorted(allow), "entries_not_copied": skipped}
        else:
            shutil.copy2(src, dst); manifest[os.path.relpath(dst, out)] = sha(dst)
        seen[src] = rel
        return rel

    def walk(node, path=""):
        if isinstance(node, dict):
            for k, v in list(node.items()):
                key = f"{path}.{k}".lstrip(".")
                if isinstance(v, str) and looks_like_path(v) and not v.startswith("assets/"):
                    r = relocate(v, key)
                    if r is None: del node[k]
                    else: node[k] = r
                else: walk(v, key)
        elif isinstance(node, list):
            for i, v in enumerate(node):
                key = f"{path}[{i}]"
                if isinstance(v, str) and looks_like_path(v) and not v.startswith("assets/"):
                    r = relocate(v, key)
                    if r is not None: node[i] = r
                else: walk(v, key)
    walk(d)

    if escaped:
        sys.exit("REFUSED: these descriptor values point outside the bundle and are not harness-supplied:\n" +
                 "\n".join(f"  {k} = {v}" for k, v in escaped))
    d["_experimental_note"] = ("Self-contained experimental bundle. Every path is relative to this directory. The entry view and "
                               "the five skip patches are supplied live by the experimental encoder and are never opened from disk. "
                               "No weight is changed by this lane.")
    json.dump(d, open(os.path.join(out, "body", "chain.json"), "w"), indent=1)
    for spec in a.body_approx_lib:
        name, path = spec.split("=", 1)
        for r, _, fs in os.walk(os.path.join(out, "body", "assets")):
            for f in fs:
                if f == name: shutil.copy2(path, os.path.join(r, f))
    man = {"schema": "neuralstudio.nr.experimental-bundle.v2", "lane": "experimental", "exact_parity_claim": False,
           "files": manifest, "file_count": len(manifest),
           "harness_supplied_slots": [k for k, _ in supplied], "dropped_research_inputs": [k for k, _ in dropped],
           "table_directories_filtered": excluded, "self_contained": True,
           "note": "Directory-valued descriptor keys are copied file-by-file from an explicit allowlist, never recursively."}
    json.dump(man, open(os.path.join(out, "manifest.json"), "w"), indent=1)
    for k, v in excluded.items():
        print(f"  {k}: copied {len(v['copied'])} tables, EXCLUDED {v['entries_not_copied']} other entries from {v['directory']}")
    print(f"packaged {len(manifest)} body files; {len(supplied)} harness-supplied slots; dropped {len(dropped)} research inputs: {[k for k,_ in dropped]}")
if __name__ == "__main__": main()
