"""Generate the batched-pipeline JSONs for every resolution profile of the runtime bundle.
Geometry per layer comes from the profile's arguments templates (+0 = W, +4 = H of the layer's token grid).
Writes profiles/<WxH>/pipeline/<entry>.json (and the legacy root pipeline/ for 1920x1080); weights stay at
the bundle root (pipeline/<layer>-weights.bin). usage: build-profile-pipelines.py [bundle]"""
import json, os, struct, subprocess, sys, tempfile
from pathlib import Path
_argv = sys.argv[1:]
_synth = None
_layouts = None
if "--layout-dir" in _argv:
    _i = _argv.index("--layout-dir")
    _layouts = Path(_argv[_i+1]).resolve()
    del _argv[_i:_i+2]
if "--synth" in _argv:            # --synth <exported dir> <WxH>
    _i = _argv.index("--synth")
    _synth = (_argv[_i + 1], _argv[_i + 2])
    del _argv[_i:_i + 3]
B = _argv[0] if _argv else "work/re9-sr-runtime-bundle"
# Formerly gated: the attention layers are correct once their per-frame window-shift phases are
# generated (SHIFTS below); PWIN_DISABLE_ATTENTION=1 writes them as .off again.
ATTENTION = {"enc4", "dec5", "dec4"} if os.environ.get("PWIN_DISABLE_ATTENTION") == "1" else set()
# Per-frame window-shift schedule (period 8) copied from kPwinShift* in src/macos_host/dlss_sr_session.mm.
# The five deep layers shift their window grid every frame; phase 0 must equal the arguments template's
# (padx, pady), which is asserted below.
SHIFT_A = [(4, 4), (0, 4), (4, 4), (0, 4), (0, 0), (4, 0), (0, 0), (4, 0)]
SHIFT_B = [(4, 0), (6, 0), (4, 4), (6, 4), (2, 6), (0, 6), (2, 2), (0, 2)]
SHIFT_C = [(0, 2), (5, 6), (4, 0), (1, 4), (7, 5), (2, 1), (3, 7), (6, 3)]
SHIFTS = {"enc3": SHIFT_A, "enc4": SHIFT_B, "dec5": SHIFT_C, "dec4": SHIFT_B, "dec3": SHIFT_A}   # enc0-2, dec2, dec1, dec0 keep the template padding
LAYERS = [("enc0", "dltss_pwin_enc0_layer", "pwin/pwin-11/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-layer.py", "enc0"]),
          ("enc1", "dltss_pwin_enc1_layer", "pwin/pwin-02/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-enc1.py"]),
          ("dec0", "dltss_pwin_dec0_layer", "pwin/pwin-11/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-layer.py", "dec0"]),
          ("enc4", "dltss_pwin_enc4_layer", "pwin/pwin-05/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-enc4.py"]),
          ("dec5", "dltss_pwin_dec5_layer", "pwin/pwin-06/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-dec5.py"]),
          ("dec4", "dltss_pwin_dec4_layer", "pwin/pwin-07/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-dec4.py"]),
          ("enc2", "dltss_pwin_enc2_layer", "pwin/pwin-03/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-enc2.py"]),
          ("enc3", "dltss_pwin_enc3_layer", "pwin/pwin-04/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-enc3.py"]),
          ("dec3", "dltss_pwin_dec3_layer", "pwin/pwin-08/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-dec3.py"]),
          ("dec2", "dltss_pwin_dec2_layer", "pwin/pwin-09/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-dec2.py"]),
          ("dec1", "dltss_pwin_dec1_layer", "pwin/pwin-10/arguments-template.bin", ["python3", "tools/pwin_pipeline/make-dec1.py"])]
# Per-dispatch nsplit for pwin_gemm3_tokens, found by greedy search on the 1920x1080 stage
# snapshots (tools/pwin_pipeline/README-nsplit or docs/apple-gpu-study.md section T.2). nsplit only
# changes how output columns are partitioned across SIMD groups, so every candidate was re-checked
# bit-identical in the harness before being kept. Keyed by layer then dispatch index.
NSPLIT = {"enc0": {1: 2, 3: 2, 9: 4}, "enc1": {1: 2, 3: 2, 6: 2}, "enc2": {1: 6, 6: 6},
          "enc3": {3: 6, 5: 12, 8: 8}, "enc4": {1: 12, 3: 8, 5: 16, 8: 10},
          "dec5": {1: 16, 3: 10}, "dec4": {0: 16, 3: 8, 5: 4},
          "dec3": {0: 8, 3: 8, 5: 3, 7: 8}, "dec2": {5: 2}, "dec0": {0: 2}}
def tune_nsplit(layer, q):
    """Apply the measured nsplit to a generated pipeline, re-deriving each dispatch's grid."""
    for index, nsplit in NSPLIT.get(layer, {}).items():
        d = q["dispatches"][index]
        assert d["function"] == "pwin_gemm3_tokens", f"{layer} dispatch {index} is {d['function']}"
        n = d["constants"]["1"]
        assert n % nsplit == 0 and (n // nsplit) % 16 == 0, f"{layer} dispatch {index}: N={n} nsplit={nsplit}"
        d["constants"]["16"] = nsplit
        tokens = d["bytes"][0]["u32"][0]
        d["grid"] = [(((tokens + 15) // 16) * nsplit + 7) // 8, 1, 1]
    return q
def geometry(path):
    return struct.unpack_from("<2I", open(path, "rb").read(), 0)
def padding(path):
    return struct.unpack_from("<2I", open(path, "rb").read(), 56)
def build_one(prof, base, out_dir, root_layout=False, profile_dir=None):
    """Generate every layer's pipeline (and its window-shift phases) for one profile geometry.
    `base` supplies the PWin arguments/replay configs; `out_dir` receives the JSONs."""
    os.makedirs(out_dir, exist_ok=True)
    profile_dir = profile_dir or base
    for layer, entry, tmpl, cmd in LAYERS:
        tpath = f"{base}/{tmpl}"
        if root_layout and layer == "enc0": tpath = f"{base}/pwin/enc0-arguments-template.bin"
        W, H = geometry(tpath)
        phases = SHIFTS.get(layer)
        if phases is not None:
            tp = padding(tpath)
            assert tuple(phases[0]) == tp, f"{layer}: template padding {tp} != phase 0 {phases[0]}"
        variants = [(None, None)] if phases is None else list(enumerate(phases))
        p = None
        for phase, shift in variants:
            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp: out = tmp.name
            env = dict(os.environ, PWIN_W=str(W), PWIN_H=str(H), PWIN_MIXP_TGS="640", PWIN_PROFILE_DIR=profile_dir)
            if shift is not None: env["PWIN_PADX"], env["PWIN_PADY"] = str(shift[0]), str(shift[1])
            if _layouts is not None: env["PWIN_LAYOUT_FILE"] = str(_layouts / f"layout-{layer}.json")
            command = [sys.executable, str(Path(__file__).resolve().with_name(Path(cmd[1]).name))] + cmd[2:]
            r = subprocess.run(command + [out], env=env, capture_output=True, text=True)
            if r.returncode != 0: raise SystemExit(f"{prof} {layer} phase {phase}: {r.stderr}")
            q = json.load(open(out)); os.remove(out)
            tune_nsplit(layer, q)
            q["library"] = "pwin_all"; q["files"] = [{"name": "pweights", "path": f"pipeline/{layer}-weights.bin"}]
            if phase not in (None, 0):
                path = f"{out_dir}/{entry}.phase{phase}.json"
                disabled_p = layer in ATTENTION
                json.dump(q, open(path + (".off" if disabled_p else ""), "w"))
                stale = path if disabled_p else path + ".off"
                if os.path.exists(stale): os.remove(stale)
                continue
            p = q
        disabled = layer in ATTENTION
        json.dump(p, open(f"{out_dir}/{entry}.json" + (".off" if disabled else ""), "w"))
        for stale in ((f"{out_dir}/{entry}.json",) if disabled else (f"{out_dir}/{entry}.json.off",)):
            if os.path.exists(stale): os.remove(stale)
        print(f"{prof:10s} {layer} W,H=({W},{H}) dispatches={len(p['dispatches'])} scratch={p['scratch_bytes']/1e6:.1f} MB{' [disabled]' if disabled else ''}")

# --synth <exported dir> <WxH>: build pipelines for a resolution the bundle has no profile for. Export the
# synthesized artefacts first with NEURALSTUDIO_SR_EXPORT_SYNTH_DIR (the session owns the synthesis rules);
# the result lands in <bundle>/pipeline-synth/<WxH>/, where the session looks when the geometry is synthesized.
if _synth is not None:
    synth_dir, res = _synth
    build_one(res, synth_dir, f"{B}/pipeline-synth/{res}")
    raise SystemExit(0)

profiles = sorted(d for d in os.listdir(f"{B}/profiles") if os.path.isdir(f"{B}/profiles/{d}/pwin"))
for prof in profiles + ["ROOT"]:
    base = f"{B}/profiles/{prof}" if prof != "ROOT" else B
    build_one(prof, base, f"{base}/pipeline", root_layout=(prof == "ROOT"),
              profile_dir=(f"{B}/profiles/1920x1080" if prof == "ROOT" else base))
