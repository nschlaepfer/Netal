"""Acceptance gate for a case-aware body validation (nr-case-validate.py v2). Pure function over the runner's receipt, the chain descriptor,
the child return code and the case files, so it can be exercised by CPU negative tests without a GPU. Fail-closed: every check must hold; the
ONLY exempted failures are the explicitly scoped comparisons against the pinned strict frame, which legitimately differ for a new input."""
import ast, hashlib, os
import json, re
# The ONLY failures a new input may legitimately produce: comparisons against the pinned strict frame, matched by anchored grammar (the runner's
# exact message forms in scripts/nr-body-resident.py). Anything else, including any other "... hash differs" text, is a genuine failure.
OLD_FRAME_EXEMPT_PATTERNS = [re.compile(x) for x in (
    r"^launch \d+: pool hash differs from the strict reference$", r"^launch \d+ [A-Za-z0-9_]+: hash differs from the strict reference$", r"^launch \d+: missing reference [A-Za-z0-9_]+$",
    r"^block \d+ [A-Za-z0-9_]+: hash differs from the strict reference$", r"^block \d+: missing reference [A-Za-z0-9_]+$", r"^repack \d+: hash differs$",
    r"^middle final view differs from the strict reference$", r"^final pool hash differs from the strict reference's last launch$", r"^restaged decoder pre-image differs from the strict body's pool_pre$")]
# the six boundary ports: descriptor role -> (relative producer path, pool offset, bytes, sealed-hash source)
PORTS = [("encoder/block22-native.e4m3", 90827776, 2211840), ("encoder/block14-native.e4m3", 93285376, 4423680), ("encoder/block8-native.e4m3", 97709056, 8847360),
         ("encoder/block4-native.e4m3", 141945856, 17694720), ("block0/skip.e4m3", 388096, 70778880)]
ENTRY = ("encoder/block22-middle.e4m3", 86649856, 1105920)
PINNED_BLOCK0_WEIGHT_SHA = "5fe2ab86289f7b21253d00368ce7a95f005ec3c8476abd7db22550b918962a5c"   # original model weights (Block0 record), same for every case
def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()
def is_exempt(f): return isinstance(f, str) and any(P.match(f) for P in OLD_FRAME_EXEMPT_PATTERNS)
def _real(f, root):
    return os.path.realpath(f if os.path.isabs(f) else os.path.join(root, f))
def producer_checks(encoder_run, case_tex_sha, r):
    """sealed producer receipts: exports hashes/sizes, Block0 receipt bound to the case input and the pinned weights, all results pass"""
    ne_p = os.path.join(encoder_run, "encoder/native_exports.json"); b0_p = os.path.join(encoder_run, "block0/run.json"); sealed = {}
    try: ne = json.load(open(ne_p))
    except Exception: r.append("producer encoder receipt missing/unreadable (encoder/native_exports.json)"); ne = None
    try: b0 = json.load(open(b0_p))
    except Exception: r.append("producer Block0 receipt missing/unreadable (block0/run.json)"); b0 = None
    if ne:
        if ne.get("result") != "pass": r.append(f"producer encoder receipt result {ne.get('result')!r}")
        if ne.get("expected_input_opened") is not False or ne.get("live_device_inputs") is not True: r.append("producer encoder receipt: expected input opened or inputs not live")
        for k in ("coverage_errors", "canary_errors", "immutable_errors"):
            if ne.get(k) != 0: r.append(f"producer encoder receipt {k}={ne.get(k)}")
        outs = ne.get("outputs") or {}
        for rel, _, n in PORTS[:4] + [ENTRY]:
            o = outs.get(os.path.basename(rel))
            if not o or o.get("bytes") != n or not o.get("sha256"): r.append(f"producer receipt lacks a sealed {n}-byte hash for {os.path.basename(rel)}")
            else: sealed[rel] = o["sha256"]
    if b0:
        if str(b0.get("result")) != "pass": r.append(f"producer Block0 receipt result {b0.get('result')!r}")
        if b0.get("input_sha256") != case_tex_sha: r.append("producer Block0 receipt input_sha256 is not the case input texture")
        if b0.get("weight_sha256") != PINNED_BLOCK0_WEIGHT_SHA: r.append("producer Block0 receipt weight_sha256 is not the pinned model weight record")
        if not b0.get("skip_sha256"): r.append("producer Block0 receipt lacks skip_sha256")
        else: sealed["block0/skip.e4m3"] = b0["skip_sha256"]
    return sealed
def decide(rec, chain, returncode, case_tex, ref_frame, encoder_run, key, tex_bytes_expected):
    """returns (result, reasons): result is "pass" only when every condition holds."""
    r = []
    # runner exit codes: 0 = receipt, no failures; 1 = receipt written, failures listed (acceptable ONLY if every listed failure is an exempt pinned-frame
    # comparison, checked below); anything else = execution error / abort
    if returncode not in (0, 1): r.append(f"runner exit code {returncode} (execution error)")
    if not rec: r.append(f"no receipt (resident.json missing or unreadable), runner exit code {returncode}"); return "fail", r
    if returncode == 0 and rec.get("failures"): r.append("runner exit 0 but the receipt lists failures (inconsistent)")
    if returncode == 1 and not rec.get("failures"): r.append("runner exit 1 but the receipt lists no failures (inconsistent)")
    if rec.get("schema") != "neuralstudio.nr.body-resident.v1": r.append(f"unexpected receipt schema {rec.get('schema')}")
    if rec.get("key") != key: r.append(f"receipt key {rec.get('key')} != {key}")
    if not isinstance(rec.get("checks"), int) or rec["checks"] <= 0: r.append("receipt reports no checks")
    fails = rec.get("failures") or []; kept = [f for f in fails if not is_exempt(f)]
    if kept: r.append("retained body failures: " + "; ".join(kept[:6]))
    if rec.get("result") not in ("pass", "fail"): r.append(f"receipt result {rec.get('result')!r}")
    if rec.get("result") == "fail" and not fails: r.append("receipt says fail without listing failures")
    if rec.get("decoder_pre_image_matches_strict") is None: r.append("restage check absent")
    L = rec.get("decoder_launches"); L = ast.literal_eval(L) if isinstance(L, str) else (L or [])
    nums = sorted(l.get("launch") for l in L)
    if nums != list(range(133, 157)): r.append(f"decoder launches incomplete: {nums[:3]}..{nums[-3:] if nums else None}")
    fin = next((l for l in L if l.get("launch") == 156), None)
    if not fin: r.append("launch 156 (final surface) absent"); return "fail", r
    t = fin.get("texture") or {}
    if not (t.get("pixels") == tex_bytes_expected and t.get("pixels_once") == t.get("pixels") and t.get("pixels_multi") == 0): r.append(f"final surface write coverage: {t}")
    if fin.get("final_surface_mismatching_values") != 0: r.append(f"final surface mismatching values {fin.get('final_surface_mismatching_values')}")
    if not ref_frame or not os.path.exists(ref_frame): r.append("case RTX reference frame missing"); return "fail", r
    if os.path.getsize(ref_frame) != tex_bytes_expected * 8: r.append(f"reference frame size {os.path.getsize(ref_frame)} != {tex_bytes_expected * 8}")
    ref_sha = sha_file(ref_frame)
    if fin.get("final_surface_sha256") != ref_sha: r.append("final surface sha256 != case RTX output sha256")
    if t.get("output_sha256") != fin.get("final_surface_sha256"): r.append("receipt texture sha != final surface sha (inconsistent receipt)")
    # provenance: the descriptor must bind THIS case's texture and exactly the six sealed ports of THIS producer run (names, offsets, lengths, hashes)
    D = (chain or {}).get("decoder") or {}; root = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); run = os.path.realpath(encoder_run)
    if not case_tex or not os.path.exists(case_tex): r.append("case input texture missing"); return "fail", r
    if os.path.realpath(D.get("tex_in", "")) != os.path.realpath(case_tex): r.append(f"descriptor tex_in is not the case texture: {D.get('tex_in')}")
    if D.get("checks") != "final": r.append(f"descriptor checks mode {D.get('checks')!r} (expected production 'final')")
    sealed = producer_checks(run, sha_file(case_tex), r)
    pats = D.get("patches") or []
    if len(pats) != len(PORTS): r.append(f"descriptor has {len(pats)} patches, expected {len(PORTS)}")
    for P, (rel, off, n) in zip(pats, PORTS):
        want = os.path.join(run, rel); got = _real(P.get("file", ""), root)
        if got != want: r.append(f"patch {rel}: descriptor names {P.get('file')} instead of the producer's file")
        if P.get("offset") != off or P.get("bytes") != n: r.append(f"patch {rel}: offset/bytes {P.get('offset')}/{P.get('bytes')} != {off}/{n}")
        if not os.path.exists(want): r.append(f"producer file missing: {rel}"); continue
        if os.path.getsize(want) != n: r.append(f"producer file {rel} is {os.path.getsize(want)} bytes, expected {n}")
        h = sha_file(want)
        if rel not in sealed: r.append(f"no sealed hash for {rel}")
        elif h != sealed[rel]: r.append(f"producer file {rel} hash != sealed receipt hash")
    ef = (chain or {}).get("entry_file", ""); want = os.path.join(run, ENTRY[0])
    if _real(ef, root) != want: r.append(f"entry view: descriptor names {ef} instead of the producer's {ENTRY[0]}")
    elif not os.path.exists(want) or os.path.getsize(want) != ENTRY[2]: r.append("producer entry view missing or wrong size")
    elif sealed.get(ENTRY[0]) != sha_file(want): r.append("producer entry view hash != sealed receipt hash")
    if (chain or {}).get("entry_offset") not in (None, ENTRY[1]): r.append(f"entry offset {(chain or {}).get('entry_offset')} != {ENTRY[1]}")
    return ("pass" if not r else "fail"), r
