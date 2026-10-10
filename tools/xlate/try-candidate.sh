#!/bin/zsh
# One candidate, one verdict: null control -> exactness -> interleaved timing -> receipt or ledger.
#
#   tools/xlate/try-candidate.sh tools/xlate/units/<unit>.json <candidate.metal|.metallib> <label>
#
# The order is the point and it is not skippable. See tools/xlate/README.md for the three failures
# that put each step here.
set -euo pipefail
repo="$(cd "$(dirname "$0")/../.." && pwd)"
work=/private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/xlate
mkdir -p "$work"

[[ $# -ge 3 ]] || { print -u2 "usage: try-candidate.sh <unit.json> <candidate> <label>"; exit 2 }
unit="$1"; candidate="$2"; label="$3"; rounds="${4:-3}"
[[ -f "$unit" ]] || { print -u2 "no such unit manifest: $unit"; exit 2 }

field() { python3 -c "import json,sys; d=json.load(open('$unit')); v=d.get('$1',''); print(v if not isinstance(v,list) else ' '.join(map(str,v)))"; }
bundle_src="$repo/$(field bundle)"; artifact="$(field artifact)"; probe="$repo/$(field probe)"
capture="$repo/$(field capture)"; source_path="$(field source)"; entry="$(field entry)"
oracle=(${(s: :)$(field oracle)}); repeat_env="$(field repeat_env)"; repeat="$(field repeat)"
compile_flags=(${(s: :)$(field compile)})

env_pairs() { python3 -c "import json; d=json.load(open('$unit')); print(' '.join(f'{k}={v}' for k,v in d.get('env',{}).items()))"; }
probe_env=(${(s: :)$(env_pairs)})

compile_metal() {   # <src> <out.metallib>
  xcrun -sdk macosx metal ${compile_flags[@]} -c "$1" -o "$work/$3.air" 2>/dev/null
  xcrun -sdk macosx metallib "$work/$3.air" -o "$2"
}

stage() {           # <metallib> <label> -> prints bundle dir
  local b="$work/bundle-$2"; rm -rf "$b"; cp -R "$bundle_src" "$b"; cp "$1" "$b/$artifact"
  # The runtime hash-checks the manifest, which is the point of the manifest. Restate this one row
  # rather than bypassing the check for the other files.
  python3 - "$b" "$artifact" <<'PY'
import hashlib, json, pathlib, sys
b, target = pathlib.Path(sys.argv[1]), sys.argv[2]
p = b / "manifest.json"
if p.exists():
    m = json.loads(p.read_text()); data = (b / target).read_bytes()
    for row in m.get("files", []):
        if row.get("path") == target:
            row["sha256"] = hashlib.sha256(data).hexdigest(); row["bytes"] = len(data); break
    p.write_text(json.dumps(m, indent=2, sort_keys=True) + "\n")
PY
  print "$b"
}

run() {             # <bundle> <tag> [extra env...] -> "sha1,sha2,... min med"
  local b="$1" tag="$2"; shift 2
  local r="$work/receipt-$tag.txt"; rm -f "$r"
  env ${probe_env[@]} "$@" "$probe" "$b" "$capture" "$r" > "$work/log-$tag.txt" 2>&1 || true
  [[ -s "$r" ]] || { print "NORECEIPT"; return }
  local shas ms
  shas=$(grep output_sha "$r" | cut -d= -f2 | cut -c1-8 | tr '\n' ',')
  ms=$(grep gpu_ms "$r" | cut -d= -f2 | sort -n | awk '{a[NR]=$1} END {printf "%.2f %.2f", a[1], a[int((NR+1)/2)]}')
  print "$shas $ms"
}

exact() {           # <sha csv> -> 0 if every oracle hash matches in order
  local got=(${(s:,:)1}) i
  for i in {1..${#oracle}}; do [[ "${got[$i]:-}" == "${oracle[$i]}" ]] || return 1; done
  return 0
}

print "unit      $(basename $unit)"
print "candidate $candidate"

# --- 1. null control: prove the harness is pointed at code that actually runs -------------------
if [[ -n "$source_path" && -f "$repo/$source_path" && -n "$entry" ]]; then
  python3 - "$repo/$source_path" "$entry" "$work/null.metal" <<'PY'
import io, sys, re
src, entry, out = sys.argv[1], sys.argv[2], sys.argv[3]
lines = io.open(src, encoding="utf-8").read().split("\n")
start = next(i for i, l in enumerate(lines) if re.search(r"kernel\s+void\s+" + re.escape(entry) + r"\b", l))
brace = next(i for i in range(start, len(lines)) if lines[i].rstrip().endswith(") {"))
lines.insert(brace + 1, "    if (true) return;   // XLATE NULL CONTROL")
io.open(out, "w", encoding="utf-8").write("\n".join(lines))
PY
  compile_metal "$work/null.metal" "$work/null.metallib" null
  nb=$(stage "$work/null.metallib" "null-$label")
  nr=($(run "$nb" "null-$label"))
  if [[ "${nr[1]}" == "NORECEIPT" ]] || exact "${nr[1]}"; then
    print "\n  NULL CONTROL FAILED — an empty kernel produced the accepted receipts."
    print "  The harness is not measuring this artifact. Nothing below would mean anything."
    exit 3
  fi
  print "null      receipts move (${nr[1]%%,*}...) — harness is live"
else
  print "null      skipped (manifest has no source/entry; exactness still enforced)"
fi

# --- 2. exactness -------------------------------------------------------------------------------
cand="$candidate"
[[ "$cand" == *.metal ]] && { compile_metal "$cand" "$work/$label.metallib" "$label"; cand="$work/$label.metallib" }
cb=$(stage "$cand" "$label")
cr=($(run "$cb" "$label"))
if [[ "${cr[1]}" == "NORECEIPT" ]]; then print "exact     NO RECEIPT"; tail -3 "$work/log-$label.txt"; exit 4; fi
if ! exact "${cr[1]}"; then
  print "exact     DIVERGED  ${cr[1]}"
  ledger="$(dirname $unit)/$(basename $unit .json)-ledger.md"
  print -- "\n- **$label** ($(date -u +%Y-%m-%dT%H:%M:%SZ)) — diverged: \`${cr[1]}\`. Timing not taken; a fast wrong kernel is not a result." >> "$ledger"
  print "ledger    recorded in $ledger"
  exit 5
fi
print "exact     yes"

# --- 3. interleaved timing against a measured noise floor ---------------------------------------
ib=$(stage "$bundle_src/$artifact" "base-$label")
typeset -a base_ms cand_ms
repeat_arg=()
[[ -n "$repeat_env" && -n "$repeat" ]] && repeat_arg=("$repeat_env=$repeat")
for i in $(seq 1 $rounds); do
  b=($(run "$ib" "base-$label-$i" ${repeat_arg[@]})); base_ms+=(${b[2]})
  c=($(run "$cb" "$label-$i" ${repeat_arg[@]}));      cand_ms+=(${c[2]})
done
python3 - "$label" "${(j:,:)base_ms}" "${(j:,:)cand_ms}" "${repeat:-1}" "$unit" <<'PY'
import sys, statistics, datetime, os
label, base, cand, repeat, unit = sys.argv[1], [float(x) for x in sys.argv[2].split(",")], \
    [float(x) for x in sys.argv[3].split(",")], int(sys.argv[4] or 1), sys.argv[5]
spread = max(base) - min(base)
delta = (min(base) - min(cand)) / repeat
noise = spread / repeat
print(f"timing    incumbent min {min(base):.2f}  candidate min {min(cand):.2f}  (repeat={repeat})")
print(f"noise     incumbent spread over {len(base)} rounds = {spread:.2f} ms ({noise:.3f} ms per dispatch)")
wins = sum(1 for b, c in zip(base, cand) if c < b)
if abs(delta) < noise:
    verdict = f"INCONCLUSIVE — {delta:+.3f} ms/dispatch is inside the {noise:.3f} ms noise floor"
elif delta > 0 and wins == len(base):
    verdict = f"ACCEPT — {delta:+.3f} ms/dispatch, exact, and won all {wins} pairs"
else:
    verdict = f"REJECT — {delta:+.3f} ms/dispatch, won {wins} of {len(base)} pairs"
print("verdict   " + verdict)
if not verdict.startswith("ACCEPT"):
    ledger = os.path.join(os.path.dirname(unit), os.path.basename(unit)[:-5] + "-ledger.md")
    with open(ledger, "a") as f:
        f.write(f"\n- **{label}** ({datetime.datetime.now(datetime.timezone.utc):%Y-%m-%dT%H:%M:%SZ}) — exact, {verdict.split(' — ')[1]}.\n")
    print(f"ledger    recorded in {ledger}")
PY
