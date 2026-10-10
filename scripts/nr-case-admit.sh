#!/bin/zsh
# One bounded fresh-case body run with storage admission: nr-case-admit.sh CASE_DIR PRODUCER_ROOT KEY ATTEMPT_ROOT
# Admits only if measured free space minus this run's peak (0.3 GiB: measured 0.24 GiB untrimmed + margin) stays >= 12 GiB (Codex agreement 03:50 UTC); journals admission and
# outcome; runs under the atomic GPU lock; trims regenerable stage bulk (pool/texture dumps) afterwards, keeping receipts, final surface hash and logs.
set -u; cd /private/tmp/claude-501/-Users-nico/deb7f1d0-b1d5-4951-9b30-84ff130544fc/scratchpad/nrwt; PY=/opt/homebrew/bin/python3.13
CASE=$1; PROD=$2; KEY=$3; AR=$4; ATT=$AR/$KEY; J=docs/NR_AGENT_COORDINATION.md; PEAK_GIB=0.3; FLOOR_GIB=12   # peak measured on an untrimmed production attempt (char-v4/production-5a-b: 243 MB incl. 210 MB staged decoder constants + 33 MB run outputs); the earlier 2.5 GiB figure was the full fast chain with the encoder driver
FREE=$(df -k /Users/nico | tail -1 | awk '{printf "%.2f", $4/1048576}')
ok=$($PY -c "print(int($FREE - $PEAK_GIB >= $FLOOR_GIB))")
TS=$(date -u +"%Y-%m-%d %H:%M UTC")
if [ "$ok" != "1" ]; then printf '\n### %s — Fable — storage admission REFUSED for case body run %s: free %s GiB - peak %s GiB < %s GiB floor; not started\n' "$TS" "$KEY" "$FREE" "$PEAK_GIB" "$FLOOR_GIB" >> $J; echo "refused: free $FREE"; exit 3; fi
printf '\n### %s — Fable — storage admission for ONE case body run (%s): free %.2f GiB - peak %s GiB = %.2f GiB >= %s GiB floor; producer %s\n' "$TS" "$KEY" "$FREE" "$PEAK_GIB" "$(($FREE - $PEAK_GIB))" "$FLOOR_GIB" "$PROD/$KEY" >> $J
$PY scripts/nr-gpu-lock.py run --owner Fable --note "fresh dense-color case body run $KEY (case3, correctness only, ~4 s GPU)" -- $PY scripts/nr-case-validate.py --attempt $ATT --case-dir $CASE --encoder-run $PROD/$KEY --key $KEY \
  --split-attempt build/nr_split/attempt-v13 --vit-attempt build/nr_vit/attempt-v8 --dec-lib build/nr_ptx/attempt-v15/dec.metallib --post-lib build/nr_ptx/attempt-v15/post.metallib --copy-lib build/nr_ptx/attempt-v15/copy.metallib \
  --probe build/nr_body/v3/nr_body_cli --vit-tp-lib build/nr_vit/tp-attempt-v8/tp.metallib --split-tp-lib build/nr_tp/attempt-v5/tp.metallib; RC=$?
# keep: receipts, logs, the final surface dump (16 MB, the acceptance payload) and out_view; trim regenerable pool/stage bulk
find $ATT -type f \( -name "pool_pre*.bin" -o -name "pool_post*.bin" -o -path "*/stage/*.bin" -o -name "tex_out_155.bin" \) -delete 2>/dev/null
RES=$($PY -c "import json;d=json.load(open('$ATT/case-validate.json'));print(d['result'], 'mismatching', d['final_surface_mismatching_values'], 'reasons', d['gate_reasons'][:3])" 2>/dev/null || echo "no receipt")
TS=$(date -u +"%Y-%m-%d %H:%M UTC"); printf '\n### %s — Fable — case body run %s finished (exit %s): %s; free now %s GiB; receipts in `%s`\n' "$TS" "$KEY" "$RC" "$RES" "$(df -k /Users/nico | tail -1 | awk '{printf "%.2f", $4/1048576}')" "$ATT" >> $J
echo "$KEY exit $RC: $RES"; exit $RC
