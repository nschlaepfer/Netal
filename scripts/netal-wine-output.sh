#!/bin/zsh
# Wine console handles can outlive the command. Spool output so a background Wine
# process cannot keep the app's command-substitution/result pipes open indefinitely.
netal_wine_capture() {
  local capture_dir result
  capture_dir="$(mktemp -d "${TMPDIR:-/tmp}/netal-wine.XXXXXXXX")" || return 1
  if "$@" >"$capture_dir/stdout" 2>"$capture_dir/stderr"; then result=0; else result=$?; fi
  # Bound what is forwarded to the app. The command has exited; bridge readiness
  # and inference status are still verified separately by the caller.
  tail -c 65536 "$capture_dir/stdout"
  if (( result != 0 )); then tail -c 16384 "$capture_dir/stderr" >&2; fi
  rm -f "$capture_dir/stdout" "$capture_dir/stderr"
  rmdir "$capture_dir"
  return "$result"
}
