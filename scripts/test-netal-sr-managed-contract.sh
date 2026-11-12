#!/bin/sh
# CPU-only; caller can place temporary artifacts in a private TMPDIR.
set -eu
root=$(CDPATH= cd -- "$(dirname "$0")/.." && pwd)
work=$(mktemp -d "${TMPDIR:-/tmp}/netal-managed-contract.XXXXXX")
trap 'rm -rf "$work"' EXIT HUP INT TERM
"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -pedantic "$root/scripts/test-netal-sr-managed-contract.cpp" -o "$work/contract"
"$work/contract"
