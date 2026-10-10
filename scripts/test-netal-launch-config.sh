#!/bin/zsh
# Loads the real Unix helper into a synthetic host, never into a game.
set -euo pipefail
repository="$(cd "$(dirname "$0")/.." && pwd)"
out="$repository/build/launch-config-tests"
mkdir -p "$out/original" "$out/relocated package"
flags=(-arch x86_64 -std=c++20 -O2 -Wall -Wextra -Werror)
xcrun clang++ "${flags[@]}" -dynamiclib -Wl,-install_name,@rpath/netal_wine_host_v1.so "$repository/src/macos_host/re9_wine_unixlib.cpp" -o "$out/original/helper.so"
xcrun clang++ "${flags[@]}" -dynamiclib "$repository/src/macos_host/netal_launch_fixture.cpp" -o "$out/original/fixture.dylib"
xcrun clang++ "${flags[@]}" "$repository/src/macos_host/netal_launch_config_test.cpp" -o "$out/test-host"
codesign --force --sign - "$out/original/helper.so" "$out/original/fixture.dylib" "$out/test-host" >/dev/null
"$out/test-host" "$out/original/helper.so" "$out/original/fixture.dylib"
cp "$out/original/helper.so" "$out/relocated package/helper.so"
cp "$out/original/fixture.dylib" "$out/relocated package/fixture.dylib"
"$out/test-host" "$out/relocated package/helper.so" "$out/relocated package/fixture.dylib"
python3 - "$out/relocated package/helper.so" "$repository" <<'PY'
from pathlib import Path
import sys
assert sys.argv[2].encode() not in Path(sys.argv[1]).read_bytes(), 'build checkout path embedded in helper'
print('relocated helper contains no build checkout path')
PY
