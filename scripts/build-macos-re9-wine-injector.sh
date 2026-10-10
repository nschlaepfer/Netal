#!/bin/zsh
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repository="$(cd "$script_dir/.." && pwd)"
build_dir="$repository/build/re9_crossover"
llvm_dir="/opt/homebrew/opt/llvm/bin"
bridge="$build_dir/libNeuralStudioRe9MetalFXNoop.dylib"
hook_dll="$build_dir/netal_wine_host_v1.dll"
hook_so="$build_dir/netal_wine_host_v1.so"
controller="$build_dir/neuralstudio_re9_inject.exe"

scope=bridge-and-wine-helpers
if [[ $# -eq 1 && "$1" == --helpers-only ]]; then
  scope=wine-helpers-only
elif [[ $# -ne 0 ]]; then
  print -u2 'usage: build-macos-re9-wine-injector.sh [--helpers-only]'; exit 64
fi
if [[ "$scope" == bridge-and-wine-helpers ]]; then
  zsh "$script_dir/build-macos-re9-metalfx-noop-bridge.sh" >/dev/null
fi
mkdir -p "$build_dir"

"$llvm_dir/llvm-dlltool" -m i386:x86-64 \
  -d "$repository/src/macos_host/re9_wine_kernel32.def" \
  -l "$build_dir/re9_kernel32.lib"
"$llvm_dir/llvm-dlltool" -m i386:x86-64 \
  -d "$repository/src/macos_host/re9_wine_ntdll.def" \
  -l "$build_dir/re9_ntdll.lib"

"$llvm_dir/clang" --target=x86_64-pc-windows-msvc -c -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -Werror \
  "$repository/src/macos_host/re9_wine_inject_controller.c" \
  -o "$build_dir/re9_wine_inject_controller.obj"
"$llvm_dir/clang" --target=x86_64-pc-windows-msvc -c -ffreestanding -fno-builtin \
  -fno-stack-protector -Wall -Wextra -Werror \
  "$repository/src/macos_host/re9_wine_injector_dll.c" \
  -o "$build_dir/re9_wine_injector_dll.obj"

/opt/homebrew/bin/lld-link /nologo /nodefaultlib /machine:x64 /subsystem:console \
  /entry:mainCRTStartup /out:"$controller" \
  "$build_dir/re9_wine_inject_controller.obj" "$build_dir/re9_kernel32.lib"
/opt/homebrew/bin/lld-link /nologo /nodefaultlib /machine:x64 /subsystem:windows /dll \
  /entry:DllMain /out:"$hook_dll" \
  "$build_dir/re9_wine_injector_dll.obj" "$build_dir/re9_kernel32.lib" \
  "$build_dir/re9_ntdll.lib"

# Wine recognizes modules found through WINEDLLPATH by this 32-byte DOS-stub marker.
# The linked PE has room for it between the 64-byte DOS header and e_lfanew.
dd if=/dev/zero of="$hook_dll" bs=1 seek=64 count=32 conv=notrunc status=none
print -n 'Wine builtin DLL' | dd of="$hook_dll" bs=1 seek=64 conv=notrunc status=none

xcrun --sdk macosx clang++ -arch x86_64 -dynamiclib -std=c++20 \
  -O2 -Wall -Wextra -Werror -mmacosx-version-min=26.0 \
  -Wl,-install_name,@rpath/netal_wine_host_v1.so \
  "$repository/src/macos_host/re9_wine_unixlib.cpp" -o "$hook_so"
codesign --force --sign - "$hook_so" >/dev/null
mkdir -p "$repository/build/netal-tools"
xcrun --sdk macosx clang++ -std=c++20 -O2 -Wall -Wextra -Werror \
  "$repository/src/macos_host/netal_launch_config_tool.cpp" \
  -o "$repository/build/netal-tools/netal-launch-config"
codesign --force --sign - "$repository/build/netal-tools/netal-launch-config" >/dev/null

file "$controller" "$hook_dll" "$hook_so"
strings "$hook_dll" | rg -q '^Wine builtin DLL$'
shasum -a 256 "$controller" "$hook_dll" "$hook_so" "$repository/build/netal-tools/netal-launch-config"
if [[ "$scope" == bridge-and-wine-helpers ]]; then shasum -a 256 "$bridge"; fi
print "scope=$scope"
print "result=pass"
