#!/usr/bin/env python3
"""Build the local SR model-capture helpers from source, without a research checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IMAGE = 'ubuntu@sha256:33ceb71981b602c1a7443a53469e4dba065f7503eab3078a2d7a57a2ab987517'
FILES = ['dlss-sr-cuda-capture.exe', 'nvcuda.dll', 'nvapi64.dll', 'nvngx.dll', 'nvcuda-forward-trace.dll']
BUILD = r'''
set -eu
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq
apt-get install -y -qq --no-install-recommends gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 >/dev/null
for pair in dlss_sr_nvcuda_trace:nvcuda nvapi_arch_trace:nvapi64 nvngx_cuda_thunk:nvngx; do
    source=${pair%:*}; output=${pair#*:}
    x86_64-w64-mingw32-gcc -std=c11 -O2 -Wall -Wextra -Werror -shared \
        -fno-optimize-sibling-calls -static-libgcc "/src/$source.c" -o "/out/$output.dll"
done
x86_64-w64-mingw32-g++ -std=c++20 -O2 -Wall -Wextra -Werror -municode \
    -static-libgcc -static-libstdc++ -I/ngx/include /src/dlss_sr_cuda_capture.cpp \
    -o /out/dlss-sr-cuda-capture.exe
x86_64-w64-mingw32-g++ -std=c++20 -O2 -Wall -Wextra -Werror -shared \
    -static-libgcc -static-libstdc++ /src/dlss_sr_nvcuda_forward_trace.cpp \
    -o /out/nvcuda-forward-trace.dll
x86_64-w64-mingw32-g++ --version > /out/compiler.txt
dpkg-query -W gcc-mingw-w64-x86-64 g++-mingw-w64-x86-64 mingw-w64-common > /out/packages.txt
'''


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--headers', required=True, type=Path, help='locally supplied NVIDIA DLSS SDK root, containing include/')
    parser.add_argument('--output', type=Path, default=ROOT / 'build/model-capture')
    parser.add_argument('--image', default=DEFAULT_IMAGE, help='Docker builder image; receipt records resolved image identity')
    args = parser.parse_args()
    headers = args.headers.resolve()
    output = args.output.absolute()
    if output.exists():
        parser.error('output already exists; use a new directory to preserve the existing tool set')
    if not (headers / 'include/nvsdk_ngx.h').is_file() or not (headers / 'include/nvsdk_ngx_params.h').is_file():
        parser.error('--headers must contain include/nvsdk_ngx.h and include/nvsdk_ngx_params.h')
    source = ROOT / 'src/model_capture'
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(['docker', 'info', '--format', '{{.ServerVersion}}'], check=True)
        with tempfile.TemporaryDirectory(prefix='.capture-build-', dir=output.parent) as temporary:
            stage = Path(temporary) / 'out'
            stage.mkdir()
            subprocess.run(['docker', 'run', '--rm', '--platform', 'linux/amd64',
                            '--mount', f'type=bind,source={source},target=/src,readonly',
                            '--mount', f'type=bind,source={headers},target=/ngx,readonly',
                            '--mount', f'type=bind,source={stage},target=/out',
                            args.image, 'sh', '-c', BUILD], check=True)
            image = json.loads(subprocess.check_output(['docker', 'image', 'inspect', args.image], text=True))[0]
            for name in FILES:
                if not (stage / name).is_file():
                    raise RuntimeError('build did not produce ' + name)
            receipt = {
                'schema': 'netal.capture-build.v1',
                'image_id': image['Id'], 'image_digests': image.get('RepoDigests', []),
                'platform': 'linux/amd64',
                'source': {p.name: sha(p) for p in sorted(source.iterdir()) if p.suffix in {'.c', '.cpp'}},
                'headers': {p.relative_to(headers).as_posix(): sha(p) for p in sorted((headers / 'include').rglob('*.h'))},
                'outputs': {name: {'sha256': sha(stage / name), 'bytes': (stage / name).stat().st_size} for name in FILES},
                'compiler': (stage / 'compiler.txt').read_text(),
                'packages': (stage / 'packages.txt').read_text(),
                'models_included': False,
            }
            (stage / 'build-receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
            stage.rename(output)
    except (OSError, subprocess.CalledProcessError, RuntimeError) as error:
        print(f'build failed: {error}', file=sys.stderr)
        return 1
    print(f'result=pass output={output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
