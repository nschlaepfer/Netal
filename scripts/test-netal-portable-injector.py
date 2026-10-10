#!/usr/bin/env python3
"""Exercise the production portable-injector script with isolated command fixtures."""
from pathlib import Path
import os
import shutil
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix='Netal portable ') as directory:
    root = Path(directory)
    (root/'scripts').mkdir()
    shutil.copy2(repo/'scripts/inject-re9-metalfx-noop.sh', root/'scripts')
    shutil.copy2(repo/'scripts/netal-wine-output.sh', root/'scripts')
    binary = root/'build/re9_crossover'
    binary.mkdir(parents=True)
    for name in ('neuralstudio_re9_inject.exe', 'netal_wine_host_v1.dll', 'netal_wine_host_v1.so', 'libNeuralStudioRe9MetalFXNoop.dylib'):
        (binary/name).write_text('fixture')
    commands = root/'commands'
    commands.mkdir()
    cx = root/'CrossOver fixture'
    (cx/'bin').mkdir(parents=True)
    wine = cx/'bin/wine'
    wine.write_text('''#!/bin/zsh
if [[ "$4" == --ux-app ]]; then print -r -- "$NETAL_TEST_ROOT/bottle"; else print "mac_pid=${NETAL_TEST_QUERY_PID:-$NETAL_TEST_PID}"; fi
''')
    wine.chmod(0o755)
    tool = root/'build/netal-tools/netal-launch-config'
    tool.parent.mkdir()
    tool.write_text('''#!/bin/zsh
print -r -- "session_dir=$NETAL_TEST_ROOT/build/re9_crossover"
print -r -- "log=$NETAL_TEST_ROOT/build/re9_crossover/re9-live-metalfx.log"
''')
    tool.chmod(0o755)
    for name, body in {'pgrep': 'print "$NETAL_TEST_PID"', 'defaults': 'exit 0',
                       'cp': 'touch "$NETAL_TEST_ROOT/unexpected-copy"; exit 99'}.items():
        file = commands/name
        file.write_text('#!/bin/zsh\n'+body+'\n')
        file.chmod(0o755)
    controller = cx/'bin/cxstart'
    controller.write_text('''#!/bin/zsh
print -rl -- "$@" > "$NETAL_TEST_ROOT/controller-arguments.txt"
print 'event=bridge_loaded detail=fixture' > "$NETAL_TEST_ROOT/build/re9_crossover/re9-live-unixlib.log"
print 'event=bridge_ready detail=fixture' > "$NETAL_TEST_ROOT/build/re9_crossover/re9-live-metalfx.log"
''')
    controller.chmod(0o755)
    environment = dict(os.environ, PATH=str(commands)+os.pathsep+os.environ['PATH'],
                       NETAL_CROSSOVER_ROOT=str(cx), NETAL_TEST_ROOT=str(root), NETAL_TEST_PID=str(os.getpid()))
    arguments = ['--portable', f'--pid={os.getpid()}', '--target-exe=Test Game.exe', '--bottle=Netal Isolated Fixture']
    result = subprocess.run(['zsh', str(root/'scripts/inject-re9-metalfx-noop.sh'), *arguments],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 0 and 'result=pass' in result.stdout, result
    passed = (root/'controller-arguments.txt').read_text().splitlines()
    assert passed[-2:] == ['Test Game.exe', 'Z:'+str(binary/'netal_wine_host_v1.dll').replace('/', '\\')], passed
    assert not (root/'unexpected-copy').exists()
    assert not (cx/'lib').exists()
    (root/'controller-arguments.txt').unlink(missing_ok=True)
    result = subprocess.run(['zsh', str(root/'scripts/inject-re9-metalfx-noop.sh'), *arguments],
                            env=dict(environment, NETAL_TEST_QUERY_PID=str(os.getpid()+100000)),
                            capture_output=True, text=True, timeout=15)
    assert result.returncode == 4 and 'does not match' in result.stderr, result
    assert not (root/'controller-arguments.txt').exists()
    # A pre-existing shared hook must cause refusal, not load another runtime's Unix library.
    conflict = cx/'lib/wine/x86_64-unix/netal_wine_host_v1.so'
    conflict.parent.mkdir(parents=True)
    conflict.write_text('existing runtime')
    (root/'controller-arguments.txt').unlink(missing_ok=True)
    result = subprocess.run(['zsh', str(root/'scripts/inject-re9-metalfx-noop.sh'), *arguments],
                            env=environment, capture_output=True, text=True, timeout=15)
    assert result.returncode == 3 and 'Refusing existing hook' in result.stderr, result
    assert conflict.read_text() == 'existing runtime'
    assert not (root/'controller-arguments.txt').exists() and not (root/'unexpected-copy').exists()
print('Portable injector routing, PID mismatch and conflict refusal passed (simulated)')
