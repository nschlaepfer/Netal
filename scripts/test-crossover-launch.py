#!/usr/bin/env python3
"""Test launcher routing in an isolated fake runtime; never starts or kills a real game."""
from pathlib import Path
import shlex
import os
import shutil
import subprocess
import tempfile

repo = Path(__file__).resolve().parents[1]
with tempfile.TemporaryDirectory(prefix="Netal launcher ' ") as directory:
    root = Path(directory)
    scripts = root / 'scripts'
    scripts.mkdir()
    commands = root / 'bin'
    commands.mkdir()
    shutil.copy2(repo / 'scripts/crossover-exact-sr.sh', scripts)
    shutil.copy2(repo / 'scripts/mac-dlss-attach.sh', scripts)
    shutil.copy2(repo / 'scripts/netal-wine-output.sh', scripts)
    helper_dir = root / 'build/netal-tools'; helper_dir.mkdir(parents=True)
    binary_dir = root / 'build/re9_crossover'; binary_dir.mkdir()
    for name in ('neuralstudio_re9_inject.exe','netal_wine_host_v1.dll','netal_wine_host_v1.so','libNeuralStudioRe9MetalFXNoop.dylib'):
        (binary_dir/name).write_text('fixture')
    for kind in ('sr','fg'):
        bundle = root / f'work/re9-{kind}-runtime-bundle'; bundle.mkdir(parents=True); (bundle/'manifest.json').write_text('{}')
    helper = helper_dir / 'netal-launch-config'
    helper.write_text('''#!/bin/zsh
print "log=${0:h}/../re9_crossover/re9-live-metalfx.log"
if [[ "${NETAL_TEST_STALE:-0}" == 1 && -f "${0:h}/seen" ]]; then print 'start_sec=2'; else print 'start_sec=1'; fi
touch "${0:h}/seen"
''')
    helper.chmod(0o755)
    injector = scripts / 'inject-re9-metalfx-noop.sh'
    injector.write_text('''#!/bin/zsh
print -rl -- "$@" > "${0:h}/../arguments.txt"
print 'event=bridge_ready' > "${0:h}/../build/re9_crossover/re9-live-metalfx.log"
''')
    for name, body in {
        'pgrep': "print 12345",
        'ps': "print 'C:\\\\Games\\\\Example\\\\other-game.exe'",
        'lsof': 'exit 1',
        'vmmap': "print libNeuralStudioRe9MetalFXNoop",
        'sleep': 'exit 0',
    }.items():
        path = commands / name
        path.write_text('#!/bin/zsh\n' + body + '\n')
        path.chmod(0o755)
    cx = root/'CrossOver fixture'
    (cx/'bin').mkdir(parents=True)
    wine = cx/'bin/wine'
    wine.write_text('''#!/bin/zsh
[[ "$1" == --bottle && "$2" == 'Other Bottle' ]] || exit 93
if [[ "$4" == --ux-app ]]; then
  if [[ "$6" == WINEPREFIX ]]; then print -r -- "$NETAL_TEST_ROOT/bottle"; else print '/fixture/system/windows:/fixture/system/unix'; fi
  exit 0
fi
operation="$5"; target="$6"
if [[ "$target" == steam.exe ]]; then
  [[ "${NETAL_TEST_STEAM:-absent}" != absent ]] || exit 2
  [[ "$operation" != --query || "${NETAL_TEST_STEAM:-absent}" != unprepared ]] || exit 8
else
  [[ "${NETAL_TEST_AMBIGUOUS:-0}" != 1 ]] || exit 9
  [[ "${NETAL_TEST_GAME:-running}" != absent || -f "$NETAL_TEST_ROOT/started" ]] || exit 2
  [[ "$operation" != --query || "${NETAL_TEST_QUERY_FAILURE:-0}" != 1 ]] || exit 8
fi
print 'windows_pid=208'
if [[ "$operation" == --query ]]; then print "mac_pid=${NETAL_TEST_PID:-12345}"; fi
''')
    wine.chmod(0o755)
    start = cx/'bin/cxstart'
    start.write_text('''#!/bin/zsh
print -rl -- "$@" > "$NETAL_TEST_ROOT/start-arguments.txt"
touch "$NETAL_TEST_ROOT/started"
''')
    start.chmod(0o755)
    env = dict(os.environ, PATH=str(commands) + os.pathsep + os.environ['PATH'],
               NETAL_CROSSOVER_ROOT=str(cx), NETAL_TEST_ROOT=str(root))
    def run(script, arguments, stale=False, **settings):
        (helper_dir/'seen').unlink(missing_ok=True)
        (root/'arguments.txt').unlink(missing_ok=True)
        (root/'started').unlink(missing_ok=True)
        (root/'start-arguments.txt').unlink(missing_ok=True)
        return subprocess.run(['/bin/zsh', str(scripts/script), *arguments], cwd=root,
                              env=dict(env, NETAL_TEST_STALE=str(int(stale)), **settings),
                              text=True, capture_output=True, timeout=15)

    launch = ['--bottle', 'Other Bottle', '--appid', '123', '--exe', 'other-game.exe', '--sustained', '--mode', 'hw']
    result = run('crossover-exact-sr.sh', [*launch, '--fg'])
    assert result.returncode == 0, result.stderr
    summary = next(line.removeprefix('summary: ') for line in result.stdout.splitlines() if line.startswith('summary: '))
    summary_args = shlex.split(summary)
    assert len(summary_args) == 4 and summary_args[:1] == ['python3'] and summary_args[2] == '--log', summary_args
    assert Path(summary_args[3]).resolve() == (binary_dir/'re9-live-metalfx.log').resolve()
    args = (root / 'arguments.txt').read_text().splitlines()
    assert '--bottle=Other Bottle' in args, args
    assert '--target-exe=other-game.exe' in args, args
    assert '--pid=12345' in args, args
    assert '--arm-fg' in args and '--sustained' in args and '--portable' in args, args
    assert (root / 'build/re9_crossover/re9-live-metalfx.log').exists()
    result = run('crossover-exact-sr.sh', launch, stale=True)
    assert result.returncode == 5 and 'Target process changed' in result.stderr, result
    result = run('crossover-exact-sr.sh', launch, NETAL_TEST_GAME='absent')
    assert result.returncode == 0, result.stderr
    startup = (root/'start-arguments.txt').read_text().splitlines()
    launch_env = startup[startup.index('--env')+1]
    assert shlex.split(launch_env) == [f'WINEDLLPATH={binary_dir}:/fixture/system/windows:/fixture/system/unix'], startup
    result = run('crossover-exact-sr.sh', launch, NETAL_TEST_STEAM='prepared')
    assert result.returncode == 0, result.stderr
    with subprocess.Popen(['/bin/sleep', '20']) as owned_fixture:
        try:
            result = run('crossover-exact-sr.sh', [*launch, '--relaunch'], NETAL_TEST_STEAM='unprepared', NETAL_TEST_PID=str(owned_fixture.pid))
            assert result.returncode == 4 and 'Quit Steam in bottle' in result.stderr, result
            assert owned_fixture.poll() is None and not (root/'arguments.txt').exists() and not (root/'started').exists()
        finally:
            if owned_fixture.poll() is None: owned_fixture.terminate()
            owned_fixture.wait()
    result = run('crossover-exact-sr.sh', launch, NETAL_TEST_QUERY_FAILURE='1')
    assert result.returncode == 4 and not (root/'arguments.txt').exists(), result
    result = run('crossover-exact-sr.sh', launch, NETAL_TEST_AMBIGUOUS='1')
    assert result.returncode == 9 and not (root/'arguments.txt').exists(), result
    fg_manifest = root/'work/re9-fg-runtime-bundle/manifest.json'
    fg_manifest.unlink()
    result = run('crossover-exact-sr.sh', launch)
    assert result.returncode == 0, result.stderr
    assert '--arm-fg' not in (root/'arguments.txt').read_text().splitlines()
    result = run('crossover-exact-sr.sh', [*launch, '--fg'])
    assert result.returncode == 2 and not (root/'arguments.txt').exists(), result
    (binary_dir/'netal_wine_host_v1.so').unlink()
    result = run('crossover-exact-sr.sh', launch)
    assert result.returncode == 2 and not (root/'arguments.txt').exists(), result
    result = run('mac-dlss-attach.sh', ['--exe', 'Test Game.exe', '--bottle', 'Other Bottle'])
    assert result.returncode == 0, result.stderr
    assert '--target-exe=Test Game.exe' in (root/'arguments.txt').read_text().splitlines()
    result = run('mac-dlss-attach.sh', ['--exe', 'Test Game.exe', '--bottle', 'Other Bottle'], stale=True)
    assert result.returncode == 5 and 'Target process changed' in result.stderr, result
    result = run('mac-dlss-attach.sh', [])
    assert result.returncode == 2 and not (root/'arguments.txt').exists(), result
print('13 CrossOver launcher/attach scenarios passed (simulated, no game run)')
