#!/usr/bin/env python3
"""Opt-in real CrossOver test in a newly created disposable bottle; no games or models."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import time
import uuid


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path, help='new evidence directory; must not exist')
    parser.add_argument('--crossover-root', type=Path, default=Path('/Applications/CrossOver.app/Contents/SharedSupport/CrossOver'))
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    cx = args.crossover_root.resolve()
    helpers = root/'build/re9_crossover'
    config_tool = root/'build/netal-tools/netal-launch-config'
    required = [helpers/n for n in ('neuralstudio_re9_inject.exe', 'netal_wine_host_v1.dll', 'netal_wine_host_v1.so')] + [config_tool]
    for path in required:
        if not path.is_file():
            raise RuntimeError('Build helpers first with scripts/build-macos-re9-wine-injector.sh --helpers-only')
    protected = [cx/'lib/wine/x86_64-windows/netal_wine_host_v1.dll', cx/'lib/wine/x86_64-unix/netal_wine_host_v1.so']
    if any(path.exists() or path.is_symlink() for path in protected):
        raise RuntimeError('Existing CrossOver hook would confound the test; no files were changed')
    if ':' in str(output):
        raise RuntimeError('The output path cannot contain the Wine search-path separator')
    output.mkdir(parents=True, exist_ok=False)
    runtime = output/'runtime with spaces'
    runtime.mkdir()
    for path in required[:3]:
        shutil.copy2(path, runtime/path.name)
    bottle = output/'bottles/NetalHelperTest'
    bottle.parent.mkdir()
    env = dict(os.environ, CX_BOTTLE_PATH=str(bottle.parent), WINEDEBUG='-all')
    # Remove inherited wrapper state so it cannot select an unrelated bottle or search path.
    for key in ('CX_INITIALIZED', 'CX_ENV', 'CX_BOTTLE', 'WINEPREFIX', 'WINEDLLPATH', 'WINEDLLOVERRIDES'):
        env.pop(key, None)
    wine = [str(cx/'bin/wine'), '--bottle', str(bottle), '--no-update']

    def run(command, log, timeout=45, check=True, environment=env):
        result = subprocess.run([str(x) for x in command], env=environment, text=True,
                                capture_output=True, timeout=timeout)
        (output/log).write_text(result.stdout + result.stderr)
        if check and result.returncode:
            raise RuntimeError(f'{log}: exit {result.returncode}; inspect the saved output')
        return result

    version = run([cx/'bin/wine', '--version'], 'version.log').stdout.strip()
    run([cx/'bin/cxbottle', '--bottle', str(bottle), '--create', '--template', 'win11_64'], 'create.log', timeout=120)
    protected.append(bottle/'drive_c/windows/system32/netal_wine_host_v1.dll')
    records = []
    owned_bottles = [bottle]
    try:
        llvm = Path('/opt/homebrew/opt/llvm/bin')
        definition = output/'kernel32.def'
        definition.write_text((root/'src/macos_host/re9_wine_kernel32.def').read_text()+'Sleep\n')
        run([llvm/'llvm-dlltool', '-m', 'i386:x86-64', '-d', definition, '-l', output/'kernel32.lib'], 'imports.log')
        run([llvm/'clang', '--target=x86_64-pc-windows-msvc', '-c', '-ffreestanding', '-fno-builtin',
             '-fno-stack-protector', '-Wall', '-Wextra', '-Werror', root/'src/macos_host/netal_wine_fixture_host.c',
             '-o', output/'host.obj'], 'host-build.log')
        run(['/opt/homebrew/bin/lld-link', '/nologo', '/nodefaultlib', '/machine:x64', '/subsystem:console',
             '/entry:mainCRTStartup', f'/out:{output}/host.exe', output/'host.obj', output/'kernel32.lib'], 'host-link.log')
        bridge = runtime/'fixture.dylib'
        run(['xcrun', 'clang++', '-arch', 'x86_64', '-dynamiclib', '-Wl,-install_name,@rpath/fixture.dylib',
             root/'src/macos_host/netal_launch_fixture.cpp', '-o', bridge], 'fixture-build.log')
        run(['codesign', '--force', '--sign', '-', bridge], 'fixture-sign.log')
        base_path = run([*wine, '--ux-app', '/usr/bin/printenv', 'WINEDLLPATH'], 'base-search-path.log').stdout.strip()
        if not base_path or '\n' in base_path:
            raise RuntimeError('Cannot read the bottle search path')
        search_path = str(runtime)+':'+base_path
        for index, case in enumerate(('portable', 'missing-config', 'missing-search-path')):
            case_dir = output/case
            case_dir.mkdir()
            name = f'Netal Fixture {index}-{uuid.uuid4().hex[:8]}.exe'
            host_exe = case_dir/name
            shutil.copy2(output/'host.exe', host_exe)
            receipt = case_dir/'fixture.txt'
            process_env = dict(env, NETAL_TEST_RECEIPT=str(receipt))
            command = wine + ([] if case == 'missing-search-path' else ['--env', 'WINEDLLPATH='+shlex.quote(search_path)])
            session = None
            with (case_dir/'host.log').open('w') as stream:
                host = subprocess.Popen([*command, str(host_exe)], env=process_env, stdout=stream, stderr=subprocess.STDOUT)
                try:
                    pid = None
                    for _ in range(100):
                        rows = subprocess.check_output(['ps', '-axo', 'pid=,comm='], text=True).splitlines()
                        matches = [int(parts[0]) for row in rows if len(parts := row.strip().split(None, 1)) == 2
                                   and parts[1].replace('\\', '/').split('/')[-1] == name]
                        if len(matches) == 1 and 'fixture_host_ready' in (case_dir/'host.log').read_text():
                            pid = matches[0]
                            break
                        if host.poll() is not None:
                            raise RuntimeError('Fixture exited before readiness')
                        time.sleep(.1)
                    if pid is None:
                        raise RuntimeError('Cannot uniquely identify the new fixture process')
                    info = run([config_tool, 'info', str(pid)], f'{case}/identity.log').stdout
                    session = Path(dict(line.split('=', 1) for line in info.splitlines())['session_dir'])
                    if session.exists():
                        raise RuntimeError('Unexpected existing fixture session')
                    controller = runtime/'neuralstudio_re9_inject.exe'
                    dll = 'Z:'+str(runtime/'netal_wine_host_v1.dll')
                    found = run([*command, controller, '--find', name], f'{case}/find.log')
                    assert 'windows_pid=' in found.stdout
                    missing = run([*command, controller, '--find', 'Netal Missing Fixture.exe'], f'{case}/find-missing.log', check=False)
                    assert missing.returncode == 2 and 'target_not_found' in missing.stdout
                    query = run([*command, controller, '--query', name, dll], f'{case}/query.log', check=False)
                    if case == 'missing-search-path':
                        assert query.returncode == 8, query
                    else:
                        assert query.returncode == 0, query
                        values = dict(line.split('=', 1) for line in query.stdout.splitlines())
                        assert int(values['mac_pid']) == pid
                    assert not receipt.exists() and not session.exists(), 'Discovery must not activate or configure the bridge'
                    if case != 'missing-config':
                        run([config_tool, 'prepare', str(pid), '--bridge', bridge, '--sr_bundle', runtime/'unused-sr',
                             '--fg_bundle', runtime/'unused-fg', '--target_exe', name], f'{case}/prepare.log')
                    invalid = []
                    if case == 'portable':
                        for i, supplied in enumerate(([name, 'relative.dll'], [name, dll, 'extra'], ['X'*260, dll])):
                            result = run([*command, controller, *supplied], f'{case}/invalid-{i}.log', check=False)
                            assert result.returncode == 64 and 'invalid_arguments' in result.stdout
                            invalid.append(result.returncode)
                    result = run([*command, controller, name, dll], f'{case}/inject.log', check=False)
                    host.wait(timeout=35)
                    called = receipt.exists()
                    unix_log = session/'re9-live-unixlib.log'
                    unix = unix_log.read_text() if unix_log.exists() else ''
                    (case_dir/'unix.log').write_text(unix)
                    assert called == (case == 'portable'), f'{case}: unexpected fixture execution={called}'
                    if called:
                        assert result.returncode == 0 and 'event=bridge_loaded ' in unix
                        assert f'target={name}' in receipt.read_text()
                        assert 'arm_sr=<unset>' in receipt.read_text() and 'arm_fg=<unset>' in receipt.read_text()
                    else:
                        assert result.returncode == 8, 'Native failure must reach the controller exit status'
                    if case == 'missing-config':
                        assert 'launch configuration refused' in (case_dir/'host.log').read_text()
                    assert not any(path.exists() or path.is_symlink() for path in protected)
                    records.append({'case': case, 'status': 'pass', 'fixture_called': called,
                                    'controller_exit': result.returncode, 'query_exit': query.returncode,
                                    'query_mac_pid_matches': query.returncode == 0, 'invalid_argument_exits': invalid})
                    print(json.dumps(records[-1]), flush=True)
                finally:
                    if host.poll() is None:
                        host.wait(timeout=35)  # the owned fixture exits after 20 seconds
        # Identical executable names in independent Wine servers must never cross-route.
        other_bottle = output/'bottles/NetalHelperOther'
        owned_bottles.append(other_bottle)
        run([cx/'bin/cxbottle', '--bottle', other_bottle, '--create', '--template', 'win11_64'], 'other-create.log', timeout=120)
        protected.append(other_bottle/'drive_c/windows/system32/netal_wine_host_v1.dll')
        multi = output/'two-bottles'
        multi.mkdir()
        shared_name = f'Netal Shared {uuid.uuid4().hex[:8]}.exe'
        shared_exe = multi/shared_name
        shutil.copy2(output/'host.exe', shared_exe)
        processes, routes, native_pids = [], [], []
        with ExitStack() as stack:
            try:
                for index, selected_bottle in enumerate(owned_bottles):
                    case_dir = multi/str(index)
                    case_dir.mkdir()
                    route = [str(cx/'bin/wine'), '--bottle', str(selected_bottle), '--no-update',
                             '--env', 'WINEDLLPATH='+shlex.quote(search_path)]
                    routes.append(route)
                    stream = stack.enter_context((case_dir/'host.log').open('w'))
                    processes.append(subprocess.Popen([*route, str(shared_exe)],
                                     env=dict(env, NETAL_TEST_RECEIPT=str(case_dir/'fixture.txt')), stdout=stream, stderr=subprocess.STDOUT))
                for index, process in enumerate(processes):
                    for _ in range(100):
                        if 'fixture_host_ready' in (multi/str(index)/'host.log').read_text(): break
                        if process.poll() is not None: raise RuntimeError('Shared fixture exited before readiness')
                        time.sleep(.1)
                    else: raise RuntimeError('Shared fixture readiness timed out')
                    query = run([*routes[index], controller, '--query', shared_name, dll], f'two-bottles/{index}/query.log')
                    values = dict(line.split('=',1) for line in query.stdout.splitlines())
                    native_pids.append(int(values['mac_pid']))
                assert len(set(native_pids)) == 2
                for index, pid in enumerate(native_pids):
                    case_dir = multi/str(index)
                    run([config_tool, 'info', str(pid)], f'two-bottles/{index}/identity.log')
                    run([config_tool, 'prepare', str(pid), '--bridge', bridge, '--sr_bundle', case_dir/'sr',
                         '--fg_bundle', case_dir/'fg', '--target_exe', shared_name], f'two-bottles/{index}/prepare.log')
                    run([*routes[index], controller, shared_name, dll], f'two-bottles/{index}/inject.log')
                    receipt = (case_dir/'fixture.txt').read_text()
                    assert f'sr={case_dir}/sr' in receipt and f'target={shared_name}' in receipt
                    if index == 0: assert not (multi/'1/fixture.txt').exists()
                duplicate_log = stack.enter_context((multi/'duplicate.log').open('w'))
                duplicate = subprocess.Popen([*routes[0], str(shared_exe)], env=env, stdout=duplicate_log, stderr=subprocess.STDOUT)
                processes.append(duplicate)
                for _ in range(100):
                    if 'fixture_host_ready' in (multi/'duplicate.log').read_text(): break
                    if duplicate.poll() is not None: raise RuntimeError('Duplicate fixture exited')
                    time.sleep(.1)
                else: raise RuntimeError('Duplicate fixture readiness timed out')
                for mode in ('--find', '--query'):
                    result = run([*routes[0], controller, mode, shared_name, *([dll] if mode == '--query' else [])],
                                 f'two-bottles/duplicate-{mode[2:]}.log', check=False)
                    assert result.returncode == 9 and 'target_ambiguous' in result.stdout
                assert not any(path.exists() or path.is_symlink() for path in protected)
                records.append({'case': 'two-bottles', 'status': 'pass', 'mac_pids_distinct': True,
                                'bridge_receipts_isolated': True, 'duplicate_find_and_query_refused': True})
                print(json.dumps(records[-1]), flush=True)
            finally:
                for process in processes:
                    if process.poll() is None: process.wait(timeout=35)
        result = {'schema': 'netal.crossover-helper.v1', 'status': 'pass', 'scope': 'fixture-injection-not-gameplay',
                  'crossover_version': version, 'cases': records, 'protected_hook_files_absent': True,
                  'models_used': False, 'gpu_inference': False,
                  'helpers': {path.name: digest(path) for path in required},
                  'sources': {str(path.relative_to(root)): digest(path) for path in
                              [root/'src/macos_host/re9_wine_inject_controller.c', root/'src/macos_host/netal_wine_fixture_host.c',
                               root/'src/macos_host/re9_wine_unixlib.cpp', root/'src/macos_host/netal_launch_config.h']}}
    finally:
        # This unique prefix was created above. Never stop an existing user bottle.
        for index, owned in enumerate(owned_bottles):
            if not owned.is_dir(): continue
            server_env = dict(env, WINEPREFIX=str(owned))
            stop = run([cx/'bin/wineserver', '-k'], f'server-stop-{index}.log', check=False, environment=server_env)
            if stop.returncode not in (0, 1):
                raise RuntimeError('Disposable bottle server could not be stopped')
            run([cx/'bin/wineserver', '-w'], f'server-wait-{index}.log', timeout=30, environment=server_env)
    result['disposable_server_stopped'] = True
    (output/'result.json').write_text(json.dumps(result, indent=2)+'\n')
    print('CrossOver fixture checks passed; no gameplay or distribution claim', flush=True)


if __name__ == '__main__':
    main()
