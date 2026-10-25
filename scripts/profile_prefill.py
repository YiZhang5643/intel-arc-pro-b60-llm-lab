"""Profile one saved synthetic cold prompt on the recorded Windows server.

Stops only the verified recorded server, launches the same command under Intel
unitrace, warms up while paused, collects one cold request, closes its own hidden
console gracefully, and restores the normal preset in finally.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def call(base, path, data=None, timeout=1800):
    raw = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(base+path, data=raw,
                                 headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as f:
        return json.load(f)


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def request(base, tokens):
    start = time.perf_counter()
    result = call(base, '/completion', {'prompt': tokens, 'n_predict': 1,
                  'temperature': 0, 'seed': 20261026, 'cache_prompt': False})
    elapsed = time.perf_counter()-start
    t = result['timings']
    assert t['cache_n'] == 0 and t['prompt_n'] == len(tokens), t
    return {'wall_seconds': elapsed, 'timings': t, 'output': result.get('content', '')}


def signal_console(pid):
    # Runs in a separate helper so this caller's console is untouched.
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    k.FreeConsole()
    if not k.AttachConsole(pid):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.SetConsoleCtrlHandler(None, True):
        raise ctypes.WinError(ctypes.get_last_error())
    if not k.GenerateConsoleCtrlEvent(0, 0):
        raise ctypes.WinError(ctypes.get_last_error())
    time.sleep(0.5)


def child_pid(parent):
    ps = f"Get-CimInstance Win32_Process -Filter 'ParentProcessId={parent}' | Select-Object ProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Compress"
    value = subprocess.check_output(['powershell', '-NoProfile', '-Command', ps],
                                    creationflags=subprocess.CREATE_NO_WINDOW)
    rows = json.loads(value.decode('utf-8-sig'))
    if isinstance(rows, dict):
        rows = [rows]
    return next(r for r in rows if Path(r['ExecutablePath']).name.lower() == 'llama-server.exe')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir', type=Path, required=True)
    p.add_argument('--prompts', type=Path, required=True)
    p.add_argument('--case', default='cold-retrieval')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--unitrace', type=Path, required=True)
    p.add_argument('--skip-baseline', action='store_true')
    p.add_argument('--batch', type=int)
    p.add_argument('--ubatch', type=int)
    p.add_argument('--extra-env', action='append', default=[], metavar='KEY=VALUE')
    a = p.parse_args()
    deployment = a.deployment_dir.resolve()
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    record = deployment/'server-process.json'
    original = json.loads(record.read_text(encoding='utf-8'))
    write(out/'original-server.local.json', original)
    declared = json.loads(Path(original['preset']).read_text(encoding='utf-8'))
    if declared != original.get('effective_preset'):
        raise RuntimeError('Use a recorded server matching its preset; move runtime overrides into a preset before profiling')
    base = f"http://127.0.0.1:{original['port']}"
    slots = call(base, '/slots')
    if any(s.get('is_processing') for s in slots):
        raise RuntimeError('Server has an active request; profiling did not stop it')
    prompts = json.loads(a.prompts.read_text(encoding='utf-8'))
    tokens = prompts['cases'][a.case]['tokens']
    digest = hashlib.sha256(json.dumps(tokens, separators=(',', ':')).encode()).hexdigest()
    write(out/'input.json', {'tokens': len(tokens), 'token_sha256': digest,
                            'case': a.case, 'synthetic_source': a.prompts.parent.name})
    if not a.skip_baseline:
        print('BASELINE WARMUP', flush=True)
        write(out/'baseline-warmup.json', request(base, tokens[:4096]))
        print('BASELINE COLD', len(tokens), flush=True)
        result = request(base, tokens)
        write(out/'baseline.json', result)
        print('BASELINE', result['timings'], flush=True)
    from manage_server import stop
    stop(record)
    profile_record = out/'profile-server.local.json'
    launcher = None
    log_handle = (out/'launcher.log').open('wb')
    try:
        env = {**os.environ, **original['environment_overrides']}
        changes = {}
        for item in a.extra_env:
            key, value = item.split('=', 1)
            if not key.startswith(('GGML_', 'SYCL_', 'ONEAPI_', 'ONEDNN_', 'DNNL_')):
                raise ValueError('Only inference environment overrides allowed')
            changes[key] = value
        env.update(changes)
        env['PATH'] = str(Path(original['exe']).parent)+os.pathsep+str(a.unitrace.resolve().parent)+os.pathsep+env['PATH']
        session = 'b60p'+str(os.getpid())
        server_cmd = original['args'] + ['--log-file', str(out/'server.log')]
        for flag, value in [('-b', a.batch), ('-ub', a.ubatch)]:
            if value is not None:
                if value <= 0: raise ValueError('Batch sizes must be positive')
                server_cmd[server_cmd.index(flag)+1] = str(value)
        cmd = [str(a.unitrace.resolve()), '-d', '-h', '--chrome-kernel-logging',
               '--chrome-call-logging', '--start-paused', '--session', session,
               '--output-dir-path', str(out), '-o', str(out/'device-timing.txt')] + server_cmd
        write(out/'launch.local.json', {'command': cmd, 'environment_changes': changes,
                                       'environment_overrides': original['environment_overrides']})
        si = subprocess.STARTUPINFO()
        si.dwFlags = subprocess.STARTF_USESHOWWINDOW
        si.wShowWindow = 0
        launcher = subprocess.Popen(cmd, cwd=Path(original['exe']).parent, env=env,
                  stdout=log_handle, stderr=subprocess.STDOUT, startupinfo=si,
                  creationflags=subprocess.CREATE_NEW_CONSOLE)
        for _ in range(300):
            if launcher.poll() is not None:
                raise RuntimeError('Profile launcher exited; inspect launcher.log')
            try:
                if call(base, '/health', timeout=2).get('status') == 'ok':
                    break
            except (OSError, ValueError):
                pass
            time.sleep(1)
        else:
            raise RuntimeError('Profile server health timeout')
        child = child_pid(launcher.pid)
        if Path(child['ExecutablePath']).resolve() != Path(original['exe']).resolve():
            raise RuntimeError('Profile child path mismatch')
        pid = child['ProcessId']
        state = {**original, 'pid': pid, 'args': server_cmd, 'stdout': str(out/'launcher.log'),
                 'stderr': str(out/'server.log')}
        write(profile_record, state)
        from serve import trim_after_ready
        class Child:
            def __init__(self, n): self.pid = n
            def poll(self): return None
        state['postload_working_set_trim'] = bool(trim_after_ready(Child(pid), original['port']))
        write(profile_record, state)
        print('PROFILE WARMUP', flush=True)
        write(out/'profile-warmup.json', request(base, tokens[:4096]))
        # Named shared-memory controls work on this Windows build; verify their response.
        ctl = subprocess.run([str(a.unitrace.resolve()), '--resume', session],
                             capture_output=True, env=env)
        write(out/'resume.json', {'returncode': ctl.returncode,
                                 'stdout': ctl.stdout.decode(errors='replace'),
                                 'stderr': ctl.stderr.decode(errors='replace')})
        if ctl.returncode or b'is resumed' not in ctl.stderr+ctl.stdout:
            raise RuntimeError('unitrace did not resume collection')
        print('PROFILE COLD', len(tokens), flush=True)
        start_ns = time.time_ns()
        result = request(base, tokens)
        result['client_start_unix_ns'] = start_ns
        result['client_end_unix_ns'] = time.time_ns()
        write(out/'profile.json', result)
        print('PROFILE', result['timings'], flush=True)
        subprocess.run([str(a.unitrace.resolve()), '--pause', session],
                       capture_output=True, env=env, check=True)
        # Ctrl-C goes only to this launcher's private hidden console.
        subprocess.run([sys.executable, str(Path(__file__).resolve()),
                        '--signal-console', str(pid)], check=True,
                       creationflags=subprocess.CREATE_NO_WINDOW)
        for _ in range(60):
            try:
                call(base, '/health', timeout=1)
            except OSError:
                break
            time.sleep(1)
        try:
            launcher.wait(timeout=15)
        except subprocess.TimeoutExpired:
            launcher.terminate()
            launcher.wait(timeout=10)
        print('TRACE FILES', [p.name for p in out.glob('*.json')], flush=True)
    finally:
        # Recover ownership evidence even if startup failed before /health became ready.
        if launcher is not None and not profile_record.exists():
            try:
                candidate = child_pid(launcher.pid)
                same_exe = Path(candidate['ExecutablePath']).resolve() == Path(original['exe']).resolve()
                own_log = str(out/'server.log').lower() in (candidate.get('CommandLine') or '').lower()
                if same_exe and own_log:
                    write(profile_record, {'pid': candidate['ProcessId'], 'exe': original['exe']})
            except (subprocess.CalledProcessError, json.JSONDecodeError, StopIteration, TypeError):
                pass
        if profile_record.exists():
            stop(profile_record)
        if launcher is not None and launcher.poll() is None:
            launcher.terminate()
            launcher.wait(timeout=10)
        log_handle.close()
        restore = [sys.executable, str(ROOT/'scripts/manage_server.py'), 'start',
                   '--deployment-dir', str(deployment), '--preset', original['preset'],
                   '--runtime', str(Path(original['exe']).parent.relative_to(deployment)),
                   '--host', original['host'], '--port', str(original['port'])]
        print('RESTORING NORMAL SERVER', flush=True)
        subprocess.run(restore, check=True, timeout=650)
        print('RESTORED', flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--signal-console':
        signal_console(int(sys.argv[2]))
    else:
        main()
