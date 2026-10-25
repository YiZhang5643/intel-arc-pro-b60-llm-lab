"""Foreground llama-server launcher. Never kills an existing server or changes global settings."""
import argparse
import ctypes
import json
import os
import subprocess
import threading
import time
import urllib.request
from pathlib import Path


def command(preset, runtime_dir, model_dir, host, port, device):
    if not (0 < preset['ubatch'] <= preset['batch']):
        raise ValueError('Require 0 < ubatch <= batch')
    if preset['context'] < 4096:
        raise ValueError('context too small')
    model_dir = Path(model_dir).resolve()
    def model_path(name):
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Model names must be filenames, not paths')
        return str(model_dir / name)
    cmd = [str(Path(runtime_dir).resolve()/'llama-server.exe'), '-m', model_path(preset['model']),
           '--device', device, '-ngl', '99', '-c', str(preset['context']), '-np', '1',
           '-fa', 'on', '-ctk', preset['kv_type'], '-ctv', preset['kv_type'],
           '-b', str(preset['batch']), '-ub', str(preset['ubatch']), '--cache-ram', '2048',
           '--fit', 'off', '--jinja', '--reasoning', 'off', '--temp', '0.2',
           '--top-p', '0.9', '--top-k', '20', '--min-p', '0',
           '--host', host, '--port', str(port), '--metrics', '--alias', preset.get('alias', 'nemotron-local'), '-lv', '3']
    if preset.get('ui_mcp_proxy',False):
        cmd += ['--ui-mcp-proxy']
    if preset['mtp']:
        if preset.get('mtp_mode', 'external') == 'external':
            cmd += ['-md', model_path(preset['draft_model'])]
        elif preset['mtp_mode'] != 'embedded':
            raise ValueError('mtp_mode must be external or embedded')
        cmd += ['--device-draft', device, '-ngld', '99',
                '--spec-type', 'draft-mtp', '--spec-draft-n-max', str(preset['draft_max']),
                '--spec-draft-p-min', str(preset['draft_p_min'])]
    if preset['mtp'] and preset.get('draft_ubatch') is not None:
        if preset['draft_ubatch'] < 1: raise ValueError('draft_ubatch must be positive')
        cmd += ['--spec-draft-n-ubatch',str(preset['draft_ubatch'])]
    return cmd


def trim_after_ready(child, port):
    # Exactly once after this child becomes healthy. No repeated trimming during inference.
    for _ in range(300):
        if child.poll() is not None: return
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=1) as f:
                if json.load(f).get('status') == 'ok': break
        except (OSError, ValueError): pass
        time.sleep(1)
    else:
        print('Working-set trim skipped: health timeout', flush=True)
        return
    from ctypes import wintypes
    k = ctypes.WinDLL('kernel32', use_last_error=True)
    ps = ctypes.WinDLL('psapi', use_last_error=True)
    k.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k.OpenProcess.restype = wintypes.HANDLE
    k.CloseHandle.argtypes = [wintypes.HANDLE]
    ps.EmptyWorkingSet.argtypes = [wintypes.HANDLE]
    ps.EmptyWorkingSet.restype = wintypes.BOOL
    handle = k.OpenProcess(0x100 | 0x400, False, child.pid)
    if handle:
        try:
            trimmed=bool(ps.EmptyWorkingSet(handle))
            print('Post-load working-set trim:', trimmed, flush=True)
            return trimmed
        finally: k.CloseHandle(handle)
    return False


def environment(preset, device_selector='level_zero:0'):
    return {'ONEAPI_DEVICE_SELECTOR':device_selector, 'ZES_ENABLE_SYSMAN':'1',
            'SYCL_CACHE_PERSISTENT':'1', 'GGML_SYCL_SPARSE_FA':'0',
            'GGML_SYCL_ENABLE_HOST_PINNED_MEM':'0',
            'GGML_SYCL_MOE_Q4_REORDER':'1' if preset['q4_reorder'] else '0',
            'GGML_SYCL_ENABLE_GRAPH':'1' if preset.get('sycl_graph',False) else '0',
            'GGML_SYCL_GDN_BLOCKED':'1' if preset.get('sycl_gdn_blocked',False) else '0',
            'GGML_SYCL_FA_ONEDNN':'1' if preset.get('sycl_fa_onednn',True) else '0'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--preset', type=Path, required=True)
    p.add_argument('--runtime-dir', type=Path, required=True)
    p.add_argument('--model-dir', type=Path, required=True)
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=5001)
    p.add_argument('--device', default='SYCL0')
    p.add_argument('--device-selector', default='level_zero:0')
    p.add_argument('--trim-working-set', action='store_true', help='Windows: once after model loading')
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args()
    if not 1 <= a.port <= 65535: p.error('Invalid port')
    if a.trim_working_set and os.name != 'nt': p.error('Working-set trim is Windows only')
    preset = json.loads(a.preset.read_text(encoding='utf-8'))
    cmd = command(preset, a.runtime_dir, a.model_dir, a.host, a.port, a.device)
    overrides = environment(preset, a.device_selector)
    print(json.dumps({'command': cmd, 'environment_overrides': overrides}, indent=2), flush=True)
    if a.dry_run: return
    for path in [cmd[0], str(a.model_dir/preset['model'])] + ([str(a.model_dir/preset['draft_model'])] if preset['mtp'] and preset.get('mtp_mode','external')=='external' else []):
        if not Path(path).is_file(): p.error('Missing file: '+path)
    child = subprocess.Popen(cmd, cwd=a.runtime_dir.resolve(), env={**os.environ, **overrides})
    if a.trim_working_set:
        threading.Thread(target=trim_after_ready, args=(child, a.port), daemon=True).start()
    try: raise SystemExit(child.wait())
    except KeyboardInterrupt:
        # Ctrl-C is also delivered to the foreground child; terminate only our child if still alive.
        try: child.wait(timeout=5)
        except subprocess.TimeoutExpired: child.terminate(); child.wait()
        raise SystemExit(130)


if __name__ == '__main__': main()
