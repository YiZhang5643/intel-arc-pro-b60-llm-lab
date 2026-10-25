"""Paired cold-prefill GDN experiment; restores the recorded server in finally."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from profile_prefill import call, request, write
from manage_server import stop

ROOT = Path(__file__).resolve().parents[1]

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir', type=Path, required=True)
    p.add_argument('--runtime', required=True)
    p.add_argument('--prompts', type=Path, action='append', required=True)
    p.add_argument('--order', default='0,1,1,0')
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    order = [int(x) for x in a.order.split(',')]
    if not order or any(x not in (0, 1) for x in order):
        p.error('order must contain 0 or 1')
    out = a.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    dep = a.deployment_dir.resolve()
    runtime = (dep / a.runtime).resolve()
    if runtime.parent != dep or runtime.name == 'runtime-patched-f16':
        p.error('Use an isolated runtime directly inside deployment-dir')
    original = json.loads((dep/'server-process.json').read_text())
    declared = json.loads(Path(original['preset']).read_text())
    if declared != original['effective_preset']:
        raise RuntimeError('Recorded preset must match its file for exact restoration')
    base = f"http://127.0.0.1:{original['port']}"
    if any(x.get('is_processing') for x in call(base, '/slots')):
        raise RuntimeError('Active request; experiment did not stop the server')
    write(out/'original-server.local.json', original)
    cases = []
    for path in a.prompts:
        obj = json.loads(path.read_text())
        tokens = obj['cases']['cold-retrieval']['tokens']
        cases.append((tokens, obj['expected']))
    write(out/'inputs.json', [dict(tokens=len(t), token_sha256=hashlib.sha256(json.dumps(t,separators=(',', ':')).encode()).hexdigest()) for t,_ in cases])
    dll = runtime/'ggml-sycl.dll'
    write(out/'runtime.json', {'runtime': runtime.name, 'ggml_sycl_sha256': hashlib.sha256(dll.read_bytes()).hexdigest(), 'order': order})
    manager = [sys.executable, str(ROOT/'scripts/manage_server.py')]
    restore = manager + ['start','--deployment-dir',str(dep),'--preset',original['preset'],'--runtime',Path(original['exe']).parent.name,'--host',original['host'],'--port',str(original['port'])]
    env = dict(os.environ)
    env.pop('GGML_SYCL_GDN_BLOCKED', None)
    rows = []
    try:
        for run, flag in enumerate(order, 1):
            stop(dep/'server-process.json')
            candidate = {**declared, 'sycl_gdn_blocked': bool(flag)}
            candidate_path = out/f'run{run}-preset.local.json'
            write(candidate_path, candidate)
            launch_env = {**env, 'GGML_SYCL_GDN_BLOCKED':str(flag)}
            subprocess.run([*manager, 'start', '--deployment-dir',str(dep),'--preset',str(candidate_path),'--runtime',runtime.name,'--host',original['host'],'--port',str(original['port'])], env=launch_env, check=True, timeout=650)
            write(out/f'run{run}-server.local.json', json.loads((dep/'server-process.json').read_text()))
            write(out/f'run{run}-warmup.json', request(base, cases[0][0][:4096]))
            for tokens, expected in cases:
                print('COLD', run, flag, len(tokens), flush=True)
                row = {'run':run,'blocked':flag,'input_tokens':len(tokens), **request(base,tokens)}
                rows.append(row)
                write(out/'summary.json', rows)
                print(json.dumps(row), flush=True)
                # A cached continuation exercises MTP without a second long prefill.
                start=time.perf_counter()
                result=call(base,'/completion',{'prompt':tokens,'n_predict':256,'temperature':0,'seed':20261026,'cache_prompt':True})
                output=result.get('content','')
                quality={'run':run,'blocked':flag,'input_tokens':len(tokens),'wall_seconds':time.perf_counter()-start,'result':result,'retrieval_all_present':all(value in output for value in expected.values())}
                write(out/f'run{run}-{len(tokens)}-retrieval.json', quality)
                print('RETRIEVAL',quality['retrieval_all_present'],result['timings'],flush=True)
    finally:
        try:
            stop(dep/'server-process.json')
        finally:
            subprocess.run(restore, env=env, check=True, timeout=650)
            print('RESTORED', flush=True)

if __name__=='__main__':
    main()
