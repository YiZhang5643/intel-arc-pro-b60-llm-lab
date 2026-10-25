"""Test an isolated backend/preset on saved tokens, then restore the recorded service."""
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
from serve import command

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir',type=Path,required=True)
    p.add_argument('--runtime',required=True)
    p.add_argument('--preset',type=Path,required=True)
    p.add_argument('--device',required=True,help='Verified device name from --list-devices, e.g. Vulkan1')
    p.add_argument('--prompts',type=Path,action='append',required=True)
    p.add_argument('--repeats',type=int,default=2)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    if a.repeats<1:p.error('repeats must be positive')
    dep=a.deployment_dir.resolve()
    runtime=(dep/a.runtime).resolve()
    if runtime.parent!=dep:p.error('runtime must be a direct deployment child')
    if not (runtime/'llama-server.exe').is_file():p.error('Candidate server does not exist')
    original=json.loads((dep/'server-process.json').read_text())
    if json.loads(Path(original['preset']).read_text())!=original['effective_preset']:
        raise RuntimeError('Original server must match its preset file')
    preset=json.loads(a.preset.read_text())
    cmd=command(preset,runtime,dep/'models',original['host'],original['port'],a.device)
    for flag in ['-m','-md']:
        if flag in cmd and not Path(cmd[cmd.index(flag)+1]).is_file():raise FileNotFoundError(flag)
    base=f"http://127.0.0.1:{original['port']}"
    if any(x.get('is_processing') for x in call(base,'/slots')):
        raise RuntimeError('Active request; original server was not stopped')
    cases=[]
    for path in a.prompts:
        obj=json.loads(path.read_text())
        cases.append((obj['cases']['cold-retrieval']['tokens'],obj['expected']))
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'original-server.local.json',original)
    write(out/'inputs.json',[{'tokens':len(t),'token_sha256':hashlib.sha256(json.dumps(t,separators=(',',':')).encode()).hexdigest()} for t,_ in cases])
    write(out/'runtime.json',{'runtime':runtime.name,'preset':preset,'device':a.device,'sha256':{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in runtime.iterdir() if x.suffix.lower() in ('.exe','.dll')}})
    manager=[sys.executable,str(ROOT/'scripts/manage_server.py')]
    common=['--deployment-dir',str(dep),'--host',original['host'],'--port',str(original['port'])]
    restore=manager+['start',*common,'--runtime',Path(original['exe']).parent.name,'--preset',original['preset'],'--device',original['args'][original['args'].index('--device')+1]]
    env=dict(os.environ);env.pop('GGML_SYCL_GDN_BLOCKED',None)
    rows=[]
    try:
        stop(dep/'server-process.json')
        subprocess.run(manager+['start',*common,'--runtime',runtime.name,'--preset',str(a.preset.resolve()),'--device',a.device],check=True,env=env,timeout=650)
        write(out/'candidate-server.local.json',json.loads((dep/'server-process.json').read_text()))
        write(out/'warmup.json',request(base,cases[0][0][:4096]))
        for rep in range(1,a.repeats+1):
            for tokens,expected in cases:
                print('COLD',rep,len(tokens),flush=True)
                row={'repeat':rep,'input_tokens':len(tokens),**request(base,tokens)}
                rows.append(row);write(out/'summary.json',rows);print(json.dumps(row),flush=True)
                started=time.perf_counter()
                result=call(base,'/completion',{'prompt':tokens,'n_predict':256,'temperature':0,'seed':20261026,'cache_prompt':True})
                quality={'repeat':rep,'input_tokens':len(tokens),'wall_seconds':time.perf_counter()-started,'result':result,'retrieval_all_present':all(v in result.get('content','') for v in expected.values())}
                write(out/f'r{rep}-{len(tokens)}-retrieval.json',quality)
                print('RETRIEVAL',quality['retrieval_all_present'],result['timings'],flush=True)
    finally:
        try:
            stop(dep/'server-process.json')
        finally:
            subprocess.run(restore,check=True,env=env,timeout=650)
            print('RESTORED',flush=True)

if __name__=='__main__':
    main()
