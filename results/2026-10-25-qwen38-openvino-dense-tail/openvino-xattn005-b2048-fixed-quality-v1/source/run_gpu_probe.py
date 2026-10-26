"""Run a standalone GPU probe without another loaded model, then restore the recorded server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from manage_server import stop
from profile_prefill import call, write

ROOT=Path(__file__).resolve().parents[1]


def run_process(command, log, env, timeout):
    """Keep an owned process alive until its entire Windows child tree is cleaned up."""
    with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env) as process:
        try:
            return process.wait(timeout=timeout)
        finally:
            if process.poll() is None:
                if os.name == 'nt':
                    result=subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE,timeout=30)
                    if result.returncode and process.poll() is None:
                        raise RuntimeError('Could not stop the owned probe process tree')
                else:
                    process.kill()
                process.wait(timeout=30)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir',type=Path,required=True)
    p.add_argument('--executable',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--argument',action='append',default=[],help='Repeat for each probe argument')
    p.add_argument('--timeout',type=int,default=240)
    a=p.parse_args()
    if os.name!='nt':p.error('Recorded Windows service required')
    exe=a.executable.resolve()
    if not exe.is_file():p.error('Probe executable is missing')
    if a.timeout<=0:p.error('timeout must be positive')
    dep=a.deployment_dir.resolve();record=dep/'server-process.json'
    state=json.loads(record.read_text())
    if json.loads(Path(state['preset']).read_text())!=state['effective_preset']:
        raise RuntimeError('Original service must match its preset file')
    if any(s['is_processing'] for s in call(f"http://127.0.0.1:{state['port']}",'/slots')):
        raise RuntimeError('Active request; no process was stopped')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    write(out/'original-server.local.json',state)
    write(out/'probe.json',{'executable':exe.name,'sha256':hashlib.sha256(exe.read_bytes()).hexdigest(),'arguments':a.argument})
    env={**os.environ,'ONEAPI_DEVICE_SELECTOR':state['environment_overrides']['ONEAPI_DEVICE_SELECTOR'],'PATH':str(Path(state['exe']).parent)+os.pathsep+os.environ['PATH']}
    try:
        stop(record)
        with (out/'stdout.txt').open('wb') as f:
            returncode=run_process([str(exe),*a.argument],f,env,a.timeout)
        write(out/'status.json',{'returncode':returncode})
        if returncode:raise RuntimeError('Probe failed; see stdout.txt')
    finally:
        subprocess.run([sys.executable,str(ROOT/'scripts/manage_server.py'),'start','--deployment-dir',str(dep),'--preset',state['preset'],'--runtime',Path(state['exe']).parent.name,'--host',state['host'],'--port',str(state['port']),'--device',state['args'][state['args'].index('--device')+1]],check=True,timeout=650)
        print('Original service restored',flush=True)

if __name__=='__main__':main()
