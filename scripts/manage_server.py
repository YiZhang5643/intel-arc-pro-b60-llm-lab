"""Manage only the recorded local Windows llama-server, with executable-path verification."""
import argparse,ctypes,json,os,socket,subprocess,time,urllib.request
from pathlib import Path
from serve import command,environment,trim_after_ready

def stop(record):
    if not record.exists():return
    state=json.loads(record.read_text(encoding='utf-8'));pid=state['pid'];expected=state['exe']
    from ctypes import wintypes
    k=ctypes.WinDLL('kernel32',use_last_error=True)
    k.OpenProcess.argtypes=[wintypes.DWORD,wintypes.BOOL,wintypes.DWORD];k.OpenProcess.restype=wintypes.HANDLE
    k.QueryFullProcessImageNameW.argtypes=[wintypes.HANDLE,wintypes.DWORD,wintypes.LPWSTR,ctypes.POINTER(wintypes.DWORD)];k.QueryFullProcessImageNameW.restype=wintypes.BOOL
    k.TerminateProcess.argtypes=[wintypes.HANDLE,wintypes.UINT];k.TerminateProcess.restype=wintypes.BOOL
    k.WaitForSingleObject.argtypes=[wintypes.HANDLE,wintypes.DWORD];k.WaitForSingleObject.restype=wintypes.DWORD
    k.CloseHandle.argtypes=[wintypes.HANDLE]
    handle=k.OpenProcess(0x1000|0x100000|1,False,pid)
    if not handle:
        error=ctypes.get_last_error()
        if error!=87:raise ctypes.WinError(error)
        print('Recorded server already exited');return
    try:
        n=wintypes.DWORD(32768);name=ctypes.create_unicode_buffer(n.value)
        if not k.QueryFullProcessImageNameW(handle,0,name,ctypes.byref(n)):raise ctypes.WinError(ctypes.get_last_error())
        if os.path.normcase(os.path.abspath(name.value))!=os.path.normcase(os.path.abspath(expected)):raise RuntimeError('Recorded PID belongs to a different executable')
        if not k.TerminateProcess(handle,0):raise ctypes.WinError(ctypes.get_last_error())
        if k.WaitForSingleObject(handle,30000)!=0:raise RuntimeError('Server stop timeout')
    finally:k.CloseHandle(handle)
    state['stopped']=time.strftime('%Y%m%d-%H%M%S');record.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print('Recorded server stopped',flush=True)

def start(a,record):
    preset=json.loads(a.preset.read_text(encoding='utf-8'))
    for key in ['context','batch','ubatch','kv_type','draft_max','draft_ubatch']:
        value=getattr(a,key,None)
        if value is not None:preset[key]=value
    if a.spec is not None:preset['mtp']=a.spec=='on'
    if a.q4_reorder is not None:preset['q4_reorder']=a.q4_reorder=='on'
    deployment=a.deployment_dir.resolve();runtime=deployment/a.runtime
    try:
        with socket.create_connection(('127.0.0.1',a.port),timeout=1):pass
    except OSError:pass
    else:raise RuntimeError('Port is occupied; stop its recorded server explicitly first')
    cmd=command(preset,runtime,deployment/'models',a.host,a.port,a.device)
    # Verbose startup records offload, memory and compiled backend details for diagnosis.
    cmd[cmd.index('-lv')+1]=str(a.verbosity)
    paths=[Path(cmd[0]),deployment/'models'/preset['model']]
    if preset['mtp'] and preset.get('mtp_mode','external')=='external':paths.append(deployment/'models'/preset['draft_model'])
    for path in paths:
        if not path.is_file():raise FileNotFoundError(path)
    overrides=environment(preset,a.device_selector)
    stamp=time.strftime('%Y%m%d-%H%M%S');logs=deployment/'logs';logs.mkdir(exist_ok=True)
    out=logs/f'server-{stamp}.out.log';err=logs/f'server-{stamp}.err.log'
    with out.open('wb') as o,err.open('wb') as e:
        child=subprocess.Popen(cmd,cwd=runtime,env={**os.environ,**overrides},stdout=o,stderr=e,creationflags=subprocess.CREATE_NO_WINDOW)
    state={'pid':child.pid,'exe':cmd[0],'args':cmd,'host':a.host,'port':a.port,'context':preset['context'],
           'spec':'on' if preset['mtp'] else 'off','batch':preset['batch'],'ubatch':preset['ubatch'],
           'kv_type':preset['kv_type'],'draft_max':preset['draft_max'],'preset':str(a.preset.resolve()),'effective_preset':preset,
           'stdout':str(out),'stderr':str(err),'started':stamp,'environment_overrides':overrides}
    record.write_text(json.dumps(state,indent=2),encoding='utf-8')
    print(f"Starting {preset.get('alias','local-model')}: PID={child.pid}, MTP={state['spec']}, graph={preset.get('sycl_graph',False)}",flush=True)
    for _ in range(600):
        if child.poll() is not None:raise RuntimeError('Server exited; inspect '+str(err))
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=2) as f:
                if json.load(f).get('status')=='ok':break
        except (OSError,ValueError):pass
        time.sleep(1)
    else:raise RuntimeError('Server health timeout')
    state['postload_working_set_trim']=bool(trim_after_ready(child,a.port))
    record.write_text(json.dumps(state,indent=2),encoding='utf-8');print('READY',flush=True)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['start','stop','status','restart'])
    p.add_argument('--deployment-dir',type=Path,required=True);p.add_argument('--preset',type=Path)
    p.add_argument('--runtime',default='runtime-patched-f16');p.add_argument('--host',default='0.0.0.0');p.add_argument('--port',type=int,default=5001)
    p.add_argument('--device',default='SYCL0');p.add_argument('--device-selector',default='level_zero:0')
    for key in ['context','batch','ubatch','draft-max','draft-ubatch']:p.add_argument('--'+key,type=int)
    p.add_argument('--kv-type',choices=['f16','q8_0']);p.add_argument('--spec',choices=['on','off'])
    p.add_argument('--q4-reorder',choices=['on','off']);p.add_argument('--verbosity',type=int,default=4);a=p.parse_args()
    if os.name!='nt':p.error('Managed process lifecycle currently requires Windows')
    record=a.deployment_dir.resolve()/'server-process.json'
    if a.action=='stop':stop(record)
    elif a.action in ['start','restart']:
        if not a.preset:p.error('--preset required for start/restart')
        if a.action=='restart':stop(record)
        start(a,record)
    else:
        with urllib.request.urlopen(f'http://127.0.0.1:{a.port}/health',timeout=3) as f:print(f.read().decode())
if __name__=='__main__':main()
