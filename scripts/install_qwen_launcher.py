"""Install forwarding Qwen launchers while preserving previous user launchers."""
import argparse,os,shutil,time
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment-dir',type=Path,required=True);a=p.parse_args()
deployment=a.deployment_dir.resolve()
if not (deployment/'models').is_dir():p.error('Expected an existing deployment with models/')
repo=Path(__file__).resolve().parents[1];entry=repo/'Start-Qwen.cmd'
if not entry.is_file():p.error('Repository Start-Qwen.cmd is missing')
stamp=time.strftime('%Y%m%d-%H%M%S')
for name in ['Start-Qwen.cmd','qwen26b_server.py']:
    target=deployment/name
    if target.exists():
        backup=target.with_name(target.stem+'.before-tuning-'+stamp+target.suffix)
        if backup.exists():raise FileExistsError(backup)
        shutil.copyfile(target,backup)
try:entry_ref='%~dp0'+os.path.relpath(entry,deployment)
except ValueError:entry_ref=str(entry)
content='@echo off\ncall "'+entry_ref+'" "%~dp0."\n'
with (deployment/'Start-Qwen.cmd').open('w',encoding='utf-8',newline='\r\n') as f:f.write(content)
try:relative_repo=Path(os.path.relpath(repo,deployment))
except ValueError:relative_repo=repo
wrapper='\n'.join([
    '"""Compatibility entry; implementation and defaults live in the local lab repository."""',
    'from pathlib import Path',
    'import subprocess,sys',
    'root=Path(__file__).resolve().parent',
    'repo=(root/'+repr(str(relative_repo))+').resolve()',
    "cmd=[sys.executable,str(repo/'scripts/manage_server.py'),'--deployment-dir',str(root),'--preset',str(repo/'presets/qwen38-recommended.json'),*sys.argv[1:]]",
    'raise SystemExit(subprocess.call(cmd))',''])
with (deployment/'qwen26b_server.py').open('w',encoding='utf-8',newline='\n') as f:f.write(wrapper)
print('Installed forwarding launchers; previous files preserved with .before-tuning timestamps.')
