"""Sequential same-input probes for Windows SYCL presets; keeps the last successful server running."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--deployment-dir',type=Path,required=True);p.add_argument('--presets',nargs='+',type=Path,required=True);p.add_argument('--output-root',type=Path,required=True);a=p.parse_args()
R=Path(__file__).resolve().parent
record=a.deployment_dir/'server-process.json'
initial=json.loads(record.read_text(encoding='utf-8')) if record.exists() else {}
last_good=Path(initial['preset']) if initial.get('preset') and Path(initial['preset']).is_file() else None
base=[sys.executable,str(R/'manage_server.py')]
try:
    for preset in a.presets:
        name=preset.stem
        subprocess.run(base+['stop','--deployment-dir',str(a.deployment_dir)],check=True)
        subprocess.run(base+['start','--deployment-dir',str(a.deployment_dir),'--preset',str(preset)],check=True)
        run=a.output_root/name
        subprocess.run([sys.executable,str(R/'bench_probe.py'),'--label',name,'--output',str(run),'--preset',str(preset),'--tokens','96'],check=True)
        (run/'actual-launch.local.json').write_text((a.deployment_dir/'server-process.json').read_text(encoding='utf-8'),encoding='utf-8')
        print('COMPLETED '+name,flush=True)
        last_good=preset
except Exception:
    if last_good:
        print('Restoring last successful preset: '+str(last_good),flush=True)
        subprocess.run(base+['stop','--deployment-dir',str(a.deployment_dir)],check=True)
        subprocess.run(base+['start','--deployment-dir',str(a.deployment_dir),'--preset',str(last_good)],check=True)
    raise
