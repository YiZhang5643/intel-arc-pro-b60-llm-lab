"""Run an isolated OpenVINO benchmark and restore the recorded Windows model service."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir',type=Path,required=True)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--preset',type=Path)
    p.add_argument('--device',default='GPU.1')
    p.add_argument('--dry-run',action='store_true',help='Print resolved parameters without stopping the server or creating output')
    p.add_argument('--prompts',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--python',type=Path,help='OpenVINO venv Python; defaults to deployment-dir/openvino-venv/Scripts/python.exe')
    p.add_argument('--mtp',choices=['on','off'],default='on')
    p.add_argument('--batch',type=int,default=8192)
    p.add_argument('--cache-gb',type=int,default=5)
    p.add_argument('--draft-kv-blocks',type=int)
    p.add_argument('--kv-blocks',type=int)
    p.add_argument('--dq-group',type=int,default=32)
    p.add_argument('--xattention-threshold',type=float,help='Enable GPU XAttention; 1.0 retains all blocks')
    p.add_argument('--repeats',type=int,default=1)
    p.add_argument('--skip-quality',action='store_true')
    p.add_argument('--quality-in-last-timing',action='store_true',help='Use the last cold timing for retrieval validation, avoiding a duplicate long prefill')
    p.add_argument('--timeout',type=int,default=1800)
    a=p.parse_args()
    if a.preset:
        preset=json.loads(a.preset.read_text())
        if preset.get('backend')!='openvino-genai':p.error('Expected an OpenVINO preset')
        explicit={x.split('=',1)[0] for x in sys.argv[1:] if x.startswith('--')}
        for key in ['device','mtp','batch','cache_gb','kv_blocks','draft_kv_blocks','dq_group','xattention_threshold']:
            if key in preset and '--'+key.replace('_','-') not in explicit:setattr(a,key,preset[key])
    if a.kv_blocks is not None and a.kv_blocks<=0:p.error('kv-blocks must be positive')
    if a.xattention_threshold is not None:
        if not 0<a.xattention_threshold<=1 or a.mtp=='on':p.error('This experiment requires MTP off and XAttention threshold (0,1]')
    if a.skip_quality and a.quality_in_last_timing:p.error('Choose either skipped quality or quality in last timing')
    if min(a.batch,a.cache_gb,a.repeats,a.timeout)<=0 or a.dq_group<0:p.error('Invalid capacity, repeat count, timeout or quantization group')
    dep=a.deployment_dir.resolve()
    python=(a.python or dep/'openvino-venv/Scripts/python.exe').resolve()
    if not python.is_file():p.error('OpenVINO Python executable is missing')
    if not a.model.is_dir() or any(not x.is_file() for x in a.prompts):p.error('Model directory and prompt files must exist')
    max_input=max(len(json.loads(path.read_text())['cases']['cold-retrieval']['tokens']) for path in a.prompts)
    needed=max_input+(1 if a.skip_quality else 256)
    block_tokens=256 if a.xattention_threshold is not None else 16
    if a.kv_blocks and a.kv_blocks*block_tokens<needed:p.error('KV capacity is smaller than prompt plus requested output')
    if a.draft_kv_blocks and a.draft_kv_blocks*16<needed:p.error('Draft KV capacity is too small')
    if a.dry_run:
        print(json.dumps(vars(a),indent=2,default=str));return
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
    if a.preset:(out/'preset-used.json').write_bytes(a.preset.read_bytes())
    snapshot=a.model/'snapshot-manifest.json'
    if snapshot.exists():(out/'model-snapshot.json').write_bytes(snapshot.read_bytes())
    benchmark=ROOT/'scripts/bench_openvino.py'
    source={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in [benchmark,Path(__file__).resolve(),ROOT/'scripts/run_gpu_probe.py']}
    (out/'source-sha256.json').write_text(json.dumps(source,indent=2)+'\n')
    source_dir=out/'source';source_dir.mkdir()
    for file in [benchmark,Path(__file__).resolve(),ROOT/'scripts/run_gpu_probe.py']:
        (source_dir/file.name).write_bytes(file.read_bytes())
    benchmark=source_dir/benchmark.name
    args=[str(benchmark),'--model',str(a.model.resolve()),'--output',str(out/'measurement'),
        '--device',a.device,'--mtp',a.mtp,'--batch',str(a.batch),'--cache-gb',str(a.cache_gb),'--dq-group',str(a.dq_group),'--repeats',str(a.repeats)]
    for path in a.prompts:args+=['--prompts',str(path.resolve())]
    if a.draft_kv_blocks is not None:args+=['--draft-kv-blocks',str(a.draft_kv_blocks)]
    if a.skip_quality:args+=['--skip-quality']
    if a.quality_in_last_timing:args+=['--quality-in-last-timing']
    if a.kv_blocks:args+=['--kv-blocks',str(a.kv_blocks)]
    if a.xattention_threshold is not None:args+=['--xattention-threshold',str(a.xattention_threshold)]
    command=[sys.executable,str(ROOT/'scripts/run_gpu_probe.py'),'--deployment-dir',str(dep),
        '--executable',str(python),'--output',str(out/'lifecycle'),'--timeout',str(a.timeout)]
    command+=['--argument='+x for x in args]
    subprocess.run(command,check=True)


if __name__=='__main__':main()
