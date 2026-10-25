"""Create a reviewable benchmark snapshot from explicitly selected synthetic test runs."""
import argparse,hashlib,json,re
from pathlib import Path,PureWindowsPath
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--labels',nargs='+',required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=False)
def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8',newline='\n') as f:f.write(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
for label in a.labels:
    if not re.fullmatch('[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',label):p.error('Invalid label')
    src=a.input_root/label;dst=a.output/label
    if not src.is_dir():raise FileNotFoundError(src)
    for name in ['summary.json','prefill-summary.json','short-summary.json','prompt-hashes.json','run-metadata.json','failure.json','check-analysis.json','arithmetic.json','precedence.json','coding.json']:
        path=src/name
        if path.exists():
            obj=json.loads(path.read_text(encoding='utf-8-sig'))
            write(dst/name,obj)
    config=src/'actual-launch.local.json'
    if config.exists():
        state=json.loads(config.read_text(encoding='utf-8'))
        def sanitize(v):
            if PureWindowsPath(v).is_absolute():return PureWindowsPath(v).name
            if v.startswith('/'):return Path(v).name
            return v
        write(dst/'config.json',{'command_with_paths_replaced_by_filenames':[sanitize(x) for x in state['args']],
              'environment_overrides':state.get('environment_overrides',{}),
              'postload_working_set_trim':state.get('postload_working_set_trim','not recorded')})
manifest={x.relative_to(a.output).as_posix():hashlib.sha256(x.read_bytes()).hexdigest() for x in sorted(a.output.rglob('*')) if x.is_file()}
write(a.output/'SHA256SUMS.json',manifest)
print(f'Created {len(manifest)} evidence files. Inspect contents before publishing; only selected synthetic benchmarks belong in this snapshot.')
