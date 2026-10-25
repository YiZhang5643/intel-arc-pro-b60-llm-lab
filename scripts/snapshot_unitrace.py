"""Create a new, path-scrubbed snapshot of explicitly selected synthetic unitrace runs."""
import argparse
import csv
from datetime import datetime
import gzip
import hashlib
import io
import json
from pathlib import Path, PureWindowsPath
import re
import shutil

from analyze_unitrace import analyze


def write(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def clean_text(text):
    return '\n'.join(s.rstrip() for s in text.splitlines()).rstrip()+'\n'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def memory_samples(src, profile, pid):
    result=[]
    path=src/'gpu-memory.jsonl'
    if not path.exists():
        return result
    for line in path.read_text(encoding='utf-8').splitlines():
        row=json.loads(line)
        stamp=datetime.fromisoformat(row['utc']).timestamp()*1e9
        if not profile['client_start_unix_ns'] <= stamp <= profile['client_end_unix_ns']:
            continue
        prefix=f'pid_{pid}_'
        counters=[c for c in row['counters'] if c['instance'].startswith(prefix)]
        dedicated=[c for c in counters if c['path'].endswith('\\dedicated usage')]
        if not dedicated:
            continue
        main=max(dedicated, key=lambda c:c['value'])
        shared=next((c['value'] for c in counters if c['instance']==main['instance']
                     and c['path'].endswith('\\shared usage')),None)
        result.append({'utc':row['utc'], 'dedicated_bytes':main['value'], 'shared_bytes':shared})
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--labels',nargs='+',required=True)
    p.add_argument('--model-metadata',type=Path,required=True)
    p.add_argument('--unitrace-dir',type=Path,required=True)
    p.add_argument('--unitrace-commit',required=True)
    a=p.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    source_meta=json.loads(a.model_metadata.read_text(encoding='utf-8'))
    metadata={'analysis_date':'2026-10-25', 'hardware':source_meta['hardware'],
              'runtime':source_meta['runtime'], 'model':source_meta['model'],
              'unitrace':{'version':'2.4.1', 'commit':a.unitrace_commit,
                          'repository':'https://github.com/intel/pti-gpu',
                          'configuration':'Level Zero only; no XPTI, ITT, OpenCL, MPI, OMP or Perfetto protobuf',
                          'sha256':{n:sha(a.unitrace_dir/n) for n in ['unitrace.exe','unitrace_tool.dll']}},
              'method':'One cold request per arm, n_predict=1, exact saved synthetic token arrays, no prefix reuse. 4096-token warmup while collection paused.',
              'trace_sanitization':'Machine labels HOST<...> and DEVICE<...> changed to local. Timestamps, durations, operation names and dependencies preserved. gzip is deterministic.',
              'limitations':['No hardware XMX occupancy, bandwidth, frequency, power or stall counters',
                             'No repeated timing distribution or controlled thermal/power study',
                             '96K has no new unprofiled paired request; compare prior saved run only',
                             'Timing-only one-token output is not a model quality test',
                             'Memory counters are samples; do not treat observed maximum as instrumented peak'],
              'arms':{}}
    measurements=[]
    for label in a.labels:
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,79}',label):
            raise ValueError('Invalid label')
        src=a.input_root/label
        dst=a.output/label
        dst.mkdir()
        trace_path=next(src.glob('llama-server.*.json'))
        raw=trace_path.read_bytes()
        trace=json.loads(raw)
        profile=json.loads((src/'profile.json').read_text())
        analysis,rows=analyze(trace,profile)
        write(dst/'analysis.json',analysis)
        write(dst/'profile.json',profile)
        for name in ['input.json','baseline.json','baseline-warmup.json','profile-warmup.json']:
            if (src/name).exists():
                shutil.copyfile(src/name,dst/name)
        for e in trace['traceEvents']:
            if e.get('ph')=='M' and e.get('name')=='process_name':
                e['args']['name']=re.sub(r'(HOST|DEVICE)<[^>]+>',r'\1<local>',e['args']['name'])
        sanitized=json.dumps(trace,separators=(',',':')).encode()
        (dst/'trace.json.gz').write_bytes(gzip.compress(sanitized,compresslevel=9,mtime=0))
        timing=next(src.glob('device-timing.*.txt'))
        (dst/'device-timing.txt').write_text(clean_text(timing.read_text(encoding='utf-8')),encoding='utf-8')
        with io.StringIO(newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
            (dst/'kernels.csv.gz').write_bytes(gzip.compress(f.getvalue().encode(),compresslevel=9,mtime=0))
        launch=json.loads((src/'launch.local.json').read_text())
        start=next(i for i,v in enumerate(launch['command']) if PureWindowsPath(v).name.lower()=='llama-server.exe')
        args=launch['command'][start:]
        safe=[PureWindowsPath(v).name if PureWindowsPath(v).is_absolute() else v for v in args]
        state=json.loads((src/'profile-server.local.json').read_text())
        env={**launch['environment_overrides'],**launch['environment_changes']}
        config={'server_args_paths_replaced_by_filenames':safe,'environment_overrides':env,
                'post_load_working_set_trim':state.get('postload_working_set_trim',False), 'raw_trace_sha256':hashlib.sha256(raw).hexdigest()}
        write(dst/'config.json',config)
        text=(src/'server.log').read_text(encoding='utf-8',errors='replace')
        text=re.sub(r'[A-Za-z]:\\[^\s\"\x27]+',lambda m:PureWindowsPath(m[0]).name,text)
        (dst/'server-startup-and-request.txt').write_text(clean_text(text),encoding='utf-8')
        mem=memory_samples(src,profile,state['pid'])
        if mem:
            write(dst/'gpu-memory-samples.json',mem)
        metadata['arms'][label]={'tokens':profile['timings']['prompt_n'],
                                 'raw_trace_sha256':config['raw_trace_sha256'],
                                 'memory_sample_count':len(mem)}
        if mem:
            metadata['arms'][label]['observed_max_dedicated_bytes']=max(r['dedicated_bytes'] for r in mem)
            metadata['arms'][label]['observed_max_shared_bytes']=max(r['shared_bytes'] for r in mem if r['shared_bytes'] is not None)
        measurements.append({'label':label,'tokens':profile['timings']['prompt_n'],
                             'cache_n':profile['timings']['cache_n'],
                             'prompt_seconds':profile['timings']['prompt_ms']/1000,
                             'prompt_tok_s':profile['timings']['prompt_per_second'],
                             'client_wall_seconds':analysis['client_wall_seconds'],
                             'kernel_union_percent':analysis['kernel_union_percent']})
    for suffix in ['png','svg']:
        shutil.copyfile(a.input_root/f'kernel-timeline.{suffix}',a.output/f'kernel-timeline.{suffix}')
    write(a.output/'metadata.json',metadata)
    with (a.output/'measurements.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(measurements[0]));w.writeheader();w.writerows(measurements)
    write(a.output/'SHA256SUMS.json',{x.relative_to(a.output).as_posix():sha(x) for x in sorted(a.output.rglob('*')) if x.is_file()})
    print('Snapshot created. Review it before a user-authorized publication.')
