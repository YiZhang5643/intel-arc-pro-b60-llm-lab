"""Verify published evidence hashes and the central report numbers."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

R=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--directory',type=Path);a=p.parse_args()
E=a.directory.resolve() if a.directory else R/'results/2026-10-24-nemotron'
manifest=json.loads((E/'SHA256SUMS.json').read_text(encoding='utf-8'))
for name,digest in manifest.items():
    path=(E/name).resolve()
    if not path.is_relative_to(E.resolve()):raise ValueError('Invalid evidence path')
    if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise ValueError('Evidence changed: '+name)
if a.directory:
    print(f'Verified {len(manifest)} evidence files');raise SystemExit(0)
rows=list(csv.DictReader((E/'measurements.csv').open(encoding='utf-8',newline='')))
def get(label,test):return next(r for r in rows if r['label']==label and r['test']==test)
a=get('prefill-final-ub8192','cold-98304');b=get('official-sycl','cold-98304')
assert round(float(a['prefill_tps']))==1184
assert round(float(a['ttft_seconds']),2)==83.30
assert round(float(b['wall_seconds']),2)==218.00
for label in ['long-mtp-off-ub2048','long-mtp-on-ub2048']:
    assert int(get(label,'cold-retrieval')['input_tokens'])==98304
    assert int(get(label,'cold-retrieval')['cache_n'])==0
    for test in ['cached-retrieval','cached-code','cached-prose']:
        assert int(get(label,test)['cache_n'])>=98000
    for test in ['cold-retrieval','cached-retrieval']:
        assert int(get(label,test)['retrieval_correct'])==3
print(f'Verified {len(manifest)} evidence files and key report values')
