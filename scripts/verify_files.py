"""Verify separately downloaded runtime/model files; streams rather than loading weights."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--directory', type=Path, required=True)
p.add_argument('--manifest', type=Path, default=Path(__file__).resolve().parents[1]/'manifests/downloads.json')
p.add_argument('--name', action='append', help='Verify only the named manifest entry; repeatable')
a = p.parse_args()
rows = json.loads(a.manifest.read_text(encoding='utf-8'))
if a.name:
    unknown = set(a.name)-{r['name'] for r in rows}
    if unknown: p.error('Unknown manifest file: '+repr(unknown))
    rows = [r for r in rows if r['name'] in a.name]
for row in rows:
    path = a.directory/row['name']
    if path.stat().st_size != row['bytes']: raise SystemExit('Size mismatch: '+path.name)
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024), b''): h.update(block)
    if h.hexdigest().lower() != row['sha256'].lower(): raise SystemExit('SHA256 mismatch: '+path.name)
    print('VERIFIED '+path.name)
