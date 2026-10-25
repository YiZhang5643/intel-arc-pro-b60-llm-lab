"""Download a public Hugging Face OpenVINO snapshot at a pinned 40-hex revision.

Uses HTTP ranges and checks advertised sizes plus LFS SHA256. Partial ranged files
are restarted on the next invocation; already verified complete files are reused.
"""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request


def sha256(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='OpenVINO/Qwen3.8-27B-int4-ov')
    p.add_argument('--revision',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=8)
    a=p.parse_args()
    if not re.fullmatch(r'[0-9a-fA-F]{40}',a.revision):p.error('Use a pinned full commit SHA')
    if not re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',a.model):p.error('Invalid public repository ID')
    if not 1<=a.workers<=16:p.error('workers must be 1..16')
    api=f'https://huggingface.co/api/models/{a.model}/revision/{a.revision}?blobs=true'
    with urllib.request.urlopen(api,timeout=60) as r:meta=json.load(r)
    if meta['sha']!=a.revision.lower():raise RuntimeError('Revision mismatch')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    jobs=[];pending=[];files=[];chunk_size=32*1024*1024
    for f in meta['siblings']:
        name=f['rfilename']
        if name=='.gitattributes':continue
        dest=(out/name).resolve()
        if not dest.is_relative_to(out):raise RuntimeError('Invalid remote file path')
        expected=f.get('lfs',{}).get('sha256')
        if dest.exists():
            actual=sha256(dest)
            if dest.stat().st_size!=f['size'] or (expected and actual!=expected):
                raise RuntimeError('Existing file differs: '+name)
            files.append({'file':name,'size':f['size'],'sha256':actual});continue
        dest.parent.mkdir(parents=True,exist_ok=True)
        temp=dest.with_suffix(dest.suffix+'.ranged.part')
        with temp.open('wb') as w:w.truncate(f['size'])
        pending.append((f,temp,dest))
        for start in range(0,f['size'],chunk_size):jobs.append((f,temp,start,min(start+chunk_size,f['size'])-1))
    def fetch(job):
        f,temp,start,end=job
        for attempt in range(4):
            try:
                name=urllib.parse.quote(f['rfilename'],safe='/')
                url=f'https://huggingface.co/{a.model}/resolve/{a.revision}/{name}?download=true&nonce={time.time_ns()}'
                req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}'})
                with urllib.request.urlopen(req,timeout=120) as src:
                    if src.status!=206 or src.headers.get('Content-Range')!=f"bytes {start}-{end}/{f['size']}":
                        raise RuntimeError('Unexpected range response')
                    data=src.read()
                if len(data)!=end-start+1:raise RuntimeError('Incomplete range')
                with temp.open('r+b') as w:w.seek(start);w.write(data)
                return
            except Exception:
                if attempt==3:raise
                time.sleep(1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:list(pool.map(fetch,jobs))
    for f,temp,dest in pending:
        actual=sha256(temp);expected=f.get('lfs',{}).get('sha256')
        if temp.stat().st_size!=f['size'] or (expected and actual!=expected):raise RuntimeError('Downloaded hash mismatch: '+f['rfilename'])
        temp.replace(dest);files.append({'file':f['rfilename'],'size':f['size'],'sha256':actual})
        print('Verified',f['rfilename'],flush=True)
    (out/'snapshot-manifest.json').write_text(json.dumps({'repo':a.model,'revision':a.revision,'files':files},indent=2)+'\n',encoding='utf-8')
    print('Verified snapshot',a.model,a.revision,flush=True)


if __name__=='__main__':main()
