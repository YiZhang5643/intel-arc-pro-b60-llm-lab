"""Fetch checksum-pinned source archives and apply the two vendored Intel patches."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import urllib.request

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    args=parser.parse_args()
    root=args.root.resolve()
    if root.exists():
        parser.error('Use a new directory; existing trees are never overwritten')
    sources=json.loads((ROOT/'manifests/vulkan-toolchain-sources.json').read_text())
    locations={'google/shaderc':'shaderc','KhronosGroup/glslang':'shaderc/third_party/glslang','KhronosGroup/SPIRV-Headers':'shaderc/third_party/spirv-headers','KhronosGroup/SPIRV-Tools':'shaderc/third_party/spirv-tools','KhronosGroup/Vulkan-Headers':'Vulkan-Headers','ggml-org/llama.cpp':'llama-intelfa-b11869'}
    root.mkdir(parents=True)
    for item in sources:
        print('Downloading',item['repo'],item['ref'],flush=True)
        data=urllib.request.urlopen(item['url'],timeout=180).read()
        if hashlib.sha256(data).hexdigest()!=item['archive_sha256']:
            raise RuntimeError('Archive checksum mismatch: '+item['repo'])
        dest=root/locations[item['repo']]
        with tarfile.open(fileobj=io.BytesIO(data)) as tf:
            for member in tf.getmembers():
                parts=Path(member.name).parts[1:]
                if not parts: continue
                target=dest.joinpath(*parts).resolve()
                if not target.is_relative_to(dest.resolve()) or member.issym() or member.islnk():
                    raise RuntimeError('Unsafe archive member: '+member.name)
                if member.isdir(): target.mkdir(parents=True,exist_ok=True)
                elif member.isfile():
                    target.parent.mkdir(parents=True,exist_ok=True)
                    target.write_bytes(tf.extractfile(member).read())
    patches=ROOT/'manifests/intel-vulkan-fa'
    for line in (patches/'SHA256SUMS').read_text().splitlines():
        digest,name=line.split()
        path=patches/name
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
            raise RuntimeError('Patch checksum mismatch')
        subprocess.run(['git','-C',str(root/'llama-intelfa-b11869'),'apply','--check',str(path)],check=True)
        subprocess.run(['git','-C',str(root/'llama-intelfa-b11869'),'apply',str(path)],check=True)
    diagnostic=patches/'0003-local-prefill-dispatch-diagnostic.patch'
    if hashlib.sha256(diagnostic.read_bytes()).hexdigest()!=(patches/'LOCAL_SHA256SUMS').read_text().split()[0]:
        raise RuntimeError('Diagnostic patch checksum mismatch')
    subprocess.run(['git','-C',str(root/'llama-intelfa-b11869'),'apply','--check',str(diagnostic)],check=True)
    subprocess.run(['git','-C',str(root/'llama-intelfa-b11869'),'apply',str(diagnostic)],check=True)
    (root/'source-manifest.json').write_text(json.dumps(sources,indent=2)+'\n')
    print('Prepared',root)

if __name__=='__main__':
    main()
