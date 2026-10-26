"""Build the pinned GenAI dense-tail correction into an isolated Windows package.

Requires Git, CMake >=3.26, VS2022 C++ tools and the OpenVINO experiment venv.
Builds CPU code only. Run this before GPU timing, not during timed inference.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def run(command,**kwargs):
    print(subprocess.list2cmdline([str(x) for x in command]),flush=True)
    return subprocess.run([str(x) for x in command],check=True,**kwargs)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--deployment-dir',type=Path,required=True)
    p.add_argument('--jobs',type=int,default=4)
    a=p.parse_args()
    if os.name!='nt' or a.jobs<1:p.error('Windows and positive jobs required')
    manifest=json.loads((ROOT/'manifests/openvino-genai-dense-tail.json').read_text())
    patch=ROOT/'manifests'/manifest['patch']
    if sha(patch)!=manifest['patch_sha256']:raise ValueError('Patch hash mismatch')
    dep=a.deployment_dir.resolve();toolchain=dep/'toolchain';toolchain.mkdir(exist_ok=True)
    source=toolchain/'openvino-genai-dense-tail';build=toolchain/'build-openvino-genai-dense-tail'
    python=dep/'openvino-venv/Scripts/python.exe'
    site=dep/'openvino-venv/Lib/site-packages'
    if not python.is_file() or not (site/'openvino/cmake').is_dir():raise ValueError('Install the experiment venv first')
    if not source.exists():
        run(['git','clone','--no-checkout',manifest['repository'],source])
        run(['git','-C',source,'checkout','--detach',manifest['commit']])
    head=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if head!=manifest['commit']:raise ValueError('Existing source is at a different commit')
    status=subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True)
    if not status:run(['git','-C',source,'apply',patch])
    changed={line[3:] for line in subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).splitlines()}
    if changed!=set(manifest['modified_files']):raise ValueError('Unexpected source changes; preserve and inspect them')
    for name,expected in manifest['modified_files'].items():
        if hashlib.sha256((source/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=expected:raise ValueError('Source does not exactly match the pinned correction: '+name)
    cache=build/'CMakeCache.txt'
    if cache.exists():
        entries=dict(line.split('=',1) for line in cache.read_text().splitlines() if line.startswith('CMAKE_HOME_DIRECTORY:'))
        if Path(entries.get('CMAKE_HOME_DIRECTORY:INTERNAL','')).resolve()!=source:raise ValueError('Build directory belongs to another source')
    configure=['cmake','-S',source,'-B',build,'-G','Visual Studio 17 2022','-A','x64',
        '-DOpenVINO_DIR='+str(site/'openvino/cmake'),'-DPython3_EXECUTABLE='+str(python),
        '-DCMAKE_CXX_FLAGS=/DWIN32 /D_WINDOWS /W3 /GR /EHsc /utf-8',
        '-DBUILD_TOKENIZERS=OFF','-DENABLE_PYTHON=OFF','-DENABLE_SAMPLES=OFF',
        '-DENABLE_TESTS=ON','-DENABLE_TOOLS=OFF','-DENABLE_XGRAMMAR=OFF',
        '-DENABLE_GGUF=ON','-DENABLE_LTO=OFF']
    run(configure)
    run(['cmake','--build',build,'--config','Release','--target','openvino_genai','tests_continuous_batching','--parallel',a.jobs,'--','/p:UseMultiToolTask=true','/p:CL_MPCount='+str(a.jobs),'/p:EnforceProcessCountAcrossBuilds=true'])
    dlls=list(build.rglob('openvino_genai.dll'))
    tests=list(build.rglob('tests_continuous_batching.exe'))
    if len(dlls)!=1 or len(tests)!=1:raise ValueError('Ambiguous build output paths')
    dll=dlls[0]
    env={**os.environ,'PATH':str(dll.parent)+os.pathsep+str(site/'openvino/libs')+os.pathsep+os.environ['PATH']}
    log=build/'dense-tail-tests.txt'
    with log.open('wb') as file:
        run([tests[0],'--gtest_filter=TestScheduler.*'],stdout=file,stderr=subprocess.STDOUT,env=env)
    overlay=dep/('runtime-openvino-dense-tail-'+sha(dll)[:12])
    if not overlay.exists():
        shutil.copytree(site/'openvino_genai',overlay/'openvino_genai',ignore=shutil.ignore_patterns('__pycache__'))
        shutil.copyfile(dll,overlay/'openvino_genai/openvino_genai.dll')
    elif sha(overlay/'openvino_genai/openvino_genai.dll')!=sha(dll):raise ValueError('Existing overlay differs')
    command='import sys;sys.path.insert(0,sys.argv[1]);import openvino;import openvino_genai as g;print(g.__version__)'
    smoke=subprocess.check_output([str(python),'-c',command,str(overlay)],text=True)
    record={**manifest,'dll_sha256':sha(dll),'original_wheel_dll_sha256':sha(site/'openvino_genai/openvino_genai.dll'),
        'configure_arguments':[str(x) for x in configure[configure.index('-A')+2:]],
        'test_log_sha256':sha(log),'smoke_version':smoke.strip(),'xgrammar':'Disabled in isolated build; not used by these greedy JSON tasks.'}
    (overlay/'build-manifest.json').write_text(json.dumps(record,indent=2)+'\n')
    print('OVERLAY_READY',overlay,flush=True)


if __name__=='__main__':main()
