"""Build the source-only mojoSTEPS CPU and/or Apple GPU libraries."""
from __future__ import annotations
import argparse, hashlib, json, os, platform, shutil, subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]


def build(backend):
    output=ROOT/'.build'
    output.mkdir(exist_ok=True)
    mojo=os.environ.get('MOJO_BIN') or str(output/'toolchain/bin/mojo')
    if not Path(mojo).is_file():
        mojo=shutil.which(mojo)
    if not mojo:
        raise SystemExit('Mojo not found. Run scripts/bootstrap.sh or set MOJO_BIN.')
    cxx=os.environ.get('CXX') or shutil.which('clang++') or shutil.which('g++')
    if not cxx:
        raise SystemExit('A C++17 compiler is required.')
    suffix='dylib' if platform.system()=='Darwin' else 'so'
    cpp=output/'fft_bridge.o'
    subprocess.run([cxx,'-std=c++17','-O3','-fPIC','-ffp-contract=off','-c',str(ROOT/'src/fft_bridge.cpp'),'-o',str(cpp)],check=True)
    targets=[('cpu','native_core'),('primitives','probe_native')] if backend=='cpu' else [('gpu','gpu_resident')]
    if backend=='all':
        targets=[('cpu','native_core'),('primitives','probe_native'),('gpu','gpu_resident')]
    sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'src').rglob('*')) if p.is_file()}
    sources.update({str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((ROOT/'third_party/pocketfft').glob('*')) if p.is_file()})
    for name,source in targets:
        if name=='gpu' and (platform.system()!='Darwin' or platform.machine()!='arm64'):
            raise SystemExit('The GPU backend is qualified only on Apple silicon macOS. Use --backend cpu.')
        library=output/f'libmojosteps_{name}.{suffix}'
        cmd=[mojo,'build','--emit','shared-lib','--fp-mode','contract=off','-Xlinker',str(cpp),'-Xlinker',('-lc++' if platform.system()=='Darwin' else '-lstdc++'),str(ROOT/'src/mojo'/f'{source}.mojo'),'-o',str(library)]
        subprocess.run(cmd,check=True)
        manifest={'library':library.name,'sha256':hashlib.sha256(library.read_bytes()).hexdigest(),'source_sha256':sources,'fft':'PocketFFT (vendored BSD-3-Clause)','command':[str(Path(arg).relative_to(ROOT)) if arg.startswith(str(ROOT)+'/') else arg for arg in cmd],'mojo_version':subprocess.check_output([mojo,'--version'],text=True).strip()}
        library.with_suffix('.manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(library,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend',choices=['cpu','gpu','all'],default='cpu')
    build(p.parse_args().backend)
