"""Matched PocketFFT/FFTW experiment; optional external FFTW, macOS only.

No default forecast source or build is changed. The same Mojo dylib loads one
FFT provider in each isolated worker process. FFTs are single-threaded; workers
control ensemble parallelism. Use --help for build/prepare/run commands.
"""
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time

# Set before importing NumPy, SciPy or pySTEPS in this standalone runner.
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import numpy as np
from mojosteps import Forecast
from mojosteps.synthetic import rain_cells
from mojosteps.validation import compare,reference_forecast
WORK=ROOT/'.build/fft-compare'
MODES={'pocketfft':None,'fftw-estimate':64,'fftw-measure':0,'fftw-patient':32}


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def digest(a): return hashlib.sha256(a.tobytes()).hexdigest()
def save(path,value): Path(path).write_text(json.dumps(value,indent=2)+'\n')
def cpu_time():
    r=resource.getrusage(resource.RUSAGE_SELF)
    return r.ru_utime+r.ru_stime


def build():
    if sys.platform!='darwin' or platform.machine()!='arm64':
        raise SystemExit('This matched-provider experiment currently targets Apple silicon macOS.')
    WORK.mkdir(parents=True,exist_ok=True)
    mojo=os.environ.get('MOJO_BIN') or str(ROOT/'.build/toolchain/bin/mojo')
    cxx=os.environ.get('CXX') or shutil.which('clang++')
    prefix=Path(os.environ.get('FFTW_PREFIX','/opt/homebrew'))
    commands=[]
    commands.append([mojo,'build','--emit','shared-lib','--fp-mode','contract=off',
                     '-Xlinker','-undefined','-Xlinker','dynamic_lookup','-Xlinker','-lc++',
                     str(ROOT/'src/mojo/native_core.mojo'),'-o',str(WORK/'core.dylib')])
    for provider,source in [('pocketfft',ROOT/'src/fft_bridge.cpp'),('fftw',ROOT/'scripts/fft_compare/fftw_bridge.cpp')]:
        command=[cxx,'-std=c++17','-O3','-fPIC','-ffp-contract=off','-dynamiclib',
                 str(source),str(ROOT/'scripts/fft_compare/micro.cpp'),'-o',str(WORK/f'{provider}.dylib')]
        if provider=='fftw':
            command += ['-I'+str(prefix/'include'),str(prefix/'lib/libfftw3.dylib')]
        commands.append(command)
    for command in commands:
        subprocess.run(command,check=True)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src','scripts/fft_compare','third_party/pocketfft')
             for p in sorted((ROOT/folder).rglob('*')) if p.is_file()}
    sources['scripts/benchmark_fft.py']=sha(__file__)
    save(WORK/'build.json',dict(source_sha256=sources,
        core_sha256=sha(WORK/'core.dylib'),provider_sha256={name:sha(WORK/f'{name}.dylib') for name in ('pocketfft','fftw')},
        mojo_version=subprocess.check_output([mojo,'--version'],text=True).strip(),
        cxx_version=subprocess.check_output([cxx,'--version'],text=True).splitlines()[0],
        fftw_library_sha256=sha(prefix/'lib/libfftw3.dylib'),
        commands=[['$MOJO_BIN' if s==mojo else s.replace(str(ROOT),'$ROOT').replace(str(prefix),'$FFTW_PREFIX') for s in cmd] for cmd in commands]))


def prepare(fixture):
    WORK.mkdir(parents=True,exist_ok=True)
    if fixture:
        data=np.load(fixture);h,v=data['history'],data['velocity']
    else:
        h,v=rain_cells()
    np.savez(WORK/'fixture.npz',history=h,velocity=v)
    start=time.perf_counter()
    r=reference_forecast(h,v,14,members=96,workers=16,seed=20260817)
    np.save(WORK/'reference.npy',r)
    save(WORK/'input.json',dict(history_sha256=digest(h),velocity_sha256=digest(v),
        reference_sha256=digest(r),reference_prepare_seconds=time.perf_counter()-start,
        versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pysteps')}))


def worker(args):
    name=args.mode
    provider='pocketfft' if name=='pocketfft' else 'fftw'
    library=ctypes.CDLL(str(WORK/f'{provider}.dylib'),mode=ctypes.RTLD_GLOBAL)
    if provider=='fftw':
        library.bench_fftw_configure.argtypes=[ctypes.c_uint,ctypes.c_char_p]
        library.bench_fftw_configure.restype=ctypes.c_int
        library.bench_fftw_plan_seconds.restype=ctypes.c_double
        library.bench_fftw_plan_calls.restype=ctypes.c_uint64
        library.bench_fftw_export.argtypes=[ctypes.c_char_p]
        library.bench_fftw_export.restype=ctypes.c_int
        library.bench_fftw_version.restype=ctypes.c_char_p
        library.bench_fftw_plan_description.argtypes=[ctypes.c_int]
        library.bench_fftw_plan_description.restype=ctypes.c_char_p
        start=time.perf_counter()
        status=library.bench_fftw_configure(MODES[name],str(args.wisdom).encode() if args.wisdom else None)
        wisdom_import_seconds=time.perf_counter()-start
        assert status==1,'wisdom import failed'
        plan_seconds=lambda:library.bench_fftw_plan_seconds()
        plan_calls=lambda:library.bench_fftw_plan_calls()
    else:
        wisdom_import_seconds=0
        plan_seconds=lambda:0
        plan_calls=lambda:0
    data=np.load(WORK/'fixture.npz');h,v=data['history'],data['velocity']
    r=np.load(WORK/'reference.npy',mmap_mode='r')
    engine=Forecast(library=WORK/'core.dylib')
    rows=[];comparisons={}
    # First run is cold after dlopen/array preparation. No prior FFT or warmup.
    for run in range(args.runs+1):
        ps,pc=plan_seconds(),plan_calls()
        cpu_start=cpu_time();start=time.perf_counter()
        a=engine.forecast(h,v,14,members=96,workers=args.workers,seed=20260817)
        elapsed=time.perf_counter()-start;cpu_elapsed=cpu_time()-cpu_start
        planning=plan_seconds()-ps;calls=plan_calls()-pc
        code=digest(a)
        if code not in comparisons:
            result=compare(r,a);comparisons[code]=result
        else: result=comparisons[code]
        assert result['accepted'],(name,args.workers,result)
        rows.append(dict(wall_seconds=elapsed,cpu_seconds=cpu_elapsed,
                         average_process_cpu_cores=cpu_elapsed/elapsed,
                         fftw_planner_seconds_sum=planning,fftw_plan_calls=calls,output_sha256=code))
        del a
        print(f'{name} workers={args.workers} run={run}: {elapsed:.6f}s; planner sum={planning:.6f}s',file=sys.stderr,flush=True)
    warm=rows[1:]
    report=dict(mode=name,workers=args.workers,fft_threads=1,members=96,leads=14,seed=20260817,
        cold=rows[0],warm=warm,median_seconds=statistics.median(x['wall_seconds'] for x in warm) if warm else None,
        comparisons=comparisons,repetitions_byte_identical=len(comparisons)==1,
        wisdom_imported=bool(args.wisdom),wisdom_only=bool(args.wisdom),wisdom_import_seconds=wisdom_import_seconds,
        hardware=dict(processor=subprocess.check_output(['sysctl','-n','machdep.cpu.brand_string'],text=True).strip(),
                      physical_cpu_cores=int(subprocess.check_output(['sysctl','-n','hw.physicalcpu'],text=True)),
                      memory_bytes=int(subprocess.check_output(['sysctl','-n','hw.memsize'],text=True))),
        platform=platform.platform(),input=json.loads((WORK/'input.json').read_text()),
        build=json.loads((WORK/'build.json').read_text()),
        thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')})
    if provider=='fftw':
        wisdom=WORK/f'{name}-{args.workers}.wisdom'
        assert library.bench_fftw_export(str(wisdom).encode())==1
        report['wisdom_sha256']=sha(wisdom)
        report['fftw_version']=library.bench_fftw_version().decode()
        report['fftw_plans']={str(i):library.bench_fftw_plan_description(i).decode() for i in (0,1)}
    # Microbenchmark AFTER the forecasts, so it cannot pre-seed cold FFTW wisdom.
    micro=library.bench_fft_pairs
    pointer=ctypes.POINTER(ctypes.c_double)
    micro.argtypes=[pointer,ctypes.c_int,pointer];micro.restype=ctypes.c_double
    source=np.ascontiguousarray(np.random.RandomState(7).normal(size=(192,192)))
    error=np.zeros(1)
    address=lambda x:x.ctypes.data_as(pointer)
    micro(address(source),10,address(error))
    batches=[micro(address(source),1000,address(error))/1000 for _ in range(5)]
    assert error[0]<1e-9,error[0]
    report['micro_roundtrip']=dict(pairs_per_batch=1000,warm_batches=5,seconds_per_pair=batches,
        median_seconds_per_pair=statistics.median(batches),last_batch_max_error=float(error[0]),
        definition='192x192 r2c + c2r + normalization; planning excluded; one FFT thread')
    save(args.output,report)


def run(args):
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    jobs=[(w,m) for w in args.workers for m in MODES]
    random.Random(20261025).shuffle(jobs)
    schedule=[]
    for workers,mode in jobs:
        path=output/f'{mode}-{workers}.json'
        command=[sys.executable,__file__,'worker','--mode',mode,'--workers',str(workers),
                 '--runs',str(args.runs),'--output',str(path)]
        subprocess.run(command,check=True,timeout=600)
        schedule.append(dict(mode=mode,workers=workers,receipt=path.name))
        save(output/'schedule.json',schedule)
    for mode in ('fftw-measure','fftw-patient'):
        workers=max(args.workers)
        path=output/f'{mode}-{workers}-wisdom-restart.json'
        command=[sys.executable,__file__,'worker','--mode',mode,'--workers',str(workers),
                 '--runs','0','--wisdom',str(WORK/f'{mode}-{workers}.wisdom'),'--output',str(path)]
        subprocess.run(command,check=True,timeout=600)
    print(output,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('build')
    prepare_parser=sub.add_parser('prepare');prepare_parser.add_argument('--fixture',type=Path)
    run_parser=sub.add_parser('run')
    run_parser.add_argument('--workers',type=int,nargs='+',default=[1,4,16])
    run_parser.add_argument('--runs',type=int,default=5)
    run_parser.add_argument('--output',type=Path,default=WORK/'results')
    w=sub.add_parser('worker')
    w.add_argument('--mode',choices=MODES,required=True)
    w.add_argument('--workers',type=int,required=True)
    w.add_argument('--runs',type=int,default=5)
    w.add_argument('--output',type=Path,required=True)
    w.add_argument('--wisdom',type=Path)
    args=p.parse_args()
    if args.command=='build':build()
    elif args.command=='prepare':prepare(args.fixture)
    elif args.command=='run':run(args)
    else:worker(args)

if __name__=='__main__': main()
