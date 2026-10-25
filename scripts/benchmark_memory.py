"""Measure process RAM and Metal resources for matched full-size forecasts on macOS."""
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations
import argparse
import contextlib
import ctypes
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ[key]='1'
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
import numpy as np
WORK=ROOT/'.build/memory'
FIELDS=('rss_current','rss_lifetime_peak','physical_footprint_current','physical_footprint_lifetime_peak',
        'metal_allocated','compressed','graphics_footprint','graphics_nofootprint')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def digest(a):
    # Spatial planes are contiguous even if pySTEPS swaps member/lead axes.
    # Never call a.tobytes(): that adds a whole-forecast allocation to the peak.
    result=hashlib.sha256()
    for member in a:
        for lead in member:
            assert lead.flags.c_contiguous
            result.update(memoryview(lead).cast('B'))
    return result.hexdigest()


class Observer:
    def __init__(self,metal):
        self.lib=ctypes.CDLL(str(WORK/'probe.dylib'))
        ptr=ctypes.POINTER(ctypes.c_uint64)
        self.lib.mem_init.argtypes=[ctypes.c_int]
        self.lib.mem_snapshot.argtypes=[ptr]
        self.lib.mem_stop.argtypes=[ptr];self.lib.mem_stop.restype=ctypes.c_uint64
        self.lib.mem_device_name.restype=ctypes.c_char_p
        self.lib.mem_control.argtypes=[ptr]
        self.device_count=self.lib.mem_init(int(metal))
        if metal:assert self.device_count>0
        self.buffer=(ctypes.c_uint64*8)()
    def snapshot(self):
        assert self.lib.mem_snapshot(self.buffer)==0
        return dict(zip(FIELDS,self.buffer))
    def start(self):self.lib.mem_start()
    def stop(self):
        count=self.lib.mem_stop(self.buffer)
        return dict(zip(FIELDS,self.buffer)),count


def build():
    WORK.mkdir(exist_ok=True,parents=True)
    command=['clang++','-std=c++17','-O2','-dynamiclib','-fobjc-arc','-framework','Foundation',
             '-framework','Metal',str(ROOT/'scripts/memory_probe/probe.mm'),'-o',str(WORK/'probe.dylib')]
    subprocess.run(command,check=True)
    save(WORK/'probe-build.json',dict(source_sha256=sha(ROOT/'scripts/memory_probe/probe.mm'),
         observer_sha256=sha(WORK/'probe.dylib'),command=[x.replace(str(ROOT),'$ROOT') for x in command],
         cxx_version=subprocess.check_output(['clang++','--version'],text=True).splitlines()[0]))


def worker(args):
    observer=Observer(args.backend=='gpu')
    before_setup=observer.snapshot()
    data=np.load(args.fixture);history=data['history'];velocity=data['velocity']
    inputs={name:hashlib.sha256(value.tobytes()).hexdigest() for name,value in [('history',history),('velocity',velocity)]}
    # These are the already-qualified public synthetic receipts; no large oracle
    # arrays or numerical-comparison temporaries enter the measured process.
    accepted=json.loads((ROOT/'benchmarks/m5-gpu.json').read_text())
    assert inputs==accepted['input_sha256'],inputs
    native=None
    if args.backend=='pysteps':
        with (args.output.parent/'pysteps-import.log').open('w') as log,contextlib.redirect_stdout(log):
            from pysteps.nowcasts.steps import forecast
        def call():
            return forecast(history,velocity,args.leads,n_ens_members=args.members,n_cascade_levels=6,
                            precip_thr=10.,norain_thr=-10.,kmperpixel=.5,timestep=5.,
                            num_workers=args.workers,seed=20260817,vel_pert_method='bps')
        expected=accepted['reference_sha256']
    else:
        from mojosteps import Forecast
        native=Forecast(args.backend)
        def call():return native.forecast(history,velocity,args.leads,members=args.members,workers=args.workers,seed=20260817)
        cpu=json.loads((ROOT/'benchmarks/m5-cpu.json').read_text())
        expected=(accepted if args.backend=='gpu' else cpu)['output_sha256']
    ready=observer.snapshot()
    rows=[]
    with (args.output.parent/(args.backend+f'-{args.workers}-forecast.log')).open('w') as log:
        for run in range(args.calls):
            gc.collect();baseline=observer.snapshot()
            observer.start();start=time.perf_counter()
            with contextlib.redirect_stdout(log):value=call()
            seconds=time.perf_counter()-start
            peaks,samples=observer.stop()
            returned=observer.snapshot()
            code=digest(value)
            if args.members==96 and args.leads==14:assert code==expected,(args.backend,code,expected)
            output_bytes=value.nbytes
            del value;gc.collect();time.sleep(.1)
            released=observer.snapshot()
            rows.append(dict(call=run,cold=run==0,seconds=seconds,baseline=baseline,peak=peaks,
                             sampling_points=samples,after_return_with_output=returned,
                             after_output_deleted_gc=released,output_bytes=output_bytes,output_sha256=code))
            print(args.backend,args.workers,run,'footprint_MiB',round(peaks['physical_footprint_lifetime_peak']/2**20,2),
                  'RSS_MiB',round(peaks['rss_lifetime_peak']/2**20,2),'Metal_MiB',round(peaks['metal_allocated']/2**20,2),flush=True)
    report=dict(backend=args.backend,workers=args.workers,members=args.members,leads=args.leads,calls=rows,
                before_setup=before_setup,ready=ready,seed=20260817,input_sha256=inputs,
                process_peak_rss_bytes=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
                process_peak_physical_footprint_bytes=max(row['peak']['physical_footprint_lifetime_peak'] for row in rows),
                peak_metal_resource_bytes=max(row['peak']['metal_allocated'] for row in rows),
                metal_device=observer.lib.mem_device_name().decode(),metal_unified=bool(observer.lib.mem_unified()),
                hardware={key:subprocess.check_output(['sysctl','-n',key],text=True).strip()
                          for key in ('machdep.cpu.brand_string','hw.physicalcpu','hw.memsize')},
                platform=platform.platform(),versions={p:importlib.metadata.version(p) for p in ('numpy','scipy','pysteps')},
                observer=json.loads((WORK/'probe-build.json').read_text()),
                runner_sha256=sha(__file__),fixture_file_sha256=sha(args.fixture),
                thread_environment={k:os.environ[k] for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS')},
                sampling_interval_target_us=1000,
                units='bytes; macOS ru_maxrss is bytes',
                notes=['Mach footprint includes charged CPU/GPU memory; Metal resource bytes overlap with that accounting.',
                       'RAM lifetime peaks are OS counters. Metal is sampled at approximately 1ms intervals.',
                       'No retained previous forecast, reference array, full-output bytes copy, or comparison temporaries.',
                       'Absolute peaks include interpreter, libraries, observer, inputs, runtime and one returned output.'])
    if native:
        report['library_sha256']=sha(native.path)
        report['source_sha256']=json.loads(native.path.with_suffix('.manifest.json').read_text())['source_sha256']
    save(args.output,report)


def run(args):
    args.output.mkdir(parents=True,exist_ok=True)
    for workers in args.workers:
        for backend in ('cpu','gpu','pysteps'):
            subprocess.run([sys.executable,__file__,'worker','--backend',backend,'--workers',str(workers),
                            '--calls',str(args.calls),'--members',str(args.members),'--leads',str(args.leads),
                            '--fixture',str(args.fixture),'--output',str(args.output/f'{backend}-{workers}.json')],check=True,timeout=300)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    sub=p.add_subparsers(dest='command',required=True)
    sub.add_parser('build')
    for name in ('worker','run'):
        a=sub.add_parser(name)
        a.add_argument('--workers',type=int,nargs='+' if name=='run' else None,default=[1,16] if name=='run' else 1)
        a.add_argument('--calls',type=int,default=2)
        a.add_argument('--members',type=int,default=96);a.add_argument('--leads',type=int,default=14)
        a.add_argument('--fixture',type=Path,default=ROOT/'.build/fft-compare/fixture.npz')
        a.add_argument('--output',type=Path,default=WORK/'results')
        if name=='worker':a.add_argument('--backend',choices=('cpu','gpu','pysteps'),required=True)
    args=p.parse_args()
    if args.command=='build':build()
    elif args.command=='run':run(args)
    else:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        worker(args)

if __name__=='__main__':main()
