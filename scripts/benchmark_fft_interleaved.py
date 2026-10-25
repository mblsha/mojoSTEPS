"""Rotating-order 16-worker follow-up using frozen FFTW wisdom from the matrix."""
# SPDX-License-Identifier: BSD-3-Clause
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from benchmark_fft import MODES, WORK, sha, save


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=16)
    p.add_argument('--rounds',type=int,default=5)
    p.add_argument('--output',type=Path,default=WORK/'interleaved')
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    frozen=args.output/'wisdom';frozen.mkdir(exist_ok=True)
    identities={}
    for mode in MODES:
        if mode=='pocketfft':continue
        target=frozen/f'{mode}.wisdom'
        shutil.copyfile(WORK/f'{mode}-{args.workers}.wisdom',target)
        identities[mode]=sha(target)
    schedule=[]
    names=list(MODES)
    for round_index in range(args.rounds):
        order=names[round_index%len(names):]+names[:round_index%len(names)]
        if round_index%2:order.reverse()
        for mode in order:
            output=args.output/f'round-{round_index}-{mode}.json'
            command=[sys.executable,str(Path(__file__).with_name('benchmark_fft.py')),'worker',
                     '--mode',mode,'--workers',str(args.workers),'--runs','1','--output',str(output)]
            if mode!='pocketfft': command+=['--wisdom',str(frozen/f'{mode}.wisdom')]
            subprocess.run(command,check=True,timeout=180)
            schedule.append(dict(round=round_index,mode=mode,receipt=output.name))
            save(args.output/'schedule.json',dict(worker_limit=args.workers,rounds=args.rounds,
                 script_sha256=sha(__file__),frozen_wisdom_sha256=identities,cells=schedule))
    for mode,identity in identities.items():assert sha(frozen/f'{mode}.wisdom')==identity

if __name__=='__main__':main()
