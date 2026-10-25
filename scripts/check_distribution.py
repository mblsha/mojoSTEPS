"""Ensure release archives carry notices and do not bundle SDKs/native binaries."""
from pathlib import Path
import sys,tarfile,zipfile
for p in Path(sys.argv[1]).glob('*'):
    if p.suffix=='.whl':
        with zipfile.ZipFile(p) as z: names=z.namelist()
    elif p.name.endswith('.tar.gz'):
        with tarfile.open(p) as t: names=t.getnames()
    else: continue
    for required in ('LICENSE','NOTICE','LICENSE.numpy','LICENSE.pysteps','LICENSE.randomkit','LICENSE.numpy-random','LICENSE.md'):
        assert any(n.endswith('/'+required) for n in names),(p,required)
    assert not any(n.endswith(('.so','.dylib','.o','.a','.dll')) or '/.build/' in n or '/.venv/' in n for n in names),p
    if p.name.endswith('.tar.gz'):
        assert any(n.endswith('/src/mojo/native_core.mojo') for n in names)
        assert any(n.endswith('/third_party/pocketfft/pocketfft_hdronly.h') for n in names)
    print(p.name,'notices and source-only layout OK')
