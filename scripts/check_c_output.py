"""The C example must produce the same bytes as the Python/C-ABI boundary."""
from pathlib import Path
import sys
import numpy as np
p=Path(sys.argv[1])
a=np.load(p/'forecast.npz')['forecast']
b=np.fromfile(p/'c-output.f64',dtype='<f8').reshape(a.shape)
assert a.tobytes()==b.tobytes(), 'C and Python caller outputs differ'
print('C/Python output bytes match')
