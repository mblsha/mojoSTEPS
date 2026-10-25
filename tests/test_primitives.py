import ctypes
from pathlib import Path
import sys
import numpy as np
import pytest

@pytest.fixture(scope="module")
def lib():
    suffix="dylib" if sys.platform=="darwin" else "so"
    return ctypes.CDLL(str(Path(__file__).resolve().parents[1]/f".build/libmojosteps_primitives.{suffix}"))

PTR=ctypes.POINTER(ctypes.c_double)
def address(a): return a.ctypes.data_as(PTR)

@pytest.mark.parametrize("seed", [0, 1, 20260817, 0xffffffff])
def test_mt19937_normals(lib,seed):
    probe=lib.native_rng_probe
    probe.argtypes=[ctypes.c_uint32,PTR,ctypes.c_ssize_t]
    a=np.empty(10001)
    probe(seed,address(a),10000)
    rng=np.random.RandomState(seed)
    assert a[0]==rng.randint(1000000000)
    np.testing.assert_allclose(a[1:],rng.normal(size=10000),atol=2e-14,rtol=0)

def test_pocketfft_roundtrip(lib):
    probe=lib.native_fft_probe
    probe.argtypes=[PTR,PTR]
    a=np.random.RandomState(7).normal(size=(192,192));b=np.empty_like(a)
    probe(address(a),address(b))
    np.testing.assert_allclose(a,b,atol=1e-13,rtol=0)

def test_incremental_mask(lib):
    from scipy.ndimage import generate_binary_structure,iterate_structure
    from pysteps.nowcasts.utils import compute_dilated_mask
    a=np.full((192,192),-10.);a[0,0]=a[96,96]=a[-1,-1]=15
    b=np.empty_like(a)
    probe=lib.native_mask_probe;probe.argtypes=[PTR,PTR]
    probe(address(a),address(b))
    kernel=iterate_structure(generate_binary_structure(2,1),4)
    expected=compute_dilated_mask(a>=10,kernel,10)
    np.testing.assert_allclose(b,expected,atol=1e-13,rtol=0,equal_nan=True)
