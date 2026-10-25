"""Opt-in checks executed on real Metal hardware, never emulated by hosted CI."""
import ctypes
import os
import numpy as np
import pytest
from mojosteps import Forecast
pytestmark=[pytest.mark.gpu,pytest.mark.skipif(os.environ.get("MOJOSTEPS_TEST_GPU")!="1",reason="requires real Apple GPU opt-in")]
PTR=ctypes.POINTER(ctypes.c_double)
def address(a): return a.ctypes.data_as(PTR)

@pytest.mark.parametrize("seed", [0,1,20260817,0xffffffff])
def test_gpu_seed_chain(seed):
    probe=Forecast("gpu")._library.steps_gpu_seed_probe
    probe.argtypes=[ctypes.c_uint32,ctypes.c_ssize_t,PTR]
    actual=np.empty((96,627));probe(seed,96,address(actual))
    next_seed=seed
    for m in range(96):
        precip=np.random.RandomState(next_seed);next_seed=precip.randint(1000000000)
        motion=np.random.RandomState(next_seed);next_seed=motion.randint(1000000000)
        expected=motion.laplace(scale=1/np.sqrt(2),size=2)
        state=precip.get_state()
        assert actual[m,0]==state[2]
        assert np.array_equal(actual[m,3:],state[1])
        np.testing.assert_allclose(actual[m,1:3],expected,atol=2e-12,rtol=0)

@pytest.mark.parametrize("case", ["empty","full","edge","continuous","tied"])
def test_gpu_cdf_and_mask(case):
    from scipy.ndimage import generate_binary_structure,iterate_structure
    from pysteps.postprocessing.probmatching import nonparam_match_empirical_cdf
    from pysteps.nowcasts.utils import compute_dilated_mask
    field=np.full((192,192),-10.)
    if case=="full": field[:]=15
    elif case=="edge": field[0,0]=field[96,96]=field[-1,-1]=15
    elif case in ("continuous","tied"):
        field[40:80,50:90]=np.random.RandomState(135).uniform(10,30,size=(40,40))
    target=np.linspace(-10,40,192*192).reshape(192,192)
    if case=="tied": target=np.round(target/5)*5
    target=np.ascontiguousarray(np.sort(target,axis=None))
    actual=np.empty((2,192,192))
    probe=Forecast("gpu")._library.steps_gpu_fields_probe
    probe.argtypes=[PTR]*3;probe.restype=ctypes.c_int32
    assert probe(address(field),address(target),address(actual))==0
    expected=nonparam_match_empirical_cdf(field,target.reshape(192,192))
    np.testing.assert_allclose(actual[0],expected,atol=1e-10,rtol=0)
    kernel=iterate_structure(generate_binary_structure(2,1),4)
    expected=compute_dilated_mask(field>=10,kernel,10)
    np.testing.assert_allclose(actual[1],expected,atol=1e-13,rtol=0,equal_nan=True)


@pytest.mark.parametrize("size", [64, 128, 192, 256, 512])
@pytest.mark.parametrize("case", ["random", "tied", "low_components"])
def test_gpu_merge_path_order(size, case):
    """Check rank/index ordering across tiles, partial merges and members."""
    rng = np.random.RandomState(410)
    if case == "random":
        fields = rng.normal(size=(2, size, size))
    elif case == "tied":
        fields = rng.randint(-5, 6, size=(2, size, size)).astype(np.float64)
    else:
        # Every key has the same float32 high component. The paired-float low
        # component, followed by original pixel index, must determine rank.
        fields = 1.0 + rng.randint(-100, 101, size=(2, size, size)) * 2.0**-40
    fields = np.ascontiguousarray(fields)
    actual = np.empty((2, 2, size, size))
    probe = Forecast("gpu")._library.steps_gpu_sort_probe
    probe.argtypes = [PTR, PTR, ctypes.c_ssize_t, ctypes.c_ssize_t]
    probe.restype = ctypes.c_int32
    assert probe(address(fields), address(actual), size, 2) == 0
    for member in range(2):
        values = fields[member].ravel()
        order = np.lexsort((np.arange(size * size), values))
        hi = values.astype(np.float32).astype(np.float64)
        packed = hi + (values - hi).astype(np.float32).astype(np.float64)
        np.testing.assert_array_equal(actual[0, member].ravel(), packed[order])
        np.testing.assert_array_equal(actual[1, member].ravel(), order)
