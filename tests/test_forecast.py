import numpy as np
import pytest
from mojosteps import Forecast
from mojosteps.synthetic import rain_cells
from mojosteps.validation import compare, reference_forecast

@pytest.mark.parametrize("dtype,seed", [("float32", 20260817), ("float64", 0)])
def test_reference(engine, dtype, seed):
    h, v = rain_cells(velocity_dtype=dtype)
    options = dict(members=4, workers=1, seed=seed)
    actual = engine.forecast(h, v, 3, **options)
    reference = reference_forecast(h, v, 3, **options)
    result = compare(reference, actual)
    assert result["accepted"], result

@pytest.mark.parametrize("echoes", [0, 15])
def test_persistence(engine, echoes):
    h = np.full((3, 192, 192), -10.)
    h[-1].flat[:echoes] = 20
    v = np.zeros((2, 192, 192), dtype=np.float32)
    result = engine.forecast(h, v, 2, members=3)
    assert np.array_equal(result, np.tile(h[-1], (3, 2, 1, 1)))

def test_worker_invariance_and_seed(engine):
    h, v = rain_cells()
    a = engine.forecast(h, v, 2, members=3, workers=1, seed=1)
    b = engine.forecast(h, v, 2, members=3, workers=3, seed=1)
    c = engine.forecast(h, v, 2, members=3, workers=1, seed=2)
    assert np.array_equal(a, b, equal_nan=True)
    assert not np.array_equal(a, c, equal_nan=True)

@pytest.mark.parametrize("option,value", [("members",0),("members",97),("leads",0),("leads",15),
    ("seed",-1),("seed",2**32),("workers",0),("members",1.5),("seed",True)])
def test_invalid_options(option, value):
    engine=Forecast()
    h,v=rain_cells()
    options=dict(members=1,leads=1)
    options[option]=value
    with pytest.raises(ValueError):
        engine.forecast(h,v,**options)

def test_invalid_arrays():
    engine=Forecast()
    h,v=rain_cells()
    with pytest.raises(ValueError,match="requires history"):
        engine.forecast(h[:2],v,1)
    with pytest.raises(ValueError,match="dtype"):
        engine.forecast(h,v.astype(int),1)
    with pytest.raises(ValueError,match="real numeric"):
        engine.forecast(h.astype(complex),v,1)
    h[0,0,0]=np.nan
    with pytest.raises(ValueError,match="finite"):
        engine.forecast(h,v,1)
    h[:]=20.;v[:]=0
    with pytest.raises(RuntimeError,match="degenerate"):
        engine.forecast(h,v,1,members=1)
