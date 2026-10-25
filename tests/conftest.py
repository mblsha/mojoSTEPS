import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import pytest
from mojosteps import Forecast

@pytest.fixture(params=["cpu", "gpu"])
def engine(request):
    if request.param == "gpu" and os.environ.get("MOJOSTEPS_TEST_GPU") != "1":
        pytest.skip("real Apple GPU tests require MOJOSTEPS_TEST_GPU=1")
    return Forecast(request.param)
