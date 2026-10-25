"""Fixed-configuration STEPS forecasts; the forecast algorithm runs in native code."""
from ._native import Forecast
__version__ = "0.1.0"
__all__ = ["Forecast", "__version__"]
