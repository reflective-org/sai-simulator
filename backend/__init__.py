"""Backend package for the SAI (Stratospheric Aerosol Injection) simulator.

Re-exports the public ``get_outputs`` pipeline entry points and the ``MIN_SAI_START``
constant so callers can ``from backend import ...`` without reaching into submodules.
"""
from .get_outputs import *
from .constants import MIN_SAI_START as MIN_SAI_START
