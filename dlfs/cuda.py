import numpy as np


gpu_enable = True

try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
    
from dlfs import Variable