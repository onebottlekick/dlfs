import numpy as np


gpu_enable = True

try:
    import cupy as cp
    cupy = cp
except ImportError:
    gpu_enable = False
    
from dlfs.core import Variable


def get_array_module(x):
    if isinstance(x, Variable):
        x = x.data
        
    if not gpu_enable:
        return np
    xp = cp.get_array_module(x)
    return xp


def as_numpy(x):
    if isinstance(x, Variable):
        x = x.data
        
    if np.isscalar(x):
        return np.array(x)

    elif isinstance(x, np.ndarray):
        return x
    

def as_cupy(x):
    if isinstance(x, Variable):
        x = x.data
        
    if not gpu_enable:
        raise Exception('Can not load cupy. Please install cupy')
    return cp.asarray(x)