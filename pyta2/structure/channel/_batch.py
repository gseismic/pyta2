import numpy as np
from ...base import forward_rolling_apply
from ._rolling import rBoll

def Boll(values, n=20, F=2, **kwargs):
    return forward_rolling_apply(
        len(values), rBoll, param_args=[n, F], 
        input_args=[np.array(values)], 
        **kwargs
    )

__all__ = ['Boll']
