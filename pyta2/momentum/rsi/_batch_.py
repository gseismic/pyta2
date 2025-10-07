import numpy as np
from ...base import forward_rolling_apply
from ._rolling_ import rRSI

def RSI(values, n=14, **kwargs):
    return forward_rolling_apply(len(values), rRSI, param_args=[n], input_args=[np.array(values)], **kwargs)

__all__ = ['RSI']
