import numpy as np
from ...base import forward_rolling_apply
from ._rolling_ import rMACD

def MACD(values, n1=26, n2=12, n3=9, **kwargs):
    return forward_rolling_apply(len(values), rMACD, param_args=[n1, n2, n3], input_args=[np.array(values)], **kwargs)

__all__ = ['MACD']

