import numpy as np
from ..base import forward_rolling_apply
from ._rolling_ import *

def TwinCross(values, obj1, obj2, stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rTwinCross, param_args=[obj1, obj2, stride],
        input_args=[np.array(values)],
        **kwargs
    )

def MACross(values, l, xl, ma_type='EMA', stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rMACross, param_args=[l, xl, ma_type, stride],
        input_args=[np.array(values)],
        **kwargs
    )
    
__all__ = ['TwinCross', 'MACross']
