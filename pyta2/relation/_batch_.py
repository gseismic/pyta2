import numpy as np
from ..base import forward_rolling_apply
from ._rolling_ import rTwinCross, rMACross, rCorr

def TwinCross(values, obj1, obj2, stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rTwinCross, param_args=[obj1, obj2, stride],
        input_args=[np.asarray(values)],
        **kwargs
    )

def MACross(values, l, xl, ma_type='EMA', stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rMACross, param_args=[l, xl, ma_type, stride],
        input_args=[np.asarray(values)],
        **kwargs
    )

def Corr(values1, values2, n=20, **kwargs):
    return forward_rolling_apply(
        len(values1), rCorr, param_args=[n],
        input_args=[np.asarray(values1), np.asarray(values2)],
        **kwargs
    )

__all__ = ['TwinCross', 'MACross', 'Corr']
