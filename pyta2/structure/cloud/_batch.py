import numpy as np
from pyta2.base import forward_rolling_apply
from .ichimoku import rIchimoku, rIchimoku_HL

def Ichimoku(closes, n1=9, n2=26, n3=52, **kwargs):
    return forward_rolling_apply(
        len(closes), rIchimoku, param_args=[n1, n2, n3],
        input_args=[np.asarray(closes)],
        **kwargs
    )

def Ichimoku_HL(highs, lows, n1=9, n2=26, n3=52, **kwargs):
    return forward_rolling_apply(
        len(highs), rIchimoku_HL, param_args=[n1, n2, n3],
        input_args=[np.asarray(highs), np.asarray(lows)],
        **kwargs
    )

__all__ = ['Ichimoku', 'Ichimoku_HL']
