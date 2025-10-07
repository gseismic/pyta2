import numpy as np
from ...base import forward_rolling_apply
from ._rolling import rATR, rDecATR

def ATR(highs, lows, closes, n=20, ma_type='EMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rATR, param_args=[n, ma_type], 
        input_args=[np.array(highs), np.array(lows), np.array(closes)], 
        **kwargs
    )

def DecATR(highs, lows, closes, n=20, ma_type='EMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rDecATR, param_args=[n, ma_type], 
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )

__all__ = ['ATR', 'DecATR']
