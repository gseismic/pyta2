import numpy as np
from ...base import forward_rolling_apply
from ._rolling import rVWAP

def VWAP(highs, lows, closes, volumes, n, ma_type='SMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rVWAP, param_args=[n, ma_type], 
        input_args=[np.array(highs), np.array(lows), np.array(closes), np.array(volumes)], **kwargs)

__all__ = ['VWAP']
