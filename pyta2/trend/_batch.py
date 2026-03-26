import numpy as np
from ..base import forward_rolling_apply
from ._rolling import (
    rDMI, rBBI, rBOP, rParaSAR,
    rSMA, rEMA, rWMA, rHMA, rDEMA, rTEMA, rKAMA, rZLEMA
)
from .ma._batch import SMA, EMA, WMA, HMA, DEMA, TEMA, KAMA, ZLEMA

def DMI(highs, lows, closes, n=14, n_atr=14, **kwargs):
    return forward_rolling_apply(
        len(highs), rDMI, param_args=[n, n_atr],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes)], **kwargs)

def BBI(closes, n1=3, n2=6, n3=12, n4=24, **kwargs):
    return forward_rolling_apply(len(closes), rBBI, param_args=[n1, n2, n3, n4], input_args=[np.asarray(closes)], **kwargs)

def BOP(opens, highs, lows, closes, n=20, use_ema=False, **kwargs):
    return forward_rolling_apply(
        len(highs), rBOP, param_args=[n, use_ema],
        input_args=[np.asarray(opens), np.asarray(highs), np.asarray(lows), np.asarray(closes)], **kwargs)

def ParaSAR(highs, lows, step=0.02, max_step=0.2, **kwargs):
    return forward_rolling_apply(
        len(highs), rParaSAR, param_args=[step, max_step],
        input_args=[np.asarray(highs), np.asarray(lows)], **kwargs)

__all__ = [
    'DMI', 'BBI', 'BOP', 'ParaSAR',
    'SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA', 'KAMA', 'ZLEMA'
]
