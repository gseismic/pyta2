import numpy as np
from ..base import forward_rolling_apply
from ._rolling import (
    rROC, rVel, rBias, rCCI, rWR, rKDJ, rTSI, rUO, rMACD, rRSI
)

def ROC(closes, n, **kwargs):
    return forward_rolling_apply(len(closes), rROC, param_args=[n], input_args=[np.asarray(closes)], **kwargs)

def Vel(closes, n, **kwargs):
    return forward_rolling_apply(len(closes), rVel, param_args=[n], input_args=[np.asarray(closes)], **kwargs)

def Bias(closes, n, ma_type='SMA', **kwargs):
    return forward_rolling_apply(len(closes), rBias, param_args=[n, ma_type], input_args=[np.asarray(closes)], **kwargs)

def CCI(highs, lows, closes, n, **kwargs):
    return forward_rolling_apply(
        len(highs), rCCI, param_args=[n], 
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes)], **kwargs)

def WR(highs, lows, closes, n, **kwargs):
    return forward_rolling_apply(
        len(highs), rWR, param_args=[n], 
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes)], **kwargs)

def KDJ(highs, lows, closes, n1=9, n2=3, n3=3, **kwargs):
    return forward_rolling_apply(
        len(highs), rKDJ, param_args=[n1, n2, n3],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes)], **kwargs)

def TSI(opens, highs, lows, closes, n=25, method='ema', gap_factor=1, **kwargs):
    return forward_rolling_apply(
        len(closes), rTSI, param_args=[n, method, gap_factor],
        input_args=[np.asarray(opens), np.asarray(highs), np.asarray(lows), np.asarray(closes)],
        **kwargs
    )

def UO(highs, lows, closes, n1=7, n2=14, n3=28, **kwargs):
    return forward_rolling_apply(
        len(closes), rUO, param_args=[n1, n2, n3],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes)],
        **kwargs
    )

def MACD(closes, fast=12, slow=26, signal=9, **kwargs):
    return forward_rolling_apply(len(closes), rMACD, param_args=[slow, fast, signal], input_args=[np.asarray(closes)], **kwargs)

def RSI(closes, n=14, **kwargs):
    return forward_rolling_apply(len(closes), rRSI, param_args=[n], input_args=[np.asarray(closes)], **kwargs)


__all__ = ['ROC', 'Vel', 'Bias', 'CCI', 'WR', 'KDJ', 'TSI', 'UO', 'MACD', 'RSI']
