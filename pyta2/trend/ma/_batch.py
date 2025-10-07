import numpy as np
from ...base import forward_rolling_apply
from ._rolling import rSMA, rEMA, rWMA, rHMA, rDEMA, rTEMA

def SMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rSMA, param_args=[n], input_args=[np.array(values)], **kwargs)

def EMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rEMA, param_args=[n], input_args=[np.array(values)], **kwargs)

def WMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rWMA, param_args=[n], input_args=[np.array(values)], **kwargs)

def HMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rHMA, param_args=[n], input_args=[np.array(values)], **kwargs)

def DEMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rDEMA, param_args=[n], input_args=[np.array(values)], **kwargs)

def TEMA(values, n, **kwargs):
    return forward_rolling_apply(len(values), rTEMA, param_args=[n], input_args=[np.array(values)], **kwargs)

__all__ = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA']
