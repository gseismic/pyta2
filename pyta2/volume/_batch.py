import numpy as np
from pyta2.base import forward_rolling_apply
from ._rolling import rVWAP, rOBV, rMFI, rCMF, rEMV, rVPT

def VWAP(highs, lows, closes, volumes, n, ma_type='SMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rVWAP, param_args=[n, ma_type], 
        input_args=[np.array(highs), np.array(lows), np.array(closes), np.array(volumes)], **kwargs)

def OBV(closes, vols, **kwargs):
    return forward_rolling_apply(
        len(closes), rOBV, param_args=[],
        input_args=[np.array(closes), np.array(vols)], **kwargs)

def MFI(highs, lows, closes, vols, n, **kwargs):
    return forward_rolling_apply(
        len(highs), rMFI, param_args=[n],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes), np.asarray(vols)], **kwargs)

def CMF(highs, lows, closes, vols, n, **kwargs):
    return forward_rolling_apply(
        len(highs), rCMF, param_args=[n],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(closes), np.asarray(vols)], **kwargs)

def EMV(highs, lows, vols, n=14, **kwargs):
    return forward_rolling_apply(
        len(highs), rEMV, param_args=[n],
        input_args=[np.asarray(highs), np.asarray(lows), np.asarray(vols)], **kwargs)

def VPT(closes, vols, **kwargs):
    return forward_rolling_apply(
        len(closes), rVPT, param_args=[],
        input_args=[np.asarray(closes), np.asarray(vols)], **kwargs)

__all__ = ['VWAP', 'OBV', 'MFI', 'CMF', 'EMV', 'VPT']
