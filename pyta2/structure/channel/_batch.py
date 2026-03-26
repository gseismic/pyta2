import numpy as np
from pyta2.base import forward_rolling_apply
from ._rolling import rBoll, rDC, rFibDC, rKeltner, rChanExit


def Boll(values, n=20, F=2, **kwargs):
    return forward_rolling_apply(
        len(values), rBoll, param_args=[n, F],
        input_args=[np.array(values)],
        **kwargs
    )


def DC(highs, lows, n=20, **kwargs):
    return forward_rolling_apply(
        len(highs), rDC, param_args=[n],
        input_args=[np.array(highs), np.array(lows)],
        **kwargs
    )


def FibDC(highs, lows, n=20, **kwargs):
    return forward_rolling_apply(
        len(highs), rFibDC, param_args=[n],
        input_args=[np.array(highs), np.array(lows)],
        **kwargs
    )


def Keltner(highs, lows, closes, n=20, F=2, ma_type='EMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rKeltner, param_args=[n, F, ma_type],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def ChanExit(highs, lows, closes, n=20, F=3, **kwargs):
    return forward_rolling_apply(
        len(highs), rChanExit, param_args=[n, F],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


__all__ = ['Boll', 'DC', 'FibDC', 'Keltner', 'ChanExit']
