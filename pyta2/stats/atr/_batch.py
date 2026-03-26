import numpy as np
from pyta2.base import forward_rolling_apply
from ._rolling import rATR, rDecATR, rATR1, rEATR1, rSATR1, rATRC1, rEATRC1, rSATRC1


def ATR(highs, lows, closes, n=20, ma_type='EMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rATR, param_args=[n, ma_type],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def SATR(highs, lows, closes, n=20, **kwargs):
    """Simple ATR (SMA-based). Alias for ATR with ma_type='SMA'."""
    return forward_rolling_apply(
        len(highs), rATR, param_args=[n, 'SMA'],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def DecATR(highs, lows, closes, n=20, ma_type='EMA', **kwargs):
    return forward_rolling_apply(
        len(highs), rDecATR, param_args=[n, ma_type],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def ATR1(highs, lows, closes, n=20, stride=1, **kwargs):
    return forward_rolling_apply(
        len(highs), rATR1, param_args=[n, stride],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def EATR1(highs, lows, closes, n=20, stride=1, **kwargs):
    return forward_rolling_apply(
        len(highs), rEATR1, param_args=[n, stride],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def SATR1(highs, lows, closes, n=20, stride=1, **kwargs):
    return forward_rolling_apply(
        len(highs), rSATR1, param_args=[n, stride],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def ATRC1(highs, lows, closes, n=20, **kwargs):
    return forward_rolling_apply(
        len(highs), rATRC1, param_args=[n],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def EATRC1(highs, lows, closes, n=20, **kwargs):
    return forward_rolling_apply(
        len(highs), rEATRC1, param_args=[n],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


def SATRC1(highs, lows, closes, n=20, **kwargs):
    return forward_rolling_apply(
        len(highs), rSATRC1, param_args=[n],
        input_args=[np.array(highs), np.array(lows), np.array(closes)],
        **kwargs
    )


__all__ = ['ATR', 'SATR', 'DecATR', 'ATR1', 'EATR1', 'SATR1', 'ATRC1', 'EATRC1', 'SATRC1']
