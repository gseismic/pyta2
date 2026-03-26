import numpy as np
from ..base import forward_rolling_apply
from .zscore import rZScore
from .er import rER
from .misc import (
    rWMean, rWStd, rWVar, rWSkew, rWKurt, rWQuantile,
    rMax, rMin, rSum,
)
from ..relation.cross import rTwinCross, rMACross


def ZScore(values, n, **kwargs):
    return forward_rolling_apply(
        len(values), rZScore, param_args=[n],
        input_args=[np.array(values)],
        **kwargs
    )


def ER(values, n=10, **kwargs):
    return forward_rolling_apply(
        len(values), rER, param_args=[n],
        input_args=[np.array(values)],
        **kwargs
    )


def TwinCross(values, obj1, obj2, stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rTwinCross, param_args=[obj1, obj2, stride],
        input_args=[np.array(values)],
        **kwargs
    )


def MACross(values, l, xl, ma_type='EMA', stride=1, **kwargs):
    return forward_rolling_apply(
        len(values), rMACross, param_args=[l, xl, ma_type, stride],
        input_args=[np.array(values)],
        **kwargs
    )


def WMean(values, n, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWMean, param_args=[n, weights],
        input_args=[np.array(values)],
        **kwargs
    )


def WStd(values, n, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWStd, param_args=[n, weights],
        input_args=[np.array(values)],
        **kwargs
    )


def WVar(values, n, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWVar, param_args=[n, weights],
        input_args=[np.array(values)],
        **kwargs
    )


def WSkew(values, n, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWSkew, param_args=[n, weights],
        input_args=[np.array(values)],
        **kwargs
    )


def WKurt(values, n, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWKurt, param_args=[n, weights],
        input_args=[np.array(values)],
        **kwargs
    )


def WQuantile(values, n, q=0.5, weights=None, **kwargs):
    return forward_rolling_apply(
        len(values), rWQuantile, param_args=[n, q, weights],
        input_args=[np.array(values)],
        **kwargs
    )


__all__ = [
    'ZScore', 'ER',
    'TwinCross', 'MACross',
    'WMean', 'WStd', 'WVar', 'WSkew', 'WKurt', 'WQuantile',
]