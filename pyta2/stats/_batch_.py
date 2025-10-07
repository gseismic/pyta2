import numpy as np
from ..base import forward_rolling_apply
from .zscore import rZScore

def ZScore(values, n, **kwargs):
    return forward_rolling_apply(
        len(values), rZScore, param_args=[n],
        input_args=[np.array(values)],
        **kwargs
    )

__all__ = ['ZScore']