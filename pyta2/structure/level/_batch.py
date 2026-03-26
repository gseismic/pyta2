import numpy as np
from pyta2.base import forward_rolling_apply
from .ntv import rNTV

def NTV(opens, highs, lows, closes, volumes, method='color_volume_bin', delta=0.01, k=3, n=20, **kwargs):
    return forward_rolling_apply(
        len(closes), rNTV, param_args=[method, delta, k, n],
        input_args=[np.asarray(opens), np.asarray(highs), np.asarray(lows), np.asarray(closes), np.asarray(volumes)],
        **kwargs
    )

__all__ = ['NTV']
