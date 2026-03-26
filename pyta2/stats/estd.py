import numpy as np
from ..base import rIndicatorC 
from ..trend.ma import rEMA


class rStdx(rIndicatorC):
    """
    去均值后，指数权重求标准差
    """
    def __init__(self, n_mean, n_ema):
        rIndicatorC.__init__(self, 
                             name='EStd',
                             window=n, 
                             output_dim=3)
        self.n_std = n_std
        self.n_ema = n_ema
        self.fn_ma = rSMA(n_mean)
        self.fn_ema_std = rEMA(n_mean)

    def rolling(self, values):
        assert 0
        if len(values) < self.n:
            return np.nan, np.nan, np.nan

        ma = self.fn_ema.rolling(values)
        mid = np.mean(values[-self.n:])
        std = np.std(values[-self.n:])
        ub = mid + self.F * std
        lb = mid - self.F * std
        return ub, mid, lb
