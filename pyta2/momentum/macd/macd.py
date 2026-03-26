import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.trend.ma.ema import rEMA
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque


class rMACD(rIndicator):
    """Moving Average Convergence Divergence (MACD)"""
    name = "MACD"

    def __init__(self, n1=26, n2=12, n3=9, **kwargs):
        assert n1 > n2, f'{self.name} n1 must be greater than n2, got {n1}, {n2}'
        assert n3 > 0, f'{self.name} n3 must be greater than 0, got {n3}'
        
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self._ema_slow = rEMA(n1)
        self._ema_fast = rEMA(n2)
        self._ema_dif = rEMA(n3)
        self._difs_cache = NumpyDeque(maxlen=n3)
        
        super(rMACD, self).__init__(
            window=n1 + n3 - 1,
            schema=[
                ('dif', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('dea', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('macd', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ema_slow.reset()
        self._ema_fast.reset()
        self._ema_dif.reset()
        self._difs_cache.clear()

    def forward(self, values):
        slow = self._ema_slow.rolling(values)
        fast = self._ema_fast.rolling(values)
        
        dif = fast - slow
        self._difs_cache.append(dif)
        
        # DEA is the EMA of DIF
        dea = self._ema_dif.rolling(self._difs_cache.values)
        macd = 2 * (dif - dea)
        
        return dif, dea, macd

    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3})'
    
