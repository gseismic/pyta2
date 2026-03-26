import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.atr.atr import rATR
from pyta2.stats.highlow import rHigh, rLow


class rChanExit(rIndicator):
    """
    ChanExit - Chandelier Exit
    基于 ATR 和最高/最低价的通道
    """
    name = "ChanExit"

    def __init__(self, n, F=3, **kwargs):
        assert n >= 1
        self.n = n
        self.F = F
        self._high = rHigh(n)
        self._low = rLow(n)
        self._atr = rATR(n)
        
        super(rChanExit, self).__init__(
            window=n,
            schema=[
                ('ub', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lb', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._high.reset()
        self._low.reset()
        self._atr.reset()

    def forward(self, highs, lows, closes):
        atr = self._atr.rolling(highs, lows, closes)
        high_val = self._high.rolling(highs)
        low_val = self._low.rolling(lows)

        if len(closes) < self.window:
            return np.nan, np.nan, np.nan

        mid = (high_val + low_val) * 0.5
        # 按照原代码逻辑返回 ub, mid, lb
        # ub = low + F*atr
        # lb = high - F*atr
        return low_val + self.F * atr, mid, high_val - self.F * atr

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.F})'
