import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.highlow import rHigh, rLow


class rDC(rIndicator):
    """
    Applied to trend trading to indicate if a trend is strengthening.
    - During a Bullish Trend, price moving into overbought territory can indicate a strengthening trend.
    - During a Bearish Trend, price moving into oversold territory can indicate a strengthening trend.

    Ref:
        https://www.tradingview.com/wiki/Donchian_Channels_(DC)
    """
    name = "DC"

    def __init__(self, n=20, **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        self._H = rHigh(n)
        self._L = rLow(n)
        
        super(rDC, self).__init__(
            window=n,
            schema=[
                ('ub', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('lb', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._H.reset()
        self._L.reset()

    def forward(self, highs, lows):
        if len(highs) < self.n:
            return np.nan, np.nan, np.nan
        
        ub = self._H.rolling(highs)
        lb = self._L.rolling(lows)
        mid = (ub + lb) * 0.5
        return ub, mid, lb

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
