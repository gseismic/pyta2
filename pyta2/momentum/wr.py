import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.highlow import rHigh, rLow


class rWRx(rIndicator):
    """WRx - Williams %R (Extended, HLC input)

    输出 wr, high, low 三个值。

    Range: [0, 100]  (0 = overbought, 100 = oversold)

    Reference:
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    """
    name = 'WRx'

    def __init__(self, n, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._high = rHigh(n)
        self._low  = rLow(n)
        super(rWRx, self).__init__(
            window=n,
            schema=[
                ('wr',   Box(low=0,       high=100,   shape=(), dtype=np.float64)),
                ('high', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('low',  Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._high.reset()
        self._low.reset()

    def forward(self, highs, lows, closes):
        if len(highs) < self.n:
            return np.nan, np.nan, np.nan
        h = self._high.forward(highs)
        l = self._low.forward(lows)
        rng = h - l
        if rng == 0:
            return np.nan, h, l
        wr = (closes[-1] - l) / rng * 100
        return wr, h, l

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rWR(rIndicator):
    """WR - Williams %R

    Reference:
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r
    """
    name = 'WR'

    def __init__(self, n, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._wrx = rWRx(n)
        super(rWR, self).__init__(
            window=n,
            schema=[
                ('wr', Box(low=0, high=100, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._wrx.reset()

    def forward(self, highs, lows, closes):
        wr, _h, _l = self._wrx.forward(highs, lows, closes)
        return wr

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
