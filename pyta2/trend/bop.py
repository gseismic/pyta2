import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque
from pyta2.trend.ma.sma import rSMA
from pyta2.trend.ma.ema import rEMA


class rBOP(rIndicator):
    """BOP - Balance of Power

    衡量多空力量均衡，取值范围 [-1, 1]，正值代表多方占优。

    Formula:
        bop_raw = (close - open) / (high - low)
        bop = MA(bop_raw, n)

    Reference:
        https://www.marketvolume.com/technicalanalysis/balanceofpower.asp
    """
    name = 'BOP'

    def __init__(self, n=20, use_ema=False, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self.use_ema = use_ema
        self._ma = rEMA(n) if use_ema else rSMA(n)
        self._values = NumpyDeque(maxlen=n)
        super(rBOP, self).__init__(
            window=n,
            schema=[
                ('bop', Box(low=-1, high=1, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ma.reset()
        self._values.clear()

    def forward(self, opens, highs, lows, closes):
        rng = highs[-1] - lows[-1]
        if rng == 0:
            bop_raw = 0.0
        else:
            bop_raw = (closes[-1] - opens[-1]) / rng
        self._values.append(bop_raw)
        if len(self._values) < self.n:
            return np.nan
        return self._ma.rolling(self._values.values)

    @property
    def full_name(self):
        return f'{self.name}({self.n},{"EMA" if self.use_ema else "SMA"})'
