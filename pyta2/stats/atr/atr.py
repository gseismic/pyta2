import numpy as np
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque
from pyta2.base.indicator import rIndicator


class rATR(rIndicator):
    """Average True Range (ATR) indicator"""
    name = "ATR"

    def __init__(self, n=20, ma_type='EMA', **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        from pyta2.trend.ma.api import get_ma_class
        self.n = n
        self.ma_type = ma_type
        self.fn_ma = get_ma_class(ma_type)(n)
        self.values_TR = NumpyDeque(maxlen=n)
        super(rATR, self).__init__(
            window=n,
            schema=[
                ('atr', Box(low=0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        self.fn_ma.reset()
        self.values_TR.clear()

    def _cache_data(self, highs, lows, closes):
        # 计算 True Range (TR)
        high_low = highs[-1] - lows[-1]
        if len(highs) == 1:
            TR = high_low
        else:
            high_close = abs(highs[-1] - closes[-2])
            low_close = abs(lows[-1] - closes[-2])
            TR = max(high_low, high_close, low_close)
        self.values_TR.append(TR)

    def forward(self, highs, lows, closes):
        self._cache_data(highs, lows, closes)
        if len(highs) < self.n:
            return np.nan
        return self.fn_ma.rolling(self.values_TR.values)

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'