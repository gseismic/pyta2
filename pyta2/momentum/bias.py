import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.trend.ma.sma import rSMA


class rBias(rIndicator):
    """Bias - 乖离率

    单摆的相对位置，ma 是单摆的中心点。

    Formula:
        bias = (close - ma) / ma
    """
    name = 'Bias'

    def __init__(self, n, ma_type='SMA', **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        from pyta2.trend.ma.api import get_ma_class
        self.n = n
        self.ma_type = ma_type
        self._ma = get_ma_class(ma_type)(n)
        super(rBias, self).__init__(
            window=n,
            schema=[
                ('bias', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('ma',   Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ma.reset()

    def forward(self, closes):
        ma = self._ma.rolling(closes)
        if len(closes) < self.n:
            return np.nan, np.nan
        bias = (closes[-1] - ma) / ma
        return bias, ma

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.ma_type})'
