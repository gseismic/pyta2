import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.trend.ma.sma import rSMA


class rBBI(rIndicator):
    """BBI - Bull and Bear Index (多空均线)

    四条 SMA 的平均值，平滑价格趋势。

    Formula:
        bbi = (SMA(n1) + SMA(n2) + SMA(n3) + SMA(n4)) / 4
    """
    name = 'BBI'

    def __init__(self, n1=3, n2=6, n3=12, n4=24, **kwargs):
        assert n1 < n2 < n3 < n4, f'n1<n2<n3<n4 required, got {n1},{n2},{n3},{n4}'
        self.n1, self.n2, self.n3, self.n4 = n1, n2, n3, n4
        self._ma1 = rSMA(n1)
        self._ma2 = rSMA(n2)
        self._ma3 = rSMA(n3)
        self._ma4 = rSMA(n4)
        super(rBBI, self).__init__(
            window=n4,
            schema=[
                ('bbi', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ma1.reset()
        self._ma2.reset()
        self._ma3.reset()
        self._ma4.reset()

    def forward(self, closes):
        v1 = self._ma1.rolling(closes)
        v2 = self._ma2.rolling(closes)
        v3 = self._ma3.rolling(closes)
        v4 = self._ma4.rolling(closes)
        if len(closes) < self.n4:
            return np.nan
        return (v1 + v2 + v3 + v4) * 0.25

    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3},{self.n4})'
