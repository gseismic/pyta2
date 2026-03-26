import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.highlow import rHigh, rLow


class rKDJ(rIndicator):
    """KDJ

    随机指标，测量价格在近期高低范围中的相对位置。
    K、D 值使用指数平滑；J 值为超买超卖信号。

    Reference:
        https://github.com/kaelzhang/kdj
        https://ichihedge.wordpress.com/2017/03/22/indicator-calculating-kdj/
    """
    name = 'KDJ'

    def __init__(self, n1=9, n2=3, n3=3, ktimes=3, dtimes=2, **kwargs):
        assert n1 > n2, f'n1 must > n2, got {n1} <= {n2}'
        assert n1 > n3, f'n1 must > n3, got {n1} <= {n3}'
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.ktimes = ktimes
        self.dtimes = dtimes
        self._high = rHigh(n1)
        self._low  = rLow(n1)
        self._prev_k = 50.0
        self._prev_d = 50.0
        super(rKDJ, self).__init__(
            window=n1,
            schema=[
                ('k', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('d', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('j', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._high.reset()
        self._low.reset()
        self._prev_k = 50.0
        self._prev_d = 50.0

    def forward(self, highs, lows, closes):
        if len(highs) < self.n1:
            return np.nan, np.nan, np.nan

        h = self._high.forward(highs)
        l = self._low.forward(lows)
        rng = h - l
        if rng == 0:
            rsv = 50.0
        else:
            rsv = (closes[-1] - l) / rng * 100

        k = self._prev_k * (self.n2 - 1) / self.n2 + rsv / self.n2
        d = self._prev_d * (self.n3 - 1) / self.n3 + k / self.n3
        j = k * self.ktimes - d * self.dtimes

        self._prev_k = k
        self._prev_d = d
        return k, d, j

    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3})'
