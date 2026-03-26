import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.stats.path import rPath


class rKAMA(rIndicator):
    """KAMA - Kaufman's Adaptive Moving Average

    自适应移动平均线，根据价格效率比（ER）动态调整平滑系数。

    Params:
        n1: ER 计算窗口（净变动 / 路径长度）
        n2: 快速 EMA 的周期（最小平滑系数）
        n3: 慢速 EMA 的周期（最大平滑系数）

    Reference:
        https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:kaufman_s_adaptive_moving_average
    """
    name = 'KAMA'

    def __init__(self, n1=10, n2=2, n3=30, stride=1, **kwargs):
        assert n1 > 2,  f'n1 must > 2, got {n1}'
        assert n3 > n2, f'n3 must > n2, got {n3} <= {n2}'
        assert n3 > n1, f'n3 must > n1, got {n3} <= {n1}'
        self.n1, self.n2, self.n3 = n1, n2, n3
        self.stride = stride
        self._path = rPath(n1, stride=stride)
        self._kama = None
        super(rKAMA, self).__init__(
            window=n3,
            schema=[
                ('kama', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._path.reset()
        self._kama = None

    def forward(self, closes):
        if len(closes) < self.n3:
            return np.nan

        path, gain, loss = self._path.forward(closes)
        if path == 0:
            return self._kama if self._kama is not None else np.nan

        er = abs(gain - loss) / path
        fast_sc  = 2.0 / (self.n2 + 1)
        slow_sc  = 2.0 / (self.n3 + 1)
        sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2

        if self._kama is None:
            self._kama = float(np.mean(closes[-self.n3:]))
        self._kama = self._kama + sc * (closes[-1] - self._kama)
        return self._kama

    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3})'
