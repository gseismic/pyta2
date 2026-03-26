import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rTSI(rIndicator):
    """TSI - True Strength Index by Liu Shengli

    一种提前指标，衡量上涨/下跌幅度在总幅度中的占比。

    事实:
        r := 单位时间累积上涨幅度/单位时间累积震荡幅度
        上涨趋势: r > 0.5
        下跌趋势: r < 0.5
        震荡趋势: r = 0.5

    指标意义:
        tsi: 上涨幅度/总幅度
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
        uas/das: 上涨/下跌的 EMA
    """
    name = 'TSI'

    def __init__(self, n, method='ema', gap_factor=1, **kwargs):
        assert method in ('sma', 'ema'), f'method must be sma or ema, got {method}'
        self.n = n
        self.method = method
        self.gap_factor = gap_factor
        # 增量 EMA
        alpha = 2.0 / (n + 1) if method == 'ema' else 1.0 / n
        self._alpha = alpha
        self._uas = None
        self._das = None
        self._prev_close = None
        super(rTSI, self).__init__(
            window=n,
            schema=[
                ('tsi', Box(low=0,       high=100,   shape=(), dtype=np.float64)),
                ('uas', Box(low=0,       high=np.inf, shape=(), dtype=np.float64)),
                ('das', Box(low=0,       high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._uas = None
        self._das = None
        self._prev_close = None

    def _ema_update(self, current, prev):
        if prev is None:
            return current
        return prev + self._alpha * (current - prev)

    def forward(self, opens, highs, lows, closes):
        gap = opens[-1] - closes[-2] if len(closes) > 1 else 0.0
        up   = (highs[-1] - lows[-1]
                - max(opens[-1] - closes[-1], 0)
                + self.gap_factor * max(gap, 0))
        down = (highs[-1] - lows[-1]
                - max(closes[-1] - opens[-1], 0)
                + self.gap_factor * max(-gap, 0))

        self._uas = self._ema_update(up, self._uas)
        self._das = self._ema_update(down, self._das)

        total = self._uas + self._das
        if total == 0:
            tsi = 50.0
        else:
            tsi = self._uas / total * 100
        return tsi, self._uas, self._das

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.method})'
