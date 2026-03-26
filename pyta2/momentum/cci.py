import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.utils.deque import NumpyDeque
from pyta2.trend.ma.sma import rSMA


class rCCIx(rIndicator):
    """CCIx - Commodity Channel Index (Extended)

    输出除 cci 值外，还包括 ma 和 dev。

    Reference:
        https://www.investopedia.com/terms/c/commoditychannelindex.asp

    Formula:
        tp = (high + low + close) / 3
        cci = (tp - ma(tp)) / (0.015 * mean_abs_dev(tp))
    """
    name = 'CCIx'

    def __init__(self, n=20, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._prices = NumpyDeque(maxlen=n)
        self._fn_ma = rSMA(n)
        super(rCCIx, self).__init__(
            window=n,
            schema=[
                ('cci', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('ma',  Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('dev', Box(low=0,       high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._prices.clear()
        self._fn_ma.reset()

    def forward(self, highs, lows, closes):
        tp = (highs[-1] + lows[-1] + closes[-1]) / 3.0
        self._prices.append(tp)
        ma = self._fn_ma.rolling(self._prices.values)

        if len(highs) < self.n:
            return np.nan, np.nan, np.nan

        dev = np.mean(np.abs(self._prices.values - ma))
        if dev == 0:
            return np.nan, ma, dev
        cci = (tp - ma) / (0.015 * dev)
        return cci, ma, dev

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rCCI(rIndicator):
    """CCI - Commodity Channel Index

    Reference:
        https://www.investopedia.com/terms/c/commoditychannelindex.asp
    """
    name = 'CCI'

    def __init__(self, n=20, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._ccix = rCCIx(n)
        super(rCCI, self).__init__(
            window=n,
            schema=[
                ('cci', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ccix.reset()

    def forward(self, highs, lows, closes):
        cci, _ma, _dev = self._ccix.forward(highs, lows, closes)
        return cci

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
