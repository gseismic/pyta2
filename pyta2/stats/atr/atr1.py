import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.deque import NumpyDeque
from pyta2.utils.space.box import Box


class rBaseATR1(rIndicator):
    """ATR Average True Range (Strtided inputs)"""
    name = "BaseATR1"

    def __init__(self, name, ma_callback, n, stride=60, **kwargs):
        assert stride >= 1
        assert n >= 1
        self.n = n
        self.stride = stride
        self._ma = ma_callback
        self._values_TR = NumpyDeque(maxlen=n)
        self._first = True
        self._i = 0
        self._atr = np.nan
        
        super(rBaseATR1, self).__init__(
            window=n * stride,
            schema=[
                ('atr', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        if name: self.name = name

    def reset_extras(self):
        self._values_TR.clear()
        self._ma.reset()
        self._first = True
        self._i = 0
        self._atr = np.nan

    def forward(self, values):
        if len(values) < self.n * self.stride:
            return np.nan

        if self._first:
            # 初始化，填充历史 TR
            for i in range(1, self.n):
                _V = values[(-i-1)*self.stride : -i*self.stride]
                tr = np.max(_V) - np.min(_V)
                self._values_TR.append(tr)
            self._first = False

        if self._i == self.stride or self._i == 0:
            tr = np.max(values[-self.stride:]) - np.min(values[-self.stride:])
            self._values_TR.append(tr)
            self._atr = self._ma.rolling(self._values_TR.values)
            self._i = 1
        else:
            self._i += 1
        return self._atr


class rEATR1(rBaseATR1):
    """Exponential ATR (Strided)"""
    def __init__(self, n=28, stride=1, **kwargs):
        from pyta2.trend.ma.ema import rEMA
        super(rEATR1, self).__init__(name="EATR1",
                                     ma_callback=rEMA(n),
                                     n=n, stride=stride, **kwargs)


class rSATR1(rBaseATR1):
    """Simple ATR (Strided)"""
    def __init__(self, n=28, stride=1, **kwargs):
        from pyta2.trend.ma.sma import rSMA
        super(rSATR1, self).__init__(name="SATR1",
                                     ma_callback=rSMA(n),
                                     n=n, stride=stride, **kwargs)


class rATR1(rBaseATR1):
    """Standard ATR (Strided) using AlphaEMA"""
    def __init__(self, n=14, stride=1, **kwargs):
        from pyta2.trend.ma.ema import rAlphaEMA
        super(rATR1, self).__init__(name="ATR1",
                                    ma_callback=rAlphaEMA(alpha=1.0/n, window=n),
                                    n=n, stride=stride, **kwargs)
