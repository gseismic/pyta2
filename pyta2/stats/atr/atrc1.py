import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.deque import NumpyDeque
from pyta2.utils.space.box import Box
from pyta2.stats.highlow import rRangeHL


class rBaseATRC1(rIndicator):
    """Continuous ATR Average True Range (Strided)"""
    name = "BaseATRC1"

    def __init__(self, name, ma_callback, n, stride=60, **kwargs):
        assert stride >= 1
        assert n >= 1
        self.n = n
        self.stride = stride
        self._ma = ma_callback
        self._values_TR = NumpyDeque(maxlen=n*stride)
        self._range_HL = rRangeHL(stride)
        
        super(rBaseATRC1, self).__init__(
            window=n * stride,
            schema=[
                ('atr', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        if name: self.name = name

    def reset_extras(self):
        self._values_TR.clear()
        self._range_HL.reset()
        self._ma.reset()

    def forward(self, values):
        if len(values) < self.stride:
            return np.nan
        
        # 连续滚动计算 TR
        _range = self._range_HL.rolling(values[-self.stride:])
        self._values_TR.append(_range)
        
        if len(values) < self.window:
            return np.nan
            
        return self._ma.rolling(self._values_TR.values)


class rEATRC1(rBaseATRC1):
    """Continuous Exponential ATR (Strided)"""
    def __init__(self, n=28, stride=60, **kwargs):
        from pyta2.trend.ma.ema import rEMA
        super(rEATRC1, self).__init__(name="EATRC1",
                                       ma_callback=rEMA(n*stride),
                                       n=n, stride=stride, **kwargs)


class rSATRC1(rBaseATRC1):
    """Continuous Simple ATR (Strided)"""
    def __init__(self, n=28, stride=60, **kwargs):
        from pyta2.trend.ma.sma import rSMA
        super(rSATRC1, self).__init__(name="SATRC1",
                                       ma_callback=rSMA(n*stride),
                                       n=n, stride=stride, **kwargs)


class rATRC1(rBaseATRC1):
    """Continuous standard ATR (Strided) using AlphaEMA"""
    def __init__(self, n=14, stride=60, **kwargs):
        from pyta2.trend.ma.ema import rAlphaEMA
        super(rATRC1, self).__init__(name="ATRC1",
                                      ma_callback=rAlphaEMA(alpha=1.0/(n*stride),
                                                           window=(n*stride)),
                                      n=n, stride=stride, **kwargs)
