import numpy as np
from pyta2.base import rIndicatorHL
from pyta2.utils.deque import NumPyDeque
# from pyta2.stats.highlow import rRangeHL, rHigh, rLow
from pyta2.stats.highlow import rHigh, rLow


class rStrideATR(rIndicatorHL):
    """

    注意: 连续模式，均值周期是: n*stride
    """

    def __init__(self, n, stride):
        assert (stride >= 1)
        assert (n >= 1)
        from pyta2.trend.ma.ema import rAlphaEMA
        super(rStrideATR, self).__init__(name='StrideATR',
                                         window=n+stride, # todo
                                         output_dim=1,
                                         output_keys=['atr'])
        self.n = n
        self.stride = stride
        self.range_values = NumPyDeque(maxlen=n*stride)
        self._fn_high = rHigh(stride)
        self._fn_low = rLow(stride)
        # self.__ma = rEMA(n)
        self.__ma = rAlphaEMA(1/n, n)

    def rolling(self, highs, lows):
        high = self._fn_high.rolling(highs)
        low = self._fn_low.rolling(lows)
        if len(highs) < self.window:
            return np.nan

        tr = high - low
        tr /= np.sqrt(self.stride)
        self.range_values.push(tr)
        return self.__ma.rolling(self.range_values.values)
