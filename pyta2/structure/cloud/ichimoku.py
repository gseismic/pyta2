import numpy as np
from ...base import rIndicator
from ...base.schema import Schema
from ...utils.space.box import Scalar
from ...stats import rHigh, rLow
from ...utils.deque.numpy_deque import NumpyDeque


class rIchimoku_HL(rIndicator):
    """IKH

    本质是一种平移指标: 因为均线有滞后，右移后提前了
    Ref:
        使用: https://academy.binance.com/zh/articles/ichimoku-clouds-explained
        多空指示标准: basic: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ichimoku_cloud
        https://www.binance.vision/economics/ichimoku-clouds-explained

        https://blog.xuite.net/metafun/life/80241486-%E4%B8%80%E7%9B%AE%E5%9D%87%E8%A1%A1%E5%9C%96+%28Ichimoku+Kinko+Hyo%29
        http://fx.taojin88.com/?p=12759
        signal:https://www.investopedia.com/terms/i/ichimoku-cloud.asp
        pattern: https://stockcharts.com/docs/doku.php?id=scans%3Aadvanced_scan_syntax%3Aichimoku_scans
    """

    def __init__(self, n1=9, n2=26, n3=52, **kwargs):
        assert n1 < n2 < n3
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self._high1 = rHigh(n1)
        self._low1 = rLow(n1)
        self._high2 = rHigh(n2)
        self._low2 = rLow(n2)
        self._high3 = rHigh(n3)
        self._low3 = rLow(n3)

        self._leading_As = NumpyDeque(maxlen=n2)
        self._leading_Bs = NumpyDeque(maxlen=n2)
        super(rIchimoku_HL, self).__init__(
            window=n3,
            schema=Schema([
                ('conversionLine', Scalar()),
                ('baseLine', Scalar()),
                ('lA', Scalar()),
                ('lB', Scalar()),
            ]),
            **kwargs
        )

    def forward(self, highs, lows):
        high1, low1 = self._high1.rolling(highs), self._low1.rolling(lows)
        high2, low2 = self._high2.rolling(highs), self._low2.rolling(lows)
        high3, low3 = self._high3.rolling(highs), self._low3.rolling(lows)

        conversion_line = (high1 + low1) * 0.5  # 9-period
        base_line = (high2 + low2) * 0.5  # 26-period

        # leading span A, 需要延迟 26个点取值
        leading_span_A = (conversion_line + base_line) * 0.5
        self._leading_As.push(leading_span_A)
        # leading span B, 需要延迟 26个点取值
        leading_span_B = (high3 + low3) * 0.5
        self._leading_Bs.push(leading_span_B)

        if len(highs) < self.n3:
            return np.nan, np.nan, np.nan, np.nan

        lA, lB = self._leading_As[-self.n2], self._leading_Bs[-self.n2]
        # 忽略close滞后LagSpan, 因为当前值永远是nan，改变的是值
        # 使用是需要，把close滞后26周期即可
        return conversion_line, base_line, lA, lB


class rIchimoku(rIndicator):

    def __init__(self, n1=9, n2=26, n3=52, **kwargs):
        self.__ichimoku_hl = rIchimoku_HL(n1, n2, n3)
        super(rIchimoku, self).__init__(
            window=n3,
            schema=Schema([
                ('conversionLine', Scalar()),
                ('baseLine', Scalar()),
                ('lA', Scalar()),
                ('lB', Scalar()),
            ]),
            **kwargs
        )

    def forward(self, values):
        return self.__ichimoku_hl.rolling(values, values)
