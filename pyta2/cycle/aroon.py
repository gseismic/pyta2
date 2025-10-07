import numpy as np
from ...base import rIndicator
from ...base.schema import Schema
from ...utils.space.box import Scalar


class rAroon(rIndicator):
    """
    举个栗子，若取时间段为25天，如果今天为最高价，则"最高价后的天数"为0，AroonUp =（25-0）/ 25×100%= 100%；如果10天前为最高价，则"最高价后的天数"为10，AroonUp =（25-10）/ 25×100%= 60%
    ref:
        https://zhuanlan.zhihu.com/p/27559632
        https://www.investopedia.com/terms/a/aroonoscillator.asp
        使用: 双aroon: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon_oscillator
    """

    def __init__(self, n=25, **kwargs):
        assert (n >= 1)
        self.n = n
        self.iloc_max = None
        self.iloc_min = None
        super(rAroon, self).__init__(
            window=n,
            schema=Schema([
                ('high_elapsed', Scalar()),
                ('low_elapsed', Scalar()),
            ]),
            **kwargs
        )

    def forward(self, values):
        # 单摆的相对位置, ma 是单摆的中心点
        if len(values) < self.n:
            return np.nan, np.nan
        self.iloc_max = np.argmax(values[-self.n:])
        self.iloc_min = np.argmin(values[-self.n:])
        high_elapsed = 100*self.iloc_max/(self.n-1)
        low_elapsed = 100*self.iloc_min/(self.n-1)
        # diff = high_elapsed - low_elapsed
        # return high_elapsed, low_elapsed, diff
        return high_elapsed, low_elapsed
