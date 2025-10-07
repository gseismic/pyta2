import numpy as np
from ..trend._moving import mEMA
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar


class rTTI1(rIndicator):
    """TTI1

    与 TSI1 呼应
    一种提前指标
    True Time Index by Liu Shengli

    假设:
        平均上涨斜率 == 平均下跌斜率
        推出: 上涨时间正比于上涨幅度
    事实:
        r := 单位时间累积上涨时间/单位时间
        上涨趋势: r > 0.5
        下跌趋势: r < 0.5
        震荡趋势: r = 0.5
    指标意义:
        tti: 上涨时间/总时间
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
        dti: 上涨下跌时间差/总幅度
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
            相等: 震荡
    思想：
        价格上涨中本不需要多少成交量
        放量上涨，指标上涨
        放量，价格却不涨，为反转
    """
    def __init__(self, n, stride, **kwargs):
        assert(stride >= 1)
        self.n = n
        self.stride = stride
        self.__up_ema = mEMA(n)
        self.__down_ema = mEMA(n)
        super(rTTI1, self).__init__(
            window=(n+stride+1),
            schema=Schema([
                ('tti', Scalar()),
                ('dti', Scalar()),
                ('up_time', Scalar()),
                ('down_time', Scalar()),
                ('up_time_ma', Scalar()),
                ('down_time_ma', Scalar()),
            ]),
            **kwargs
        )

    def forward(self, values):
        if len(values) <= self.stride:
            return [np.nan]*6

        delta = values[-1] - values[-1-self.stride]

        up_time, down_time = 0, 0
        if delta > 0:
            up_time = 1
        elif delta < 0:
            down_time = 1
        else:
            up_time, down_time = 0.5, 0.5

        up_time_ma = self.__up_ema.moving(up_time)
        down_time_ma = self.__down_ema.moving(down_time)

        # moving mode need moving first
        if len(values) < self.window:
            return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

        tti = up_time_ma/(up_time_ma + down_time_ma)
        dti = (up_time_ma - down_time_ma)/(up_time_ma + down_time_ma)
        return tti, dti, up_time, down_time, up_time_ma, down_time_ma
