import numpy as np
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar
from ..trend._moving import mEMA, mSMA


class rTTI(rIndicator):
    """TTI

    一种提前指标
    True (Relative/Absolute) Time Index by Liu Shengli

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
        dti: rti - (1-rti)
            上涨下跌时间差/总幅度
            过大: 代表上涨动力枯竭
            过小: 下跌动力枯竭
            相等: 震荡
    """

    def __init__(self, n, ma_type=None, gap_factor=1, **kwargs):
        # SMA: 震荡思想, 反映的思想是： 涨久易跌，跌久易涨
        # EMA: 趋势思想, 反映近期趋势
        assert ma_type is None or ma_type.lower() in ['sma', 'ema']
        self.ma_type = ma_type or 'ema'
        self.gap_factor = gap_factor
        self.fn_ma = mEMA if self.ma_type.lower() == 'ema' else mSMA
        self.n = n
        self.fn_up_ma = self.fn_ma(n)
        self.fn_down_ma = self.fn_ma(n)
        super(rTTI, self).__init__(
            window=n,
            schema=Schema([
                ('rti', Scalar()),
                ('uat', Scalar()),
                ('dat', Scalar()),
            ]),
            **kwargs
        )

    def forward(self, opens, highs, lows, closes):
        '''
        return:
            rti: relative time-strength index
            uat: up absolute time
            dat: down absolute time
        '''
        gap = opens[-1] - closes[-2] if len(closes) > 1 else .0
        up = highs[-1] - lows[-1] - \
            np.max(opens[-1] - closes[-1], 0) + \
            self.gap_factor * np.max(gap, 0)
        down = highs[-1] - lows[-1] - \
            np.max(closes[-1] - opens[-1], 0) + \
            self.gap_factor * np.max(-gap, 0)
        total = up + down

        up_ratio = up/total if total != 0 else 0.5
        up_time = 1 * up_ratio
        down_time = 1 - up_time

        uat = self.fn_up_ma.moving(up_time)
        dat = self.fn_down_ma.moving(down_time)

        rti = uat/(uat + dat) * 100
        return rti, uat, dat
