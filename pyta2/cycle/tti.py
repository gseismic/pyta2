import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


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
    name = "TTI"

    def __init__(self, n, ma_type='ema', gap_factor=1, **kwargs):
        # SMA: 震荡思想, 反映的思想是： 涨久易跌，跌久易涨
        # EMA: 趋势思想, 反映近期趋势
        assert ma_type.lower() in ['sma', 'ema']
        self.n = n
        self.ma_type = ma_type.lower()
        self.gap_factor = gap_factor
        # 内部增量 EMA 逻辑
        self._ua_ma_val = None
        self._da_ma_val = None
        self.alpha = 2.0 / (n + 1)
        
        super(rTTI, self).__init__(
            window=n,
            schema=[
                ('rti', Box(low=0.0, high=100.0, shape=(), dtype=np.float64)),
                ('uat', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('dat', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._ua_ma_val = None
        self._da_ma_val = None

    def _update_ma(self, ma_val, current_val):
        if ma_val is None or np.isnan(ma_val):
            return current_val
        return (current_val - ma_val) * self.alpha + ma_val

    def forward(self, opens, highs, lows, closes):
        """
        return:
            rti: relative time-strength index
            uat: up absolute time
            dat: down absolute time
        """
        if len(closes) < 2:
            return np.nan, np.nan, np.nan
            
        gap = opens[-1] - closes[-2]
        up = highs[-1] - lows[-1] - max(opens[-1] - closes[-1], 0) + self.gap_factor * max(gap, 0)
        down = highs[-1] - lows[-1] - max(closes[-1] - opens[-1], 0) + self.gap_factor * max(-gap, 0)
        total = up + down

        up_ratio = up / total if total != 0 else 0.5
        up_time = up_ratio
        down_time = 1.0 - up_time

        self._ua_ma_val = self._update_ma(self._ua_ma_val, up_time)
        self._da_ma_val = self._update_ma(self._da_ma_val, down_time)

        if len(closes) < self.window:
            return np.nan, np.nan, np.nan

        uat = self._ua_ma_val
        dat = self._da_ma_val

        rti = uat / (uat + dat) * 100 if (uat + dat) != 0 else 50.0
        return rti, uat, dat

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
