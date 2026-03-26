import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rTPI1(rIndicator):
    """TPI1

    与 TSI1 呼应
    一种提前指标
    上涨下跌概率
    True Probability Dense Index by Liu Shengli

    假设:
        平均上涨斜率 == 平均下跌斜率
        推出: 上涨时间正比于上涨幅度
    事实:
        统计上涨数量，上涨数量比值
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
    """
    name = "TPI1"

    def __init__(self, n, binsize, stride=1, **kwargs):
        assert binsize >= 1
        self.n = n
        self.stride = stride
        self.binsize = binsize
        
        # 内部状态：用于计算 EMA
        self._prob_ema_val = None
        self.alpha = 2.0 / (n + 1)
        
        super(rTPI1, self).__init__(
            window=(n + stride + binsize + 1),
            schema=[
                ('tpi', Box(low=0.0, high=1.0, shape=(), dtype=np.float64)),
                ('prob', Box(low=0.0, high=1.0, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._prob_ema_val = None

    def _update_ema(self, ema_val, current_val):
        if ema_val is None or np.isnan(ema_val):
            return current_val
        return (current_val - ema_val) * self.alpha + ema_val

    def forward(self, values):
        if len(values) < self.stride + self.binsize:
            return np.nan, np.nan

        n_positive = 0.0
        # 统计在 binsize 窗口内的上涨次数
        for i in range(self.binsize):
            # values[-1-i] 是从当前值开始往前的第 i 个值
            v_curr = values[-1 - i]
            v_prev = values[-1 - i - self.stride]
            delta = v_curr - v_prev
            if delta > 0:
                n_positive += 1.0
            elif delta == 0:
                n_positive += 0.5

        prob = n_positive / self.binsize
        self._prob_ema_val = self._update_ema(self._prob_ema_val, prob)

        if len(values) < self.window:
            return np.nan, np.nan

        return self._prob_ema_val, prob

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.binsize},{self.stride})'
