import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rParaSAR(rIndicator):
    """ParaSAR - Parabolic SAR

    抛物线止损指标，自适应跟踪趋势。

    输出:
        is_long: 当前是否为多头趋势
        sar: 止损/止盈价格

    Reference:
        https://www.tradingview.com/wiki/Parabolic_SAR_(SAR)
        https://www.prorealcode.com/topic/parabolic-SAR-code/
    """
    name = 'ParaSAR'

    def __init__(self, start=0.02, step=0.02, stop=0.2, **kwargs):
        self.start = start
        self.step  = step
        self.stop  = stop
        self._long = None
        self._hp   = None
        self._lp   = None
        self._sar  = None
        self._af   = start
        super(rParaSAR, self).__init__(
            window=2,
            schema=[
                ('is_long', Box(low=0,       high=1,      shape=(), dtype=np.bool_)),
                ('sar',     Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._long = None
        self._hp   = None
        self._lp   = None
        self._sar  = None
        self._af   = self.start

    def forward(self, highs, lows):
        if len(highs) < 3:
            self._long = False
            self._sar  = np.nan
            return self._long, self._sar

        if len(highs) == 3:
            self._af = self.start
            if highs[-1] > highs[-2]:
                self._long = True
                self._sar  = lows[-1]
                self._hp   = highs[-2]
            else:
                self._long = False
                self._sar  = highs[-1]
                self._lp   = lows[-2]

        # 更新 SAR
        if self._long:
            self._sar = self._sar + self._af * (self._hp - self._sar)
            self._sar = min(self._sar, lows[-2], lows[-3])
        else:
            self._sar = self._sar + self._af * (self._lp - self._sar)
            self._sar = max(self._sar, highs[-2], highs[-3])

        # 检查转向
        reverse = False
        if self._long:
            if lows[-1] < self._sar:
                reverse      = True
                self._long   = False
                self._sar    = self._hp
                self._lp     = lows[-1]
                self._af     = self.start
        else:
            if highs[-1] > self._sar:
                reverse      = True
                self._long   = True
                self._sar    = self._lp
                self._hp     = highs[-1]
                self._af     = self.start

        # 更新 AF
        if not reverse:
            if self._long:
                if highs[-1] > self._hp:
                    self._hp = highs[-1]
                    self._af = min(self._af + self.step, self.stop)
            else:
                if lows[-1] < self._lp:
                    self._lp = lows[-1]
                    self._af = min(self._af + self.step, self.stop)

        return self._long, self._sar

    @property
    def full_name(self):
        return f'{self.name}({self.start},{self.step},{self.stop})'
