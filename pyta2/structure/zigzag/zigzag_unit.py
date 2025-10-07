import numpy as np
from ...utils.vector import MovingVector
from .base import rZigZagBase

"""
Updates:
    - [@2024-11-13] from pyta，移除了不必要的highs,lows，因为已经不需要用它们来算atr
    - [@2024-03-10 22:48:24] 只是把rZigZag_XXX_ATR去掉atr计算，转而直接输入units
"""


class rZigZag_Unit(rZigZagBase):
    name = 'ZigZag_Unit'
    # NOTE: [@2024-03-10 22:48:24] 只是把rZigZag_XXX_ATR去掉atr计算，转而直接输入units

    def __init__(self, delta, **kwargs):
        self._closes = MovingVector()
        self.ref_unit = None
        super(rZigZag_Unit, self).__init__(delta, **kwargs)

    def reset_extras(self):
        self._closes.clear()
        self.ref_unit = None
    def forward_ready(self, units, closes) -> bool:
        return len(closes) >= self.required_window
    
    def pre_forward(self, units, closes):
        return self._cache_and_compute(units, closes)

    def safe_forward(self, units, closes):
        return self._cache_and_compute(units, closes)
    
    def _cache_and_compute(self, units, closes):
        confirmed_at = self._default_confirmed_at
        self._closes.append(closes[-1])

        # ATR 计算
        unit = units[-1]
        i = self._closes.notional_len - 1  # latest
        assert i == self.g_index

        if self.ref_unit is None and not np.isnan(units[-1]):
            self.ref_unit = units[-1]

        if self.ref_unit is None:
            self.i_low = i
            self.i_high = i
            return confirmed_at, self.searching_dir, self.i_high, self.i_low

        if self.searching_dir == 0:
            if self._closes[i] > self._closes[self.i_high]:
                self.i_high = i
            if self._closes[i] < self._closes[self.i_low]:
                self.i_low = i
        elif self.searching_dir == 1:
            if self._closes[i] > self._closes[self.i_high]:
                # 如果继续创新高，之后的求最低点比较点也要顺延
                self.i_high = i
                self.i_low = i
            if self._closes[i] < self._closes[self.i_low]:
                self.i_low = i
        elif self.searching_dir == -1:
            if self._closes[i] < self._closes[self.i_low]:
                self.i_low = i
                self.i_high = i
            if self._closes[i] > self._closes[self.i_high]:
                self.i_high = i

        delta_over_ihigh = (
            self._closes[i] - self._closes[self.i_high])/self.ref_unit
        delta_over_ilow = (self._closes[i] -
                           self._closes[self.i_low])/self.ref_unit

        if self.searching_dir in [1, 0] and delta_over_ihigh <= - self.down_delta:
            confirmed_at = self.i_high
            self._ICs.append(self.g_index)
            self._Is.append(self.i_high)
            self._Ts.append(1)
            self._Vs.append(self._closes[self.i_high])
            self.i_high = i
            self.i_low = i
            self.searching_dir = -1
            self.ref_unit = units[-1] # IC 位置
        elif self.searching_dir in [-1, 0] and delta_over_ilow >= self.up_delta:
            confirmed_at = self.i_low
            self._ICs.append(self.g_index)
            self._Is.append(self.i_low)
            self._Ts.append(-1)
            self._Vs.append(self._closes[self.i_low])
            self.i_high = i
            self.i_low = i
            self.searching_dir = 1
            self.ref_unit = units[-1] # 更新参考unit

        if confirmed_at is not None:
            self._closes.rekeep_n(self._closes.notional_len - confirmed_at)

        return confirmed_at, self.searching_dir, self.i_high, self.i_low
    
    @property
    def full_name(self):
        return f"{self.name}({self.delta})"
