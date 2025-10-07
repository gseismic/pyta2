from ...utils.vector import MovingVector
from .base import rZigZagBase
import warnings

"""
Note: 
    1. 
"""

class rZigZag_HL(rZigZagBase):
    name = "ZigZag_HL"

    def __init__(self, delta, use_pct=True, **kwargs):
        self.use_pct = use_pct
        self._highs = MovingVector()
        self._lows = MovingVector()
        super(rZigZag_HL, self).__init__(delta, **kwargs)

    def reset_extras(self):
        self._highs.clear()
        self._lows.clear()
        
    def forward_ready(self, highs, lows) -> bool:
        assert len(highs) == len(lows)
        return len(highs) >= self.required_window
    
    def pre_forward(self, highs, lows):
        return self._cache_and_compute(highs, lows)

    def safe_forward(self, highs, lows):
        return self._cache_and_compute(highs, lows)

    def _cache_and_compute(self, highs, lows):
        confirmed_at = self._default_confirmed_at
        self._highs.append(highs[-1])
        self._lows.append(lows[-1])

        i = self._highs.notional_len - 1  # latest

        if self.searching_dir == 0:
            if self._highs[i] > self._highs[self.i_high]:
                self.i_high = i
            if self._lows[i] < self._lows[self.i_low]:
                self.i_low = i
        elif self.searching_dir == 1:
            if self._highs[i] > self._highs[self.i_high]:
                self.i_high = i
                self.i_low = i
            if self._lows[i] < self._lows[self.i_low]:
                self.i_low = i
        elif self.searching_dir == -1:
            if self._lows[i] < self._lows[self.i_low]:
                self.i_low = i
                self.i_high = i
            if self._highs[i] > self._highs[self.i_high]:
                self.i_high = i

        # [@2024-03-19 14:01:28] 
        # XXX TODO 简单跳过存在问题
        warnings.warn("简单跳过存在问题, fix it in future")
        if self.i_low == self.i_high:
            return confirmed_at, self.searching_dir, self.i_high, self.i_low

        delta_over_ihigh = self._lows[i] - self._highs[self.i_high]
        delta_over_ilow = self._highs[i] - self._lows[self.i_low]
        if self.use_pct:
            delta_over_ihigh /= self._highs[self.i_high]
            delta_over_ilow /= self._lows[self.i_low]

        if self.searching_dir in [1, 0] and delta_over_ihigh <= - self.down_delta:
            confirmed_at = self.i_high
            self._ICs.append(self.g_index)
            self._Is.append(self.i_high)
            self._Ts.append(1)
            self._Vs.append(self._highs[self.i_high])
            self.i_high = i
            self.i_low = i
            self.searching_dir = -1
        elif self.searching_dir in [-1, 0] and delta_over_ilow >= self.up_delta:
            confirmed_at = self.i_low
            self._ICs.append(self.g_index)
            self._Is.append(self.i_low)
            self._Ts.append(-1)
            self._Vs.append(self._lows[self.i_low])
            self.i_high = i
            self.i_low = i
            self.searching_dir = 1

        if confirmed_at is not None:
            self._highs.rekeep_n(self._highs.notional_len - confirmed_at)
            self._lows.rekeep_n(self._lows.notional_len - confirmed_at)

        return confirmed_at, self.searching_dir, self.i_high, self.i_low
    
    @property
    def full_name(self):
        return f"{self.name}({self.delta}, use_pct={self.use_pct})"