import numpy as np
from ...utils.vector import MovingVector
from .base import rZigZagBase


class rZigZag_HL_Unit(rZigZagBase):
    name = 'ZigZag_HL_Unit'
    def __init__(self, delta, **kwargs):
        self._highs = MovingVector()
        self._lows = MovingVector()
        self._closes = MovingVector()
        self.ref_unit = None
        super(rZigZag_HL_Unit, self).__init__(delta, **kwargs)

    def reset_extras(self):
        self.ref_unit = None
        self._highs.clear()
        self._lows.clear()
        self._closes.clear()
        
    def forward_ready(self, units, highs, lows, closes) -> bool:
        print(f'{len(highs)=}')
        print(f'{self.required_window=}')
        print(f'{len(highs) >= self.required_window=}')
        return len(highs) >= self.required_window

    def pre_forward(self, units, highs, lows, closes):
        return self._cache_and_compute(units, highs, lows, closes)

    def safe_forward(self, units, highs, lows, closes):
        return self._cache_and_compute(units, highs, lows, closes)
    
    def _cache_and_compute(self, units, highs, lows, closes):
        confirmed_at = self._default_confirmed_at
        self._highs.append(highs[-1])
        self._lows.append(lows[-1])
        self._closes.append(closes[-1])

        # if isinstance(units, (list, np.ndarray)):
        #     unit = units[-1]  # self.fn_hlc(highs, lows, closes)
        # else:
        #     unit = units

        # 应该使用参考点的unit，而不是最新的unit
        i = self._highs.notional_len - 1  # latest
        if self.ref_unit is None and not np.isnan(units[-1]):
            self.ref_unit = units[-1]

        if self.ref_unit is None:
            self.i_low = i
            self.i_high = i
            return confirmed_at, self.searching_dir, self.i_high, self.i_low

        if self.searching_dir == 0:
            if self._highs[i] > self._highs[self.i_high]:
                self.i_high = i
            if self._lows[i] < self._lows[self.i_low]:
                self.i_low = i
            # print(f'{self.i_high, self.i_low=}')
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

        # [@2024-03-15 17:56:12] 同一个bar不实施比较
        if self.i_low == self.i_high:
            return confirmed_at, self.searching_dir, self.i_high, self.i_low

        delta_over_ihigh = (self._lows[i] - self._highs[self.i_high]) / self.ref_unit
        delta_over_ilow = (self._highs[i] - self._lows[self.i_low]) / self.ref_unit

        if self.searching_dir in [1, 0
                                  ] and delta_over_ihigh <= -self.down_delta:
            # 第一个点不可靠
            # if self.searching_dir == 0:
            #     print(f'**{self.i_high, self.g_index=}')
            confirmed_at = self.i_high
            self._ICs.append(self.g_index)
            self._Is.append(self.i_high)
            self._Ts.append(1)
            self._Vs.append(self._highs[self.i_high])
            self.i_high = i
            self.i_low = i
            self.searching_dir = -1
            self.ref_unit = units[-1] # 更新参考unit，在确定的那一刻的atr
        elif self.searching_dir in [-1, 0
                                    ] and delta_over_ilow >= self.up_delta:
            # if self.searching_dir == 0:
            #     print(f'**{self.i_low, self.g_index=}')
            confirmed_at = self.i_low
            self._ICs.append(self.g_index)
            self._Is.append(self.i_low)
            self._Ts.append(-1)
            self._Vs.append(self._lows[self.i_low])
            self.i_high = i
            self.i_low = i
            self.searching_dir = 1
            self.ref_unit = units[-1] # 更新参考unit
            
        if confirmed_at is not None:
            self._highs.rekeep_n(self._highs.notional_len - confirmed_at)
            self._lows.rekeep_n(self._lows.notional_len - confirmed_at)
            self._closes.rekeep_n(self._closes.notional_len - confirmed_at)

        # searching_dir == 1:
        #   i_high: 当前点到上一个确认的ZigZag点之间：价格最高的点
        #   i_low: [i_high, current] 之间的价格最低点
        # searching_dir == -1:
        #   i_low: 当前点到上一个确认的ZigZag点之间：价格最【低】的点
        #   i_high: [i_low, current] 之间的价格最低点
        return confirmed_at, self.searching_dir, self.i_high, self.i_low

    @property
    def full_name(self):
        return f"{self.name}({self.delta})" 