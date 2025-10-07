from ...utils.vector import MovingVector
from .base import rZigZagBase


class rZigZag(rZigZagBase):
    """通过单一价格序列，寻找zigzag点
    """
    name = "ZigZag"

    def __init__(self, delta, use_pct=True, **kwargs):
        self.use_pct = use_pct
        self._values = MovingVector()
        super(rZigZag, self).__init__(delta, **kwargs)
    
    def reset_extras(self):
        self._values.clear()
        
    def forward_ready(self, values) -> bool:
        return len(values) >= self.required_window
    
    def pre_forward(self, values):
        return self._cache_and_compute(values)

    def safe_forward(self, values):
        return self._cache_and_compute(values)

    def _cache_and_compute(self, values):
        '''
        只输出全局索引，避免问题复杂化，要从数据开始就开始调用.rolling，而非等window-ready
        否则全局索引需要添加额外的偏移量
        '''
        # 注释：参见batch版本
        self._values.append(values[-1])
        _values = self._values

        # confirmed, confirmed_at = False, None
        confirmed_at = self._default_confirmed_at # 而不再是None

        i = _values.notional_len - 1  # latest
        assert i == self.g_index

        if _values.notional_len <= 2:
            return confirmed_at, self.searching_dir, self.i_high, self.i_low

        if self.searching_dir == 0:
            if _values[i] > _values[self.i_high]:
                self.i_high = i
            if _values[i] < _values[self.i_low]:
                self.i_low = i
        elif self.searching_dir == 1:
            # 如果在搜索最高点，ilow为当前最高点后的最低点
            if _values[i] > _values[self.i_high]:
                self.i_high = i
                self.i_low = i
            if _values[i] < _values[self.i_low]:
                self.i_low = i
        elif self.searching_dir == -1:
            if _values[i] < _values[self.i_low]:
                self.i_low = i
                self.i_high = i
            if _values[i] > _values[self.i_high]:
                self.i_high = i

        delta_over_ihigh = _values[i] - _values[self.i_high]
        delta_over_ilow = _values[i] - _values[self.i_low]
        if self.use_pct:
            delta_over_ihigh /= _values[self.i_high]
            delta_over_ilow /= _values[self.i_low]

        if self.searching_dir in [1, 0] and delta_over_ihigh <= - self.down_delta:
            # O1
            # confirmed = True
            confirmed_at = self.i_high

            self._ICs.append(self.g_index)
            self._Is.append(self.i_high)
            self._Ts.append(1)
            self._Vs.append(_values[self.i_high])
            self.i_high = i
            assert self.i_low == i
            self.i_low = i
            self.searching_dir = -1
        elif self.searching_dir in [-1, 0] and delta_over_ilow >= self.up_delta:
            # confirmed = True
            confirmed_at = self.i_low

            self._ICs.append(self.g_index)
            self._Is.append(self.i_low)
            self._Ts.append(-1)
            self._Vs.append(_values[self.i_low])
            self.i_low = i
            assert self.i_high == i
            self.i_high = i
            self.searching_dir = 1

        if confirmed_at is not None:
            self._values.rekeep_n(self._values.notional_len - confirmed_at)
            
        return confirmed_at, self.searching_dir, self.i_high, self.i_low
    
    @property
    def full_name(self):
        return f"{self.name}({self.delta}, use_pct={self.use_pct})"
