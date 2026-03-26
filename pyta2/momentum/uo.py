import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from ..utils.deque import NumpyDeque


class rUO(rIndicator):
    """UO - Ultimate Oscillator (终极震荡指标)
    
    由 Larry Williams 开发，结合三个不同时间段的买压强度。
    
    Formula:
        BP = Close - min(Low, PrevClose)
        TR = max(High, PrevClose) - min(Low, PrevClose)
        UO = 100 * (4*SumBP7/SumTR7 + 2*SumBP14/SumTR14 + SumBP28/SumTR28) / (4+2+1)
    """
    name = 'UO'

    def __init__(self, n1=7, n2=14, n3=28, **kwargs):
        assert 1 < n1 < n2 < n3, f"n1, n2, n3 must be in increasing order, got {n1}, {n2}, {n3}"
        self.n1, self.n2, self.n3 = n1, n2, n3
        self._bp_cache = NumpyDeque(maxlen=n3)
        self._tr_cache = NumpyDeque(maxlen=n3)
        super(rUO, self).__init__(
            window=n3,
            schema=[
                ('uo', Box(low=0, high=100, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._bp_cache.clear()
        self._tr_cache.clear()

    def forward(self, highs, lows, closes):
        if len(closes) < 2:
            return np.nan
        
        prev_close = closes[-2]
        curr_low = lows[-1]
        curr_high = highs[-1]
        curr_close = closes[-1]
        
        bp = curr_close - min(curr_low, prev_close)
        tr = max(curr_high, prev_close) - min(curr_low, prev_close)
        
        self._bp_cache.append(bp)
        self._tr_cache.append(tr)
        
        if len(self._bp_cache) < self.n3:
            return np.nan
        
        sums_bp = []
        sums_tr = []
        for n in [self.n1, self.n2, self.n3]:
            # 获取最近 n 个 
            bp_seg = self._bp_cache.values[-n:]
            tr_seg = self._tr_cache.values[-n:]
            sums_bp.append(np.sum(bp_seg))
            sums_tr.append(np.sum(tr_seg))
            
        # 计算权重均值
        term1 = 4.0 * (sums_bp[0] / sums_tr[0]) if sums_tr[0] != 0 else 0
        term2 = 2.0 * (sums_bp[1] / sums_tr[1]) if sums_tr[1] != 0 else 0
        term3 = 1.0 * (sums_bp[2] / sums_tr[2]) if sums_tr[2] != 0 else 0
        
        uo = 100.0 * (term1 + term2 + term3) / 7.0
        return uo

    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3})'
