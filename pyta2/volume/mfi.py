import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from ..utils.deque import NumpyDeque


class rMFI(rIndicator):
    """MFI - Money Flow Index

    衡量资金流入流出的动量指标。

    Reference:
        https://www.investopedia.com/terms/m/mfi.asp
    """
    name = 'MFI'

    def __init__(self, n=14, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        self._prev_tp = None
        self._plus_mf_values = NumpyDeque(maxlen=n)
        self._minus_mf_values = NumpyDeque(maxlen=n)
        super(rMFI, self).__init__(
            window=n + 1,
            schema=[
                ('mfi', Box(low=0, high=100, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._prev_tp = None
        self._plus_mf_values.clear()
        self._minus_mf_values.clear()

    def forward(self, highs, lows, closes, volumes):
        tp = (highs[-1] + lows[-1] + closes[-1]) / 3.0
        mf = tp * volumes[-1]
        
        plus_mf, minus_mf = 0.0, 0.0
        if self._prev_tp is not None:
            if tp > self._prev_tp:
                plus_mf = mf
            elif tp < self._prev_tp:
                minus_mf = mf
        
        self._prev_tp = tp
        self._plus_mf_values.append(plus_mf)
        self._minus_mf_values.append(minus_mf)

        # 这里的 window=n+1 是因为需要前一个 tp 来决定 plus/minus，且需要累积 n 个 plus/minus 值
        if len(self._plus_mf_values) < self.n:
            return np.nan

        s_plus  = np.sum(self._plus_mf_values.values)
        s_minus = np.sum(self._minus_mf_values.values)
        
        total = s_plus + s_minus
        if total == 0:
            return 50.0
        return 100 * s_plus / total

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
