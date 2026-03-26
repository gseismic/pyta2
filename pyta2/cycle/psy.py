import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from ..utils.deque import NumpyDeque


class rPSY(rIndicator):
    """
    PSY - Psychological Line
    一段时间内，上涨周期数占比。反映市场情绪。
    """
    name = "PSY"

    def __init__(self, n, **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        self._rising_history = NumpyDeque(maxlen=n)
        super(rPSY, self).__init__(
            window=n,
            schema=[
                ('psy', Box(low=0.0, high=1.0, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        self._rising_history.clear()

    def forward(self, values):
        if len(values) < 2:
            return np.nan

        # 记录当前收盘价是否上涨
        is_rise = 1 if values[-1] > values[-2] else 0
        self._rising_history.append(is_rise)
        
        if len(values) < self.n:
            return np.nan

        # PSY = (上涨天数 / N)
        return np.sum(self._rising_history.values) / self.n

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
