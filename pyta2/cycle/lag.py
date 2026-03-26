import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rLag(rIndicator):
    """延迟线 (Lag line)"""
    name = "Lag"

    def __init__(self, n, **kwargs):
        # 0, window-1 数据为 nan
        assert n >= 0, f'{self.name} n must be at least 0, got {n}'
        self.n = n
        super(rLag, self).__init__(
            window=n,
            schema=[
                ('lag', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n + 1:
            return np.nan
        # 返回 n 个周期前的数据
        return values[-(self.n + 1)]

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
