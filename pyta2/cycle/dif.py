import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rDif(rIndicator):
    """差分线 (First-order difference)"""
    name = "Dif"

    def __init__(self, n=1, **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        super(rDif, self).__init__(
            window=n,
            schema=[
                ('dif', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n + 1:
            return np.nan
        # 返回当前值与 n 个周期前之差
        return values[-1] - values[-(self.n + 1)]

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
