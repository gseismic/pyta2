import numpy as np
from ._utils import pct_rank
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rVoR(rIndicator):
    """
    Value Of Rank (VoR)
    输出 n 个值中排名第 k 大小的数值。
    k=1: 最小值; k=n: 最大值; k=n/2: 中值。
    """
    name = 'VoR'

    def __init__(self, n, k, **kwargs):
        assert k >= 1 and k <= n and n >= 2, f'{self.name} invalid parameters n={n}, k={k}'
        self.n = n
        self.k = k
        self.pct = (k - 1) / float(n - 1) if n > 1 else 0.0
        
        super(rVoR, self).__init__(
            window=n,
            schema=[
                ('vor', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return pct_rank(values[-self.n:], self.pct)

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.k})'
