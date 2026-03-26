import numpy as np
from ._utils import pct_rank
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rRankAt(rIndicator):
    """Value of rank k over a window n"""
    name = 'RankAt'

    def __init__(self, n, k, **kwargs):
        assert n >= 2, f'{self.name} window n must be at least 2, got {n}'
        assert 1 <= k <= n, f'{self.name} rank k must be between 1 and {n}, got {k}'
        self.n = n
        self.k = k
        self.pct = (k - 1) / float(n - 1) if n > 1 else 0.0
        
        super(rRankAt, self).__init__(
            window=n,
            schema=[
                ('value', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
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
