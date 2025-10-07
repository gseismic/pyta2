import numpy as np
import scipy.stats
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar


class rCorr(rIndicator):

    def __init__(self, n, method='pearson', **kwargs):
        assert n >= 1, f'{self.name} window must be greater than 1, got {n}'
        assert method in ['pearson', 'spearman', 'kendall'], (
            f'{self.name} method must be one of [pearson, spearman, kendall], got {method}'
        )
        self.n = n 
        self.method = method
        super(rCorr, self).__init__(
            window=n,
            schema=Schema([
                ('corr', Scalar(dtype=np.float64))
            ]),
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values1, values2):
        if len(values1) < self.required_window or len(values2) < self.required_window:
            return np.nan
        
        if self.method == 'pearson':
            return np.corrcoef(values1[-self.n:], values2[-self.n:])[0, 1]
        elif self.method == 'spearman':
            return scipy.stats.spearmanr(values1[-self.n:], values2[-self.n:])[0]
        elif self.method == 'kendall':
            return scipy.stats.kendalltau(values1[-self.n:], values2[-self.n:])[0]

    @property
    def full_name(self):
        return f'Corr({self.n},{self.method})'
