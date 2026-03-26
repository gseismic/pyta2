import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rSortedAt(rIndicator):
    """
    Rolling Sorted Array Indicator.
    Returns the sorted values over a window, or specific ranks.
    """
    name = "SortedAt"

    def __init__(self, n, ranks=None, **kwargs):
        assert n >= 1
        self.n = n
        self.ranks = ranks if ranks is None or isinstance(ranks, (tuple, list)) else [ranks]
        self.sorted_values = None
        
        output_dim = n if self.ranks is None else len(self.ranks)
        
        super(rSortedAt, self).__init__(
            window=n,
            schema=[
                ('sorted', Box(low=-np.inf, high=np.inf, shape=(output_dim,), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        self.sorted_values = None

    def forward(self, values):
        if len(values) < self.n:
            output_dim = self.n if self.ranks is None else len(self.ranks)
            return np.full(output_dim, np.nan)
            
        self.sorted_values = sorted(values[-self.n:])
        if self.ranks is None:
            return np.array(self.sorted_values)
        else:
            return np.array([self.sorted_values[r-1] for r in self.ranks])

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
