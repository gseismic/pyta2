import numpy as np
from .sorted_at import rSortedAt
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rRankRangeMean(rIndicator):
    """Mean of sorted values between two ranks over a window"""
    name = "RankRangeMean"

    def __init__(self, n, start_rank=None, end_rank=None, **kwargs):
        self.start_rank = 1 if start_rank is None else start_rank
        self.end_rank = n if end_rank is None else end_rank
        assert 1 <= self.start_rank < self.end_rank <= n, (
            f'{self.name} invalid ranks: {self.start_rank} to {self.end_rank} for window {n}'
        )
        self.n = n
        self.fn_sorted_at = rSortedAt(n)
        
        super(rRankRangeMean, self).__init__(
            window=n,
            schema=[
                ('mean', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        self.fn_sorted_at.reset()

    def forward(self, values):
        # We need the full sorted array to calculate the mean of a range
        # fn_sorted_at.rolling(values) returns the array
        sorted_values = self.fn_sorted_at.rolling(values)
        
        if len(values) < self.window:
            return np.nan
        
        # NumPy array slicing [start:end] where start is 0-indexed
        return np.mean(sorted_values[self.start_rank-1:self.end_rank])
    
    @property
    def sorted_values(self):
        return self.fn_sorted_at.sorted_values

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.start_rank},{self.end_rank})'
