import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from scipy.stats import linregress


class rLinearRegress(rIndicator):
    """Linear Regression indicator (slope, intercept, next_val)"""
    name = "LinearRegress"

    def __init__(self, n, **kwargs):
        assert n >= 2, f'{self.name} window n must be at least 2, got {n}'
        self.n = n
        self.X = np.arange(1, n + 1)
        super(rLinearRegress, self).__init__(
            window=n,
            schema=[
                ('slope', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('intercept', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('next_val', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan, np.nan

        # linregress(x, y)
        res = linregress(self.X, values[-self.n:])
        slope, intercept = res.slope, res.intercept
        
        # Predicted value for the NEXT point (n+1)
        next_val = intercept + slope * (self.n + 1)
        return slope, intercept, next_val

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
