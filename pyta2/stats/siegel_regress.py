import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box
from scipy.stats import siegelslopes


class rSiegelRegress(rIndicator):
    """
    斜率滞后非常大，只适合结构性拟合(用最大最小值)

    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.siegelslopes.html#scipy.stats.siegelslopes
    """
    name = "SiegelRegress"

    def __init__(self, n, **kwargs):
        assert n >= 3, f'{self.name} window n must be at least 3, got {n}'
        self.n = n
        self.X = np.arange(1, n + 1)
        super(rSiegelRegress, self).__init__(
            window=n,
            schema=[
                ('intercept', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('slope', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('next_val', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan, np.nan

        # y, x
        slope, intercept = siegelslopes(values[-self.n:], self.X)
        
        # Predicted value for the NEXT point (n+1)
        next_val = intercept + slope * (self.n + 1)
        
        return intercept, slope, next_val

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
