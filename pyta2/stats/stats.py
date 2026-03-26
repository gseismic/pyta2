import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rSum(rIndicator):
    """Rolling sum of a series"""
    name = 'Sum'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rSum, self).__init__(
            window=n,
            schema=[
                ('sum', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.sum(values[-self.n:])


class rMax(rIndicator):
    """Rolling maximum of a series"""
    name = 'Max'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rMax, self).__init__(
            window=n,
            schema=[
                ('max', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.max(values[-self.n:])


class rMin(rIndicator):
    """Rolling minimum of a series"""
    name = 'Min'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rMin, self).__init__(
            window=n,
            schema=[
                ('min', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.min(values[-self.n:])


class rMean(rIndicator):
    """Rolling arithmetic mean of a series"""
    name = 'Mean'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rMean, self).__init__(
            window=n,
            schema=[
                ('mean', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.mean(values[-self.n:])


class rStd(rIndicator):
    """Rolling standard deviation of a series"""
    name = 'Std'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rStd, self).__init__(
            window=n,
            schema=[
                ('std', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.std(values[-self.n:])


class rMedian(rIndicator):
    """Rolling median of a series"""
    name = 'Median'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rMedian, self).__init__(
            window=n,
            schema=[
                ('median', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        return np.median(values[-self.n:])
