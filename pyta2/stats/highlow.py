import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rHigh(rIndicator):
    """Rolling maximum value of a series"""
    name = 'High'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rHigh, self).__init__(
            window=n,
            schema=[
                ('high', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
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

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rLow(rIndicator):
    """Rolling minimum value of a series"""
    name = 'Low'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rLow, self).__init__(
            window=n,
            schema=[
                ('low', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
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

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rHighLow(rIndicator):
    """Rolling high and low values of a series"""
    name = 'HighLow'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rHighLow, self).__init__(
            window=n,
            schema=[
                ('low', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)),
                ('high', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan
        subset = values[-self.n:]
        return np.min(subset), np.max(subset)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rRangeHL(rIndicator):
    """Rolling range (High - Low) of a series"""
    name = 'RangeHL'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rRangeHL, self).__init__(
            window=n,
            schema=[
                ('range', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        subset = values[-self.n:]
        return np.max(subset) - np.min(subset)

    @property
    def full_name(self):
        return f'{self.name}({self.n})'


class rMiddle(rIndicator):
    """Rolling middle point (High + Low) / 2 of a series"""
    name = 'Middle'

    def __init__(self, n, **kwargs):
        assert n >= 1
        super(rMiddle, self).__init__(
            window=n,
            schema=[
                ('mid', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )
        self.n = n

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        subset = values[-self.n:]
        return 0.5 * (np.max(subset) + np.min(subset))

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
