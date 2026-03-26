import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rPath(rIndicator):
    """
    Path indicator
    stride = 60: 相当于 1min-kline 的 TR
    """
    name = 'Path'

    def __init__(self, n, stride=1, **kwargs):
        assert n >= 1
        assert stride < n
        self.n = n
        self.stride = stride
        super(rPath, self).__init__(
            window=n,
            schema=[
                ('volatility', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('gain', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
                ('loss', Box(low=0.0, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan, np.nan

        i = 1
        gain, loss = 0.0, 0.0
        while True:
            if -i - self.stride < -self.n:
                break
            delta = values[-i] - values[-i - self.stride]
            if delta > 0:
                gain += delta
            else:
                loss += -delta
            i += self.stride

        volatility = gain + loss
        return volatility, gain, loss

    @property
    def full_name(self):
        return f'{self.name}({self.n},{self.stride})'
