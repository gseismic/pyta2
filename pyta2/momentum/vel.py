import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rVel(rIndicator):
    """Vel - Velocity

    速度：N 根 K 线的平均价格变化率。

    Formula:
        vel = (close[-1] - close[-1-stride]) / stride
    """
    name = 'Vel'

    def __init__(self, stride, **kwargs):
        assert stride >= 1, f'{self.name} stride must be >= 1, got {stride}'
        self.stride = stride
        super(rVel, self).__init__(
            window=stride + 1,
            schema=[
                ('vel', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, closes):
        if len(closes) < self.stride + 1:
            return np.nan
        return (closes[-1] - closes[-1 - self.stride]) / self.stride

    @property
    def full_name(self):
        return f'{self.name}({self.stride})'
