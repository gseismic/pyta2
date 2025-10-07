import numpy as np
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar
from ..utils.deque.numpy_deque import NumpyDeque


class rPSY(rIndicator):
    '''
    一段时间内，涨跌天数ratio
    '''

    def __init__(self, n, **kwargs):
        assert n >= 1
        self.n = n
        # self._num_rise = 0
        self._rising_count = NumpyDeque(maxlen=n)
        super(rPSY, self).__init__(
            window=n,
            schema=Schema([
                ('psy', Scalar())
            ]),
            **kwargs
        )

    def forward(self, values):
        if len(values) <= 2:
            return np.nan

        count = 1 if values[-2] < values[-1] else 0
        self._rising_count.append(count)
        if len(values) < self.n:
            return np.nan

        return np.sum(self._rising_count.values)/self.n
        #if values[-self.n-1] < values[-self.n]:
        #    self._num_rise -= 1
        #if values[-2] < values[-1]:
        #    self._num_rise += 1
        #return self._num_rise
