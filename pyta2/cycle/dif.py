from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar
from ._moving import mLag


class rDif(rIndicator):
    """延迟线
    """

    def __init__(self, n, **kwargs):
        # 0, window-1 数据为nan
        assert n >= 0
        self.n = n
        self._lag = mLag(n)
        super(rDif, self).__init__(
            window=n,
            schema=Schema([
                ('dif', Scalar())
            ]),
            **kwargs
        )

    def forward(self, values):
        return values[-1] - self._lag.moving(values[-1])
