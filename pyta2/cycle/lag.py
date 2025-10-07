from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar
from ._moving import mLag


class rLag(rIndicator):
    """延迟线
    """

    def __init__(self, n, **kwargs):
        # 0, window-1 数据为nan
        assert n >= 0
        self.n = n
        self._lag = mLag(n)
        super(rLag, self).__init__(
            window=n,
            schema=Schema([
                ('lag', Scalar())
            ]),
            **kwargs
        )

    def forward(self, values):
        return self._lag.moving(values[-1])
