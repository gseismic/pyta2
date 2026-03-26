import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rROC(rIndicator):
    """ROC - Rate of Change

    变动速率：当前价格相对 n 根前价格的涨跌幅。

    Formula:
        roc = (close[-1] - close[-n-1]) / close[-n-1]
    """
    name = 'ROC'

    def __init__(self, n, **kwargs):
        assert n >= 1, f'{self.name} n must be >= 1, got {n}'
        self.n = n
        super(rROC, self).__init__(
            window=n + 1,
            schema=[
                ('roc', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, closes):
        if len(closes) < self.n + 1:
            return np.nan
        return (closes[-1] - closes[-self.n - 1]) / closes[-self.n - 1]

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
