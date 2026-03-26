import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rCorr(rIndicator):
    """Correlation - 相关系数
    
    计算两组数值在滚动窗口内的 Pearson 相关系数。
    """
    name = 'Corr'

    def __init__(self, n, **kwargs):
        assert n >= 2
        self.n = n
        super(rCorr, self).__init__(
            window=n,
            schema=[
                ('corr', Box(low=-1.0, high=1.0, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values1, values2):
        if len(values1) < self.n or len(values2) < self.n:
            return np.nan
        
        # 使用 np.corrcoef
        c = np.corrcoef(values1[-self.n:], values2[-self.n:])
        return c[0, 1]

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
