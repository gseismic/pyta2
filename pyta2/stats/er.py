import numpy as np
from ..base.indicator import rIndicator
from ..utils.space.box import Box


class rER(rIndicator):
    """ER - Efficiency Ratio (Kaufman)
    
    也称为 Kaufman 效率比。衡量趋势的平滑程度。
    ER = |Price_Change| / Path_Length
    """
    name = 'ER'

    def __init__(self, n=10, **kwargs):
        assert n >= 1
        self.n = n
        super(rER, self).__init__(
            window=n,
            schema=[
                ('er', Box(low=0, high=1, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan
        
        change = abs(values[-1] - values[-self.n])
        # 计算路径长度（相邻差值的绝对值之和）
        diffs = np.abs(np.diff(values[-self.n:]))
        volatility = np.sum(diffs)
        
        if volatility == 0:
            return 0.0
        return change / volatility

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
