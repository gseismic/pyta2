import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box

class rEMA(rIndicator):
    name = "EMA"

    def __init__(self, n, **kwargs):
        if n < 1:
            raise ValueError(f'{self.name} n must be greater than 0, got {n}')
        self.n = n
        self.alpha = 2.0 / (n + 1)
        
        # 使用新的Schema格式 - 直接传递字典
        schema = {'ema': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)}
        
        super(rEMA, self).__init__(
            window=n,
            schema=schema,
            **kwargs
        )
    
    def reset_extras(self):
        self.ema = None

    def forward(self, values):
        # 允许 [nan, nan, ...,1, 1,1 ...]
        # 遇到nan, 将重启计算
        if self.ema is None or np.isnan(self.ema):
            self.ema = np.mean(values[-self.n:])
            return self.ema
        value = values[-1]
        if np.isnan(value):
            raise ValueError(f'{self.name} value must not be nan after {self.ema} is calculated')
        self.ema = (value - self.ema) * self.alpha + self.ema
        return self.ema

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
