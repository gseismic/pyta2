import numpy as np
import warnings
from .base import rBaseMA

class rAlphaEMA(rBaseMA):
    """EMA with custom alpha"""
    name = "AlphaEMA"

    def __init__(self, alpha, window, **kwargs):
        self.alpha = alpha
        super(rAlphaEMA, self).__init__(n=window, window=window, **kwargs)

    def reset_extras(self):
        self.ema = None

    def forward(self, values: np.ndarray):
        """
        一旦输入出现 NaN，EMA 状态将重置，直到重新凑齐 n 个有效点。
        """
        if len(values) < self.n:
            return np.nan
        
        value = values[-1]
        
        # 1. 如果当前值为 NaN，重置内部状态并返回 NaN
        # 标记为 None 后，下一轮会进入下面的 SMA 初始化逻辑
        if np.isnan(value):
            if self.ema is not None:
                warnings.warn(f'{self.name} at g_index={self.g_index} received NaN, resetting state and waiting for {self.n} valid points.')
            self.ema = None 
            return np.nan

        # 2. 如果尚未初始化（或刚因 NaN 被重置），尝试使用最近 n 个点的均值 (SMA) 初始化
        if self.ema is None:
            # np.mean(values[-self.n:]) 在包含任何 NaN 时都会返回 NaN
            # 这就实现了“重新等待 n 个有效点”的需求
            initial_sma = np.mean(values[-self.n:])
            if not np.isnan(initial_sma):
                self.ema = initial_sma
            return self.ema if self.ema is not None else np.nan

        # 3. 正常增量递归计算实现
        self.ema = (value - self.ema) * self.alpha + self.ema
        return self.ema


class rEMA(rAlphaEMA):
    """Exponential Moving Average (EMA)"""
    name = "EMA"

    def __init__(self, n, **kwargs):
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        alpha = 2.0 / (n + 1)
        # 调用父类 rAlphaEMA 的初始化逻辑，它会正确设置 n 和 window
        super(rEMA, self).__init__(alpha=alpha, window=n, **kwargs)
