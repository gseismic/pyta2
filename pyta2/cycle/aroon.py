import numpy as np
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box


class rAroon(rIndicator):
    """
    举个栗子，若取时间段为25天，如果今天为最高价，则"最高价后的天数"为0，AroonUp =（25-0）/ 25×100%= 100%；如果10天前为最高价，则"最高价后的天数"为10，AroonUp =（25-10）/ 25×100%= 60%
    ref:
        https://zhuanlan.zhihu.com/p/27559632
        https://www.investopedia.com/terms/a/aroonoscillator.asp
        使用: 双aroon: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:aroon_oscillator
    """
    name = "Aroon"

    def __init__(self, n=25, **kwargs):
        assert n >= 1, f'{self.name} window n must be at least 1, got {n}'
        self.n = n
        super(rAroon, self).__init__(
            window=n,
            schema=[
                ('high_elapsed', Box(low=0.0, high=100.0, shape=(), dtype=np.float64)),
                ('low_elapsed', Box(low=0.0, high=100.0, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        pass

    def forward(self, values):
        if len(values) < self.n:
            return np.nan, np.nan
            
        subset = values[-self.n:]
        # iloc_max 从0到n-1
        iloc_max = np.argmax(subset)
        iloc_min = np.argmin(subset)
        
        # Aroon公式：((N - 至今最高价天数) / N) * 100
        # 至今最高价天数 = (n-1) - iloc_max
        # 结果：(n - 1 - ((n-1) - iloc_max)) / (n-1) * 100 = iloc_max / (n-1) * 100
        n_m_1 = self.n - 1 if self.n > 1 else 1.0
        high_elapsed = 100 * iloc_max / n_m_1
        low_elapsed = 100 * iloc_min / n_m_1
        
        return high_elapsed, low_elapsed

    @property
    def full_name(self):
        return f'{self.name}({self.n})'
