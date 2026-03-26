import numpy as np
from collections import defaultdict
from pyta2.base.indicator import rIndicator
from pyta2.utils.space.box import Box
from pyta2.trend.ma.sma import rSMA
from pyta2.trend.ma.ema import rEMA
from pyta2.stats.atr.atr import rATR


class rNTV(rIndicator):
    """NTV - Num Try Volume (支撑阻力叠加指标)
    
    按照价格进行支撑阻力叠加。一种基于成交量的提前指标。
    """
    name = 'NTV'

    def __init__(self, method='color_volume_bin', delta=0.01, k=3, n=20, **kwargs):
        self.method = method
        self.delta = delta
        self.k = k
        self.n = n
        
        self._atr = rATR(n)
        self._sma_volume = rSMA(n)
        self._up_volume_ema = rEMA(n)
        self._down_volume_ema = rEMA(n)
        self._up_ema = rEMA(n)
        self._down_ema = rEMA(n)
        
        self.resis_forces = defaultdict(lambda: dict(price=None, value=0.0))
        self.supp_forces = defaultdict(lambda: dict(price=None, value=0.0))
        
        super(rNTV, self).__init__(
            window=n,
            schema=[
                ('resis_forces', Box(low=0, high=0, dtype=object)), # dict
                ('supp_forces',  Box(low=0, high=0, dtype=object)), # dict
                ('top_resis_k',  Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
                ('bottom_supp_k', Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
                ('top_resis_k2', Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
                ('bottom_supp_k2', Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
                ('top_resis_2k', Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
                ('bottom_supp_2k', Box(low=0, high=np.inf, shape=(), dtype=np.float64)),
            ],
            **kwargs
        )

    def reset_extras(self):
        self._atr.reset()
        self._sma_volume.reset()
        self._up_volume_ema.reset()
        self._down_volume_ema.reset()
        self._up_ema.reset()
        self._down_ema.reset()
        self.resis_forces.clear()
        self.supp_forces.clear()

    def forward(self, opens, highs, lows, closes, volumes):
        ma_volume = self._sma_volume.rolling(volumes[-1:]) # SMA(v, n)
        atr = self._atr.rolling(highs, lows, closes)
        
        top = max(opens[-1], closes[-1])
        bottom = min(opens[-1], closes[-1])
        
        i_delta_high = int(np.ceil(highs[-1] / self.delta))
        i_delta_low = int(np.floor(lows[-1] / self.delta))
        
        # 计算上涨/下跌分量
        up = highs[-1] - lows[-1] - max(opens[-1] - closes[-1], 0)
        down = highs[-1] - lows[-1] - max(closes[-1] - opens[-1], 0)
        total = up + down
        
        if total == 0:
            up_volume = 0.0
            down_volume = 0.0
        else:
            up_volume = volumes[-1] * up / total
            down_volume = volumes[-1] * down / total
            
        # ma_volume 如果太小，会导致归一化失效
        norm_vol = ma_volume if ma_volume > 0 else 1.0
        
        # 权重设置
        weights = np.array([(self.k - i) / (self.k * (self.k + 1) / 2) for i in range(self.k)])
        
        if self.method == 'color_volume_bin':
            for i in range(self.k):
                idx_h = i_delta_high + i
                idx_l = i_delta_low - i
                self.resis_forces[idx_h]['value'] += weights[i] * up_volume / norm_vol
                self.resis_forces[idx_h]['price'] = idx_h * self.delta
                self.supp_forces[idx_l]['value'] += weights[i] * down_volume / norm_vol
                self.supp_forces[idx_l]['price'] = idx_l * self.delta
        
        # 击穿逻辑
        top_idx = int(np.ceil(top / self.delta))
        bottom_idx = int(np.floor(bottom / self.delta))
        
        for idx in list(self.resis_forces.keys()):
            if idx < top_idx:
                self.resis_forces[idx]['value'] = 0.0
            elif idx < i_delta_high:
                self.resis_forces[idx]['value'] *= 0.5
                
        for idx in list(self.supp_forces.keys()):
            if idx > bottom_idx:
                self.supp_forces[idx]['value'] = 0.0
            elif idx > i_delta_low:
                self.supp_forces[idx]['value'] *= 0.5
                
        # 计算 K 窗口内的核心支撑阻力
        top_resis_k = 0.0
        bottom_supp_k = 0.0
        for i in range(self.k):
            top_resis_k += self.resis_forces[top_idx + i]['value']
            bottom_supp_k += self.supp_forces[bottom_idx - i]['value']
            
        top_resis_k2 = 0.0
        bottom_supp_k2 = 0.0
        for i in range(self.k, 2 * self.k):
            top_resis_k2 += self.resis_forces[top_idx + i]['value']
            bottom_supp_k2 += self.supp_forces[bottom_idx - i]['value']
            
        top_resis_2k = top_resis_k + top_resis_k2
        bottom_supp_2k = bottom_supp_k + bottom_supp_k2
        
        return (
            dict(self.resis_forces), dict(self.supp_forces),
            top_resis_k, bottom_supp_k,
            top_resis_k2, bottom_supp_k2,
            top_resis_2k, bottom_supp_2k
        )

    @property
    def full_name(self):
        return f'{self.name}({self.method},{self.delta},{self.k},{self.n})'
