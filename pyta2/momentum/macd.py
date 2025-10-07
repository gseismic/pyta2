
import numpy as np 
from pyta2.base.indicator import rIndicator 
from pyta2.trend.ma.ema import rEMA 
from pyta2.utils.space.box import Box 
from pyta2.utils.deque.numpy_deque import NumpyDeque 

class rMACD(rIndicator):
    '''MACD
    
    Moving Average Convergence Divergence
    
    reference:
        https://school.stockcharts.com/doku.php?id=technical_indicators:hull_moving_average
    '''
    name = "MACD"

    def __init__(self, n1=26, n2=12, n3=9, **kwargs):
        """
        Args:
            n1: 慢速EMA周期
            n2: 快速EMA周期
            n3: 差值EMA周期
        """
        if n1 < 1 or n2 < 1 or n3 < 1:
            raise ValueError(f'{self.name} n1, n2, n3 must be greater than 0, got {n1}, {n2}, {n3}')
        if n1 <= n2:
            raise ValueError(f'{self.name} n1 must be greater than n2, got {n1}, {n2}')
        
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.__ema_slow = rEMA(n1)
        self.__ema_fast = rEMA(n2)
        self.__ema_dif = rEMA(n3)
        self.DIFs = NumpyDeque(n3)
        
        # 使用新的Schema格式 - 直接传递字典
        schema = {
            'dif': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
            'dea': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64),
            'bar': Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64)
        }
        
        super(rMACD, self).__init__(
            window=self.n1 - 1 + self.n3,
            schema=schema,
            **kwargs
        )
           
    def reset_extras(self):
        self.DIFs.clear()
        self.__ema_slow.reset()
        self.__ema_fast.reset()
        self.__ema_dif.reset()
       
    def forward(self, values):
        ema_slow = self.__ema_slow.rolling(values)
        ema_fast = self.__ema_fast.rolling(values)

        dif = ema_fast - ema_slow # 快慢线差值
        self.DIFs.push(dif)
        dea = self.__ema_dif.rolling(self.DIFs) # 快慢线短周期9的EMA
        bar = 2*(dif - dea) # MACD
        return dif, dea, bar
    
    @property
    def full_name(self):
        return f'{self.name}({self.n1},{self.n2},{self.n3})'

