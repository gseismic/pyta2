
import numpy as np 
from ..base import rIndicator 
from ..trend.ma.ema import rEMA 
from ..utils.space.box import Box, Scalar 
from ..utils.deque.numpy_deque import NumpyDeque 

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
        assert n1 > 0, f'{self.name} n1 must be greater than 0, got {n1}'
        assert n2 > 0, f'{self.name} n2 must be greater than 0, got {n2}'
        assert n3 > 0, f'{self.name} n3 must be greater than 0, got {n3}'
        assert n1 > n2, f'{self.name} n1 must be greater than n2, got {n1}, {n2}'
        
        self.n1 = n1
        self.n2 = n2
        self.n3 = n3
        self.__ema_slow = rEMA(n1)
        self.__ema_fast = rEMA(n2)
        self.__ema_dif = rEMA(n3)
        self.DIFs = NumpyDeque(n3)
        
        # 使用新的Schema格式 - 直接传递字典
        schema = {
            'dif': Scalar(low=-np.inf, high=np.inf),
            'dea': Scalar(low=-np.inf, high=np.inf),
            'bar': Scalar(low=-np.inf, high=np.inf)
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

