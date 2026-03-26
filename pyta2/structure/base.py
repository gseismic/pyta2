import numpy as np
from ..utils.deque import NumpyDeque

class rZigZagBase(object):
    """ZigZag 基础逻辑"""
    def __init__(self, delta, n_cache=1024):
        if isinstance(delta, (list, tuple)):
            self.up_delta, self.down_delta = abs(delta[0]), abs(delta[1])
        else:
            self.up_delta = self.down_delta = abs(delta)
            
        self.i_low = 0
        self.i_high = 0
        self.searching_dir = 0 # 0: initial, 1: searching peak, -1: searching trough
        
        self._Is = NumpyDeque(maxlen=n_cache)  # 极值位置
        self._Ts = NumpyDeque(maxlen=n_cache)  # 极值类型 (1: high, -1: low)
        self._Vs = NumpyDeque(maxlen=n_cache)  # 极值
        self._g_index = -1

    def g_step(self):
        self._g_index += 1
        return self._g_index

    @property
    def g_index(self):
        return self._g_index
