import numpy as np
from gymnasium.spaces import Box# , Discrete
from ...base import rIndicator
from ...utils.deque import NumPyDeque


class rZigZagBase(rIndicator):
    
    _default_confirmed_at = -1

    def __init__(self, delta, zigzag_buffer_size=10_000, **kwargs):
        if isinstance(delta, (list, tuple)):
            assert len(delta) == 2
            assert delta[0] > 0 and delta[1] > 0
            self.up_delta, self.down_delta = abs(delta[0]), abs(delta[1])
        else:
            assert delta > 0
            self.up_delta, self.down_delta = abs(delta), abs(delta)
        self.delta = delta
        self.i_low = 0
        self.i_high = 0
        self.searching_dir = 0
        self._Is = NumPyDeque(zigzag_buffer_size, dtype=np.int32) # 位置
        self._Ts = NumPyDeque(zigzag_buffer_size, dtype=np.int32) # 是最大值还是最小值
        self._Vs = NumPyDeque(zigzag_buffer_size, dtype=np.float64) 
        self._ICs = NumPyDeque(zigzag_buffer_size, dtype=np.int32) # 位置
        self._YCs = NumPyDeque(zigzag_buffer_size, dtype=np.float64) # 值
        rIndicator.__init__(self,
                             window=2,
                             num_outputs=4,
                             output_dtypes=[int]*4,
                             output_keys=['confirmed_at', 'searching_dir', 'i_high', 'i_low'],
                             output_spaces=[
                                Box(low=-1, high=np.inf, shape=(), dtype=np.int32),
                                Box(low=-1, high=1, shape=(), dtype=np.int32),
                                Box(low=0, high=np.inf, shape=(), dtype=np.int32),
                                Box(low=0, high=np.inf, shape=(), dtype=np.int32),
                             ],
                             **kwargs
                             )
    @property
    def recent_Is(self):
        return self._Is.values

    @property
    def recent_Ts(self):
        return self._Ts.values

    @property
    def recent_Vs(self):
        return self._Vs.values
    
    @property
    def recent_Ys(self):
        return self._Vs.values

    @property
    def recent_ICs(self):
        return self._ICs.values
    
    @property
    def recent_YCs(self):
        return self._YCs.values

    @property
    def null_output(self):
        return [self._default_confirmed_at, self.searching_dir, self.i_high, self.i_low]