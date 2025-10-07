import numpy as np
from ..base import rIndicator
from ..base.schema import Schema
from ..utils.space.box import Scalar
from ..utils.space.discrete import Category
from ..utils.deque.numpy_deque import NumpyDeque
from ..trend.ma import get_ma_class

class rTwinCross(rIndicator):
    name = 'TwinCross'

    def __init__(self, obj1, obj2, stride=1, **kwargs):
        assert stride > 1, f'{self.name} stride must be greater than 1, got {stride}'
        self.stride = stride
        self.obj1 = obj1
        self.obj2 = obj2
        self.values1 = NumpyDeque(maxlen=stride+1)
        self.values2 = NumpyDeque(maxlen=stride+1)
        super(rTwinCross, self).__init__(
            window=max(obj1.window, obj2.window) + stride,
            schema=Schema([
                ('direction', Category([-1, 0, 1])),
                ('value1', Scalar()),
                ('value2', Scalar()),
            ]),
            **kwargs
        )
    
    def reset_extras(self):
        self.values1.clear()
        self.values2.clear()
    
    def forward(self, values):
        v1 = self.obj1.rolling(values)
        v2 = self.obj2.rolling(values)
        self.values1.append(v1)
        self.values2.append(v2)
        if len(values) < self.required_window:
            return 0, np.nan, np.nan
        
        k = self.stride
        if (self.values1[-1] > self.values2[-1] and self.values1[-1-k] < self.values2[-1-k]):
            cross_direction = 1
        elif (self.values1[-1] < self.values2[-1] and self.values1[-1-k] > self.values2[-1-k]):
            cross_direction = -1
        else:
            cross_direction = 0
        return cross_direction, v1, v2
        
    @property
    def full_name(self):
        return f'{self.name}({self.obj1.full_name},{self.obj2.full_name},{self.stride})'
    
class rMACross(rTwinCross):
    name = 'MACross'

    def __init__(self, l, xl, ma_type='EMA', stride=1, **kwargs):
        assert l < xl, f'{self.name} l must be less than xl, got {l} and {xl}'
        self.ma_type = ma_type
        self.l = l
        self.xl = xl
        self.stride = stride
        ma_cls = get_ma_class(ma_type)
        super(rMACross, self).__init__(ma_cls(l), ma_cls(xl), stride=stride, **kwargs)
    
    @property
    def full_name(self):
        return f'{self.name}({self.ma_type},{self.l},{self.xl},{self.stride})'