import numpy as np
from .base import Space

class Box(Space):
    """
    表示连续或离散数值空间的Box类。
    
    用于定义多维数值空间，支持连续和离散数值范围。
    
    Parameters:
    -----------
    low : array_like
        空间的下界
    high : array_like  
        空间的上界
    shape : tuple, optional
        空间的形状，如果为None则从low/high推断
    dtype : numpy.dtype, optional
        数据类型，默认为np.float64
        
    Examples:
    ---------
    >>> # 创建2D连续空间
    >>> space = Box(low=[0, 0], high=[1, 1])
    >>> sample = space.sample()
    >>> print(space.contains(sample))
    True
    """
    def __init__(self, low, high, shape=None, dtype=np.float64, null_value=None):
        self.low = np.asarray(low, dtype=dtype)
        self.high = np.asarray(high, dtype=dtype)
        if null_value is not None:
                self.null_value = null_value
        else:
            if dtype in [np.float64, np.float32]:
                self.null_value = np.nan
            elif dtype in [np.int32, np.int64]:
                self.null_value = -999999
            else:
                self.null_value = None
        
        # 验证边界
        if np.any(self.low > self.high):
            raise ValueError("All low values must be <= high values")
        
        # 确定形状
        if shape is None:
            self.shape = self.low.shape
        else:
            self.shape = tuple(shape)
            if self.low.shape != () and self.low.shape != self.shape:
                raise ValueError("Shape mismatch between low and given shape")
            if self.high.shape != () and self.high.shape != self.shape:
                raise ValueError("Shape mismatch between high and given shape")
        
        super().__init__(dtype)
    
    def sample(self):
        if np.issubdtype(self.dtype, np.integer):
            return np.random.randint(low=self.low, high=self.high+1, size=self.shape)
        else:
            return np.random.uniform(low=self.low, high=self.high, size=self.shape)

    def contains(self, x):
        x = np.asarray(x, dtype=self.dtype)
        return (
            x.shape == self.shape and
            np.all(x >= self.low) and 
            np.all(x <= self.high)
        )
    
    def is_null(self, x):
        """检查给定值是否为null值"""
        x = np.asarray(x, dtype=self.dtype)
        if self.null_value is None:
            return False
        return np.isnan(x) if (np.isnan(self.null_value) and np.issubdtype(self.dtype, np.floating)) else np.equal(x, self.null_value)
    
    def set_null_value(self, null_value):
        """设置新的null值"""
        self.null_value = null_value
    
    def get_null_value(self):
        """获取当前的null值"""
        return self.null_value

    def __repr__(self):
        return (f"Box(low={self.low}, high={self.high}, "
                f"shape={self.shape}, dtype={self.dtype})")
        
    def to_json(self):
        return {
            "type": "Box",
            "low": self.low.tolist(),
            "high": self.high.tolist(),
            "shape": self.shape,
            "dtype": self.dtype
        }
    
    @classmethod
    def from_json(cls, data):   
        return cls(low=data["low"], high=data["high"], shape=data["shape"], dtype=data["dtype"])


class PositiveBox(Box):
    """表示非负数值空间的Box类。"""
    def __init__(self, high=None, shape=None, dtype=np.float64):
        super(PositiveBox, self).__init__(low=0, high=high, shape=shape, dtype=dtype)

class NegativeBox(Box):
    """表示非正数值空间的Box类。"""
    def __init__(self, low=None, shape=None, dtype=np.float64):
        super(NegativeBox, self).__init__(low=low, high=0, shape=shape, dtype=dtype)

class PositiveScalar(PositiveBox):
    """表示非负标量空间的类。"""
    def __init__(self, high=None, dtype=np.float64):
        super(PositiveScalar, self).__init__(high=high, low=0, shape=(), dtype=dtype)

class NegativeScalar(NegativeBox):
    """表示非正标量空间的类。"""
    def __init__(self, low=None, dtype=np.float64):
        super(NegativeScalar, self).__init__(high=0, low=low, shape=(), dtype=dtype)

class Scalar(Box):
    """表示标量空间的类。"""
    def __init__(self, high=None, low=None, dtype=np.float64, **kwargs):
        super(Scalar, self).__init__(high=high, low=low, shape=(), dtype=dtype, **kwargs)


__all__ = ['Box', 'Scalar', 'PositiveScalar', 'NegativeScalar', 'PositiveBox', 'NegativeBox']