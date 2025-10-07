import numpy as np
from typing import Optional, Dict, Any, Union
from numpy.typing import DTypeLike, NDArray


__all__ = ['NumpyDeque', 'NumPyDeque']

class NumpyDeque(object):
    """
    q = NumpyDeque(100)
    
    ChangeLog:
        - [@2025-09-19] fix `push`
        - [@2025-09-19] 优化 __getitem__ and __setitem__
    """
    default_maxlen = int(1e8)
    def __init__(self, maxlen, dtype=np.float64, buffer_factor=None):
        assert maxlen is None or maxlen >= 1
        assert buffer_factor is None or buffer_factor >= 1.0 
        self._maxlen = maxlen if maxlen is not None else self.default_maxlen
        self._buffer_factor = buffer_factor or self._get_buffer_factor()
        self._cache_size = int(self._maxlen * self._buffer_factor)
        self._dtype = dtype or np.float64
        print('new NumpyDeque', self._maxlen, self._buffer_factor, self._cache_size, self._dtype)
        self._data = np.empty((self._cache_size,), dtype=dtype) 
        print('new NumpyDeque', self._data)
        self._start_idx = 0 
        self._end_idx = 0 
        self._view_cache = None 
    
    def _get_buffer_factor(self): 
        if self._maxlen >= 1e9: 
            return 1.2 
        elif self._maxlen >= 1e6: 
            return 1.5 
        elif self._maxlen >= 10000: 
            return 2.0
        else:
            return 3.0 

    def append(self, value):
        self.push(value)
    
    def extend(self, values):
        """批量添加值"""
        for value in values:
            self.push(value)
    
    @staticmethod
    def infer_dtype(value: Any) -> np.dtype:
        """推断数据类型 | Infer data type
        
        Args:
            value: 要推断类型的值 | Value to infer type from
            
        Returns:
            np.dtype: 推断出的NumPy数据类型 | Inferred NumPy dtype
        """
        if isinstance(value, (np.ndarray, np.generic)):
            return value.dtype
        elif isinstance(value, bool):
            return np.bool_
        elif isinstance(value, int):
            return np.int64
        elif isinstance(value, float):
            return np.float64
        elif isinstance(value, str):
            return np.dtype('O')  # 字符串使用对象类型
        elif isinstance(value, (list, tuple)):
            return np.dtype('O')  # 复合类型使用对象类型
        else:
            return np.dtype('O')  # 其他类型默认使用对象类型

    def push(self, value): 
        self._view_cache = None 
        if self._end_idx >= self._cache_size: 
            num = self._end_idx - self._start_idx 
            self._data[:num] = self._data[self._start_idx:self._end_idx]
            # self._data[num:] = 0 # 如果赋值为np.nan, 如果dtype为int, 会导致数据类型改变
            # self._data[self._start_idx:self._end_idx] = 0  # factor <2 时会导致数据被覆盖, 
            self._start_idx = 0
            self._end_idx = num
        
        # 处理 None 值
        if value is None:
            dtype = self._dtype if hasattr(self._dtype, 'kind') else np.dtype(self._dtype)
            if dtype.kind in ['f', 'c']:  # 浮点数或复数
                value = np.nan
            elif dtype.kind == 'M':  # datetime
                value = np.datetime64('NaT')
            elif dtype.kind == 'm':  # timedelta
                value = np.timedelta64('NaT')
            elif dtype.kind == 'b':  # 布尔
                value = False
            elif dtype.kind in ['i', 'u']:  # 整数
                value = 0
            else:  # 对象类型或其他
                value = None
        
        self._data[self._end_idx] = value
        self._end_idx += 1
        # shift，
        # FIXED: self._end_idx > self._maxlen:
        if self._end_idx - self._start_idx > self._maxlen:
            self._start_idx += 1

    def pop(self):
        self._view_cache = None
        if self._start_idx >= self._end_idx:
            raise IndexError("pop from empty deque")
        value = self._data[self._start_idx]
        self._start_idx += 1
        return value
    
    def clear(self):
        """清空队列"""
        self._view_cache = None
        self._start_idx = 0
        self._end_idx = 0
    
    def resize(self, new_maxlen):
        """调整队列最大长度"""
        if new_maxlen <= 0:
            raise ValueError("new_maxlen must be positive")
        
        # 保存当前数据
        current_data = self.values.copy()
        current_size = len(current_data)
        
        # 更新参数
        self._maxlen = new_maxlen
        self._buffer_factor = self._get_buffer_factor()
        self._cache_size = int(self._maxlen * self._buffer_factor)
        
        # 重新分配数据数组
        self._data = np.empty((self._cache_size,), dtype=self._dtype)
        
        # 如果新长度小于当前数据长度，只保留最新的数据
        if new_maxlen < current_size:
            current_data = current_data[-new_maxlen:]
            current_size = len(current_data)
        
        # 重新填充数据
        self._start_idx = 0
        self._end_idx = current_size
        if current_size > 0:
            self._data[:current_size] = current_data
        
        self._view_cache = None

    @property
    def maxlen(self):
        return self._maxlen

    @property
    def values(self):
        if self._view_cache is None:
            self._view_cache = self._data[self._start_idx:self._end_idx]
        return self._view_cache

    def __array__(self, dtype: Optional[DTypeLike] = None) -> NDArray:
        """NumPy数组接口 | NumPy array interface
        
        允许使用np.array(deque)直接转换 | Allows direct conversion using np.array(deque)
        """
        arr = self.values
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def __array_interface__(self) -> Dict[str, Any]:
        data = self.values
        return {
            'shape': data.shape,
            'typestr': data.dtype.str,
            'descr': data.dtype.descr,
            'data': (data.__array_interface__['data']),
            'strides': None,
            'version': 3,
        }

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return (self._end_idx - self._start_idx,)
    
    def __lt__(self, other: Union['NumpyDeque', NDArray, Any]) -> Union[bool, NDArray]:
        """小于比较运算符

        Args:
            other: 要比较的对象，可以是NumPyDeque、NumPy数组或其他类型

        Returns:
            bool或NDArray: 比较结果
        """
        if isinstance(other, NumpyDeque):
            return self.values < other.values
        return self.values < other

    def __le__(self, other: Union['NumpyDeque', NDArray, Any]) -> Union[bool, NDArray]:
        """小于等于比较运算符"""
        if isinstance(other, NumpyDeque):
            return self.values <= other.values
        return self.values <= other

    def __gt__(self, other: Union['NumpyDeque', NDArray, Any]) -> Union[bool, NDArray]:
        """大于比较运算符"""
        if isinstance(other, NumpyDeque):
            return self.values > other.values
        return self.values > other

    def __ge__(self, other: Union['NumpyDeque', NDArray, Any]) -> Union[bool, NDArray]:
        """大于等于比较运算符""" 
        if isinstance(other, NumpyDeque): 
            return self.values >= other.values 
        return self.values >= other 

    def __eq__(self, other: object) -> Union[bool, NDArray]: 
        """等于比较运算符"""
        if isinstance(other, NumpyDeque): 
            return self.values == other.values 
        return self.values == other 

    def __ne__(self, other: object) -> Union[bool, NDArray]:
        """不等于比较运算符"""
        if isinstance(other, NumpyDeque):
            return self.values != other.values
        return self.values != other

    def __getitem__(self, i):
        # return self._data[self._start_idx + i]
        return self.values[i]
    
    def __getitem_deprecated__(self, i):
        if isinstance(i, int):
            if i < 0:
                i += self._end_idx - self._start_idx 
            # self._end_idx - self._start_idx: 消耗时间远小于len(self) 
            if 0 <= i < self._end_idx - self._start_idx: 
                return self._data[self._start_idx + i] 
            else:
                raise IndexError("index out of range")
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            return self._data[self._start_idx + start : self._start_idx + stop : step]
        else:
            raise TypeError("index must be an integer or a slice")

    def __setitem__(self, i, value):
        self._view_cache = None 
        self.values[i] = value
        
    def __setitem_deprecated__(self, i, value):
        self._view_cache = None
        if isinstance(i, int):
            if i < 0:
                i += self._end_idx - self._start_idx 
            if 0 <= i < self._end_idx - self._start_idx: 
                self._data[self._start_idx + i] = value
            else:
                raise IndexError("index out of range")
        elif isinstance(i, slice):
            start, stop, step = i.indices(len(self))
            self._data[self._start_idx + start : self._start_idx + stop : step] = value
        else:
            raise TypeError("index must be an integer or a slice")

    def __len__(self):
        return self._end_idx - self._start_idx

    def __repr__(self):
        return f"NumPyDeque({self.values.tolist()})"

NumPyDeque = NumpyDeque

def benchmark(n=1000000, size=100000):
    import time, random
    q = NumpyDeque(size)
    for i in range(size):
        q.push(i)

    indices = [random.randint(-size, size - 1) for _ in range(n)]

    # 测试 __getitem__
    start = time.perf_counter()
    s1 = 0
    for idx in indices:
        s1 += q.__getitem_deprecated__(idx)
    t1 = time.perf_counter() - start

    # 测试 __getitem2__
    start = time.perf_counter()
    s2 = 0
    for idx in indices:
        s2 += q.__getitem__(idx)
    t2 = time.perf_counter() - start

    print(f"__getitem_deprecated__ (direct)     : {t1:.6f}s, sum={s1}")
    print(f"__getitem__  (via values) : {t2:.6f}s, sum={s2}")

def benchmark_set(n=500000, size=10000):
    import time, random
    q = NumpyDeque(size)
    for i in range(size):
        q.push(i)

    indices = [random.randint(-size, size - 1) for _ in range(n)]
    values = [random.random() for _ in range(n)]

    # 测试 __setitem__
    start = time.perf_counter()
    for idx, val in zip(indices, values):
        q.__setitem_deprecated__(idx, val)
    t1 = time.perf_counter() - start

    # 测试 __setitem2__
    start = time.perf_counter()
    for idx, val in zip(indices, values):
        q.__setitem__(idx, val)
    t2 = time.perf_counter() - start

    print(f"__setitem__  (via values) : {t1:.6f}s")
    print(f"__setitem_deprecated__ (direct)     : {t2:.6f}s")
    
if __name__ == "__main__":
    if 1:
        benchmark(n=5000000, size=10000)
    if 1:
        benchmark_set(n=5000000, size=10000)
    if 0:
        q = NumpyDeque(5)
        q.push(1)
        q.push(2)
        q.push(3)
        q.push(4)
        q.push(5)
        print(q)
        print(q[1:3])
        print(q[1:-1])
    if 0:
        q = NumpyDeque(3, buffer_factor=2)
        q.push(1)
        q.push(2)
        q.push(3)
        q.pop()
        q.push(4)
        print(q)
        print(q.values) # XXX 
        print(q[-1])
    if 0:
        q = NumpyDeque(3)
        print(q)
        q.push(1)
        assert(q[0] == 1)
        assert(len(q) == 1)
        print(q)
        q.push(2)
        assert(q[0] == 1)
        assert(q[1] == 2)
        assert(len(q) == 2)
        print(q)
        q.push(3)
        assert(q[0] == 1)
        assert(q[1] == 2)
        assert(q[2] == 3)
        assert(len(q) == 3)
        print(q)
        q.push(4)
        assert(q[0] == 2)
        assert(q[1] == 3)
        assert(q[2] == 4)
        assert(len(q) == 3)
        print(q)
    if 0:
        q = NumpyDeque(2, buffer_factor=2)
        q.push(1)
        q.push(2)
        q.push(3)
        q.push(4)
        q.push(5)
        q.push(6)
        print(q)
    if 0:
        q = NumpyDeque(3, buffer_factor=2)
        q.push(1)
        q.push(2)
        q.push(3)
        print(q)
        q.pop()
        q.pop()
        print(q)
        print(q[0])
    if 0:
        q = NumpyDeque(maxlen=None)
        for i in range(int(1e6)):
            q.push(i)
            print(len(q))
