import numpy as np
from typing import Optional, Dict, Any, Union
from numpy.typing import DTypeLike, NDArray


__all__ = ['NumpyDeque', 'NumPyDeque']

class NumpyDeque(object):
    """NumPy-backed deque with circular buffer and dynamic resizing."""
    def __init__(self, maxlen, dtype=np.float64, buffer_factor=None):
        assert maxlen is None or maxlen >= 1
        assert buffer_factor is None or buffer_factor >= 1.0 
        self._maxlen = maxlen 
        
        # 获取缓冲区因子
        self._buffer_factor = buffer_factor or self._get_buffer_factor()
        
        # 计算初始缓存大小
        if self._maxlen is None:
            # 无限模式：初始分配一个较小值
            self._cache_size = 1024 
        else:
            self._cache_size = int(self._maxlen * self._buffer_factor)
            
        self._dtype = dtype or np.float64
        self._data = np.empty((self._cache_size,), dtype=dtype) 
        self._start_idx = 0 
        self._end_idx = 0 
        self._view_cache = None 
    
    def _get_buffer_factor(self): 
        if self._maxlen is None:
            return 2.0
        if self._maxlen >= 1e9: 
            return 1.2 
        elif self._maxlen >= 1e6: 
            return 1.5 
        elif self._maxlen >= 10000: 
            return 2.0
        else:
            return 3.0 
    
    def is_empty(self):
        return self._end_idx == self._start_idx
    
    def is_full(self):
        if self._maxlen is None:
            return False
        return self._end_idx - self._start_idx == self._maxlen

    def _ensure_capacity(self, num_new):
        """确保能够容纳新元素 | Ensure cache can accommodate new elements"""
        if self._end_idx + num_new > self._cache_size:
            num_current = self._end_idx - self._start_idx
            
            if self._maxlen is None:
                # 无限模式：检查是否需要扩容
                # 如果当前有效数据已占缓存 50% 以上且加上新元素后溢出，则扩容
                if num_current + num_new > self._cache_size:
                    new_size = max(self._cache_size * 2, num_current + num_new + 1024)
                    new_data = np.empty((new_size,), dtype=self._dtype)
                    if num_current > 0:
                        new_data[:num_current] = self._data[self._start_idx : self._end_idx]
                    self._data = new_data
                    self._cache_size = new_size
                    self._start_idx = 0
                    self._end_idx = num_current
                else:
                    # 仅平移即可
                    if num_current > 0:
                        self._data[:num_current] = self._data[self._start_idx : self._end_idx]
                    self._start_idx = 0
                    self._end_idx = num_current
            else:
                # 固定长度模式：平移并维护 maxlen
                if num_current + num_new > self._maxlen:
                    skip = (num_current + num_new) - self._maxlen
                    self._start_idx += skip
                    num_current = self._end_idx - self._start_idx
                
                if num_current > 0:
                    self._data[:num_current] = self._data[self._start_idx : self._end_idx]
                self._start_idx = 0
                self._end_idx = num_current

    def append(self, value): 
        """添加单个值到末尾 | Add a single value to the end"""
        self._view_cache = None 
        self._ensure_capacity(1)
        
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
        # shift
        if self._maxlen is not None and self._end_idx - self._start_idx > self._maxlen:
            self._start_idx += 1

    def extend(self, values):
        """批量添加值 | Batch add values (Optimized)"""
        self._view_cache = None
        vals = np.asanyarray(values, dtype=self._dtype)
        num_new = len(vals)
        if num_new == 0:
            return

        # 如果新数据已经超过 maxlen，只保留最后 maxlen 个
        if self._maxlen is not None and num_new > self._maxlen:
            vals = vals[-self._maxlen:]
            num_new = self._maxlen

        # 确保空间
        self._ensure_capacity(num_new)

        # 批量写入
        self._data[self._end_idx : self._end_idx + num_new] = vals
        self._end_idx += num_new

        # 维护 maxlen 约束
        if self._maxlen is not None and self._end_idx - self._start_idx > self._maxlen:
            self._start_idx = self._end_idx - self._maxlen
    
    @staticmethod
    def infer_dtype(value: Any) -> np.dtype:
        """推断数据类型 | Infer data type"""
        if isinstance(value, (np.ndarray, np.generic)):
            return value.dtype
        elif isinstance(value, bool):
            return np.bool_
        elif isinstance(value, int):
            return np.int64
        elif isinstance(value, float):
            return np.float64
        elif isinstance(value, str):
            return np.dtype('O')
        elif isinstance(value, (list, tuple)):
            return np.dtype('O')
        else:
            return np.dtype('O')

    def popleft(self):
        """移除并返回最左侧(最早)的元素 | FIFO"""
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
        """调整队列大小"""
        if new_maxlen is not None and new_maxlen <= 0:
            raise ValueError("new_maxlen must be positive")
        
        # 保存当前数据
        current_data = self.values.copy()
        current_size = len(current_data)
        
        # 更新参数
        self._maxlen = new_maxlen
        self._buffer_factor = self._get_buffer_factor()
        if self._maxlen is None:
            self._cache_size = max(current_size + 1024, 1024)
        else:
            self._cache_size = int(self._maxlen * self._buffer_factor)
        
        # 重新分配数据数组
        self._data = np.empty((self._cache_size,), dtype=self._dtype)
        
        # 如果新长度小于当前数据长度，只保留最新的数据
        if self._maxlen is not None and self._maxlen < current_size:
            current_data = current_data[-self._maxlen:]
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
        n = len(self)
        if n > 16:
            # 缩略显示大数据量
            data_str = f"[{', '.join(map(str, self.values[:8].tolist()))}, ..., {', '.join(map(str, self.values[-8:].tolist()))}]"
        else:
            data_str = str(self.values.tolist())
        return f"NumpyDeque({data_str}, len={n}, maxlen={self._maxlen}, dtype={self.dtype})"


NumPyDeque = NumpyDeque

# benchmark
def benchmark(n=1000000, size=100000):
    import time, random
    q = NumpyDeque(size)
    for i in range(size):
            q.append(i)

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
            q.append(i)

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

    print(f"__setitem_deprecated__ (direct)     : {t1:.6f}s")
    print(f"__setitem__  (via values) : {t2:.6f}s")
    
if __name__ == "__main__":
    if 1:
        benchmark(n=5000000, size=10000)
    if 1:
        benchmark_set(n=5000000, size=10000)
    if 0:
        q = NumpyDeque(5)
        q.append(1)
        q.append(2)
        q.append(3)
        q.append(4)
        q.append(5)
        print(q)
        print(q[1:3])
        print(q[1:-1])
    if 0:
        q = NumpyDeque(3, buffer_factor=2)
        q.append(1)
        q.append(2)
        q.append(3)
        q.popleft()
        q.append(4)
        print(q)
        print(q.values) 
        print(q[-1])
    if 0:
        q = NumpyDeque(3)
        print(q)
        q.append(1)
        assert(q[0] == 1)
        assert(len(q) == 1)
        print(q)
        q.append(2)
        assert(q[0] == 1)
        assert(q[1] == 2)
        assert(len(q) == 2)
        print(q)
        q.append(3)
        assert(q[0] == 1)
        assert(q[1] == 2)
        assert(q[2] == 3)
        assert(len(q) == 3)
        print(q)
        q.append(4)
        assert(q[0] == 2)
        assert(q[1] == 3)
        assert(q[2] == 4)
        assert(len(q) == 3)
        print(q)
    if 0:
        q = NumpyDeque(2, buffer_factor=2)
        q.append(1)
        q.append(2)
        q.append(3)
        q.append(4)
        q.append(5)
        q.append(6)
        print(q)
    if 0:
        q = NumpyDeque(3, buffer_factor=2)
        q.append(1)
        q.append(2)
        q.append(3)
        print(q)
        q.popleft()
        q.popleft()
        print(q)
        print(q[0])
    if 0:
        q = NumpyDeque(maxlen=None)
        for i in range(int(1e6)):
            q.append(i)
            print(len(q))
