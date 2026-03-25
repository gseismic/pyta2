import numpy as np

class MovingVector(object):
    """
    基于列表的滑动向量，支持全局索引定位和按需保留最近数据。
    
    特点：
    1. 支持绝对索引：索引 i 从对象创建起计，即便部分数据已丢弃。
    2. 灵活滑动：通过 rekeep_n 手动控制内存占用。
    3. NumPy 兼容：可以直接使用 np.array(mv) 转换。
    """
    def __init__(self): 
        self._discard_end = 0 
        self._values = [] 
        self._notional_len = 0 

    def append(self, value): 
        self._notional_len += 1 
        self._values.append(value) 

    def extend(self, values):
        """批量添加元素"""
        if values:
            self._notional_len += len(values)
            self._values.extend(values)

    def rekeep_n(self, n: int):
        '''只保留最近的 n 个点的数据, 其余丢弃'''
        if n == 0:
            self._discard_end = self._notional_len
            self._values = []
            return
            
        current_len = len(self._values)
        if current_len > n:
            self._discard_end += current_len - n
            self._values = self._values[-n:]

    def __getitem__(self, i):
        # 负索引保持相对于末尾的行为
        if i < 0:
            return self._values[i]
            
        # 绝对索引逻辑
        if i - self._discard_end >= 0:
            try:
                return self._values[i - self._discard_end]
            except IndexError:
                raise IndexError(f"Index {i} out of range (notional_len={self._notional_len})")
        else:
            raise IndexError(f"Index {i} already discarded (discard_end={self._discard_end})")

    def __setitem__(self, i, value):
        if i < 0:
            self._values[i] = value
            return
            
        if i - self._discard_end >= 0:
            try:
                self._values[i - self._discard_end] = value
            except IndexError:
                raise IndexError(f"Index {i} out of range (notional_len={self._notional_len})")
        else:
            raise IndexError(f"Index {i} already discarded (discard_end={self._discard_end})")

    def __array__(self, dtype=None, copy=None):
        """支持直接转换为 NumPy 数组 (适配 NumPy 2.0 协议)"""
        return np.array(self._values, dtype=dtype, copy=copy)

    def __len__(self):
        """返回当前内存中保留的长度"""
        return len(self._values)

    @property
    def discard_end(self):
        """已被丢弃的数据数量（起始索引偏移）"""
        return self._discard_end

    @property
    def notional_len(self):
        """概念上的总长度（自创建以来 append 的总次数）"""
        return self._notional_len

    @property
    def kept_len(self):
        """当前缓存中的数据长度"""
        return len(self._values)

    @property
    def kept_values(self):
        return self._values

    def __repr__(self): 
        # 截断打印，防止大数据量下终端假死
        if len(self._values) > 6:
            v_str = f"[{self._values[0]}, {self._values[1]}, {self._values[2]}, ..., {self._values[-1]}]"
        else:
            v_str = str(self._values)
        return (f'MovingVector(notional_len={self._notional_len}, '
                f'kept_len={len(self._values)}, data={v_str})')
