import numpy as np
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Union, Sequence
from numpy.typing import NDArray, DTypeLike
from .numpy_deque import NumpyDeque

class DequeTable:
    """基于NumpyDeque的高效字典队列，支持列式存储和访问"""
    
    def __init__(
        self, 
        maxlen: Optional[int],
        buffer_factor: Optional[float] = None,
        dtypes: Optional[Dict[str, DTypeLike]] = None
    ) -> None:
        self._maxlen = NumpyDeque.default_maxlen if maxlen is None else maxlen
        self._dtypes = dtypes or {}
        self._buffer_factor = buffer_factor or 1.5
        self._columns: OrderedDict[str, NumpyDeque] = OrderedDict()
        self._size: int = 0
        self._fill_values: Dict[str, Any] = {}

        
    @staticmethod
    def get_fill_value(dtype: np.dtype) -> Any:
        """获取对应数据类型的填充值 | Get fill value for corresponding dtype"""
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min  # 使用该类型的最小值
        if np.issubdtype(dtype, np.floating):
            return np.nan
        if np.issubdtype(dtype, np.bool_):
            return False
        if np.issubdtype(dtype, np.datetime64):
            return np.datetime64('NaT')
        if np.issubdtype(dtype, np.timedelta64):
            return np.timedelta64('NaT')
        return None  # 其他类型使用None
    
    def ensure_column(self, key: str, dtype: Optional[DTypeLike] = None) -> None:
        """确保列存在"""
        if key in self._columns:
            return
        
        dtype = dtype or self._dtypes.get(key)
        self._dtypes[key] = dtype 
        print('new column', key, dtype)
        self._columns[key] = NumpyDeque(
            maxlen=self._maxlen,
            dtype=dtype,
            buffer_factor=self._buffer_factor
        )
        print('nnnew')
        # 获取填充值 | Get fill value
        if dtype is not None:
            self._fill_values[key] = self.get_fill_value(np.dtype(dtype))
        # 填充已有行 | Fill existing rows
        if self._size > 0:
            print('fill existing rows', key, self._size)
            fill_value = self._fill_values.get(key) 
            self._columns[key].extend([fill_value] * self._size) 

    def append(self, data: Dict[str, Any]) -> None:
        """添加一行数据"""
        # 确保所有列都存在
        for key, value in data.items():
            if key not in self._columns:
                dtype = self._dtypes.get(key)
                if dtype is None:
                    dtype = NumpyDeque.infer_dtype(value)
                self.ensure_column(key, dtype)
                
        for key in self._columns:
            if key not in data:
                self._columns[key].append(self._fill_values[key])
            else:
                self._columns[key].append(data[key])
                
        if self._size < self._maxlen:
            self._size += 1

    def extend(self, data: Dict[str, Sequence[Any]]) -> None:
        """批量添加数据"""
        if not data:
            return
            
        # 获取数据长度并检查一致性
        lengths = {len(v) for v in data.values()}
        if not lengths:
            return
        if len(lengths) != 1:
            raise ValueError("All columns must have the same length")
        length = lengths.pop()
        
        # 确保所有列都存在
        for key, values in data.items():
            if key not in self._columns:
                dtype = self._dtypes.get(key)
                if dtype is None:
                    dtype = NumpyDeque.infer_dtype(values[0])
                self.ensure_column(key, dtype)
                
        # 批量添加数据
        for key in self._columns:
            fill_value = self._fill_values.get(key)
            values = data.get(key, [fill_value] * length)
            self._columns[key].extend(values)
            
        self._size = min(self._size + length, self._maxlen)

    def get_column(self, key: str) -> NDArray:
        """获取列数据"""
        if key not in self._columns:
            raise KeyError(f"Column {key} does not exist")
        return self._columns[key].values

    def __getitem__(self, key: Union[str, int]) -> Union[NDArray, Dict[str, Any]]:
        """支持列访问和行访问"""
        if isinstance(key, str):
            return self.get_column(key)
        elif isinstance(key, int):
            # 行访问
            if key < 0:
                key += self._size
            if key < 0 or key >= self._size:
                raise IndexError("Index out of range")
            return {k: v.values[key] for k, v in self._columns.items()}
        raise TypeError("Key must be str or int")
    
    def __iter__(self):
        """行迭代器"""
        for i in range(self._size):
            yield {k: v.values[i] for k, v in self._columns.items()}
    
    def __reversed__(self):
        """反向行迭代器"""
        for i in range(self._size-1, -1, -1):
            yield {k: v.values[i] for k, v in self._columns.items()}

    def to_list(self) -> List[Dict[str, Any]]:
        """转换为字典列表"""
        return list(self)
    
    def to_dict(self) -> Dict[str, NDArray]:
        """转换为列字典"""
        return {k: v.values for k, v in self._columns.items()}
    
    def clear(self) -> None:
        """清空数据"""
        for deque in self._columns.values():
            deque.clear()
        self._size = 0

    def __len__(self) -> int:
        """返回行数"""
        return self._size

    @property
    def columns(self) -> List[str]:
        """返回列名列表"""
        return list(self._columns.keys())

    @property
    def maxlen(self) -> int:
        """返回最大长度"""
        return self._maxlen

    @property
    def dtypes(self) -> Dict[str, np.dtype]:
        """返回列数据类型"""
        return self._dtypes
        # return {k: v.dtype for k, v in self._columns.items()}

    def __repr__(self) -> str:
        return (
            f"DequeTable(size={self._size}, columns={self.columns}, dtypes={self.dtypes},"
            f"maxlen={self._maxlen}, data=\n{self.to_list()})"
        )

    def resize(self, new_maxlen: int) -> None:
        """调整队列大小"""
        if new_maxlen <= 0:
            raise ValueError("new_maxlen must be positive") 
            
        for deque in self._columns.values():
            deque.resize(new_maxlen)
            
        self._maxlen = new_maxlen
        self._size = min(self._size, new_maxlen)