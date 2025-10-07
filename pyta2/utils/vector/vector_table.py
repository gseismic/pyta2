from typing import Dict, Any, Optional, List, Union, Sequence, Tuple
import numpy as np
from collections import OrderedDict
from numpy.typing import NDArray, DTypeLike
from .numpy_vector import NumpyVector


class VectorTable:
    """基于NumpyVector的表格向量，支持高效的列式存储和访问
    Table-structured vectors based on NumpyVector, supports efficient columnar storage and access
    """
    
    def __init__(
        self,
        buffer_factor: float = 2.0,
        dtypes: Optional[Dict[str, DTypeLike]] = None
    ) -> None:
        """初始化向量表 | Initialize vector table
        
        Args:
            buffer_factor: 缓冲区大小倍数 | Buffer size multiplier
            dtypes: 列数据类型字典 | Column dtype dictionary
        """
        self._buffer_factor = buffer_factor
        self._dtypes = dtypes or {}
        self._columns: OrderedDict[str, NumpyVector] = OrderedDict()
        self._size: int = 0
        self._fill_values: Dict[str, Any] = {}
        
    def _get_fill_value(self, dtype: np.dtype) -> Any:
        """获取对应数据类型的填充值 | Get fill value for corresponding dtype"""
        if np.issubdtype(dtype, np.integer):
            return np.iinfo(dtype).min
        if np.issubdtype(dtype, np.floating):
            return np.nan
        if np.issubdtype(dtype, np.bool_):
            return False
        if np.issubdtype(dtype, np.datetime64):
            return np.datetime64('NaT')
        if np.issubdtype(dtype, np.timedelta64):
            return np.timedelta64('NaT')
        return None
        
    def _infer_dtype(self, value: Any) -> np.dtype:
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

    def ensure_column(self, key: str, dtype: Optional[DTypeLike] = None) -> None:
        """确保列存在 | Ensure column exists"""
        if key not in self._columns:
            if dtype is None and key in self._dtypes:
                dtype = self._dtypes[key]
            
            self._columns[key] = NumpyVector(
                dtype=dtype,
                buffer_factor=self._buffer_factor
            )
            
            if dtype is not None:
                self._fill_values[key] = self._get_fill_value(np.dtype(dtype))
                
            if self._size > 0:
                fill_value = self._fill_values.get(key)
                self._columns[key].resize(self._size, fill_value)

    def append(self, data: Dict[str, Any]) -> None:
        """添加一行数据 | Append a row of data"""
        for key, value in data.items():
            if key not in self._columns:
                dtype = self._dtypes.get(key) or self._infer_dtype(value)
                self.ensure_column(key, dtype)
                if key not in self._fill_values:
                    self._fill_values[key] = self._get_fill_value(dtype)
                
        for key in self._columns:
            if key not in data:
                self._columns[key].append(self._fill_values[key])
            else:
                self._columns[key].append(data[key])
                
        self._size += 1

    def extend(self, data: Dict[str, Sequence[Any]]) -> None:
        """批量添加数据 | Extend with multiple rows"""
        lengths = {len(v) for v in data.values()}
        if not lengths:
            return
        if len(lengths) != 1:
            raise ValueError("All columns must have the same length")
        length = lengths.pop()
        
        for key, values in data.items():
            if key not in self._columns:
                first_value = next((v for v in values if v is not None), None)
                dtype = self._dtypes.get(key)
                if dtype is None and first_value is not None:
                    dtype = self._infer_dtype(first_value)
                self.ensure_column(key, dtype)
                if key not in self._fill_values and dtype is not None:
                    self._fill_values[key] = self._get_fill_value(dtype)
                
        for key in self._columns:
            if key not in data:
                fill_value = self._fill_values[key]
                self._columns[key].extend([fill_value] * length)
            else:
                self._columns[key].extend(data[key])
                
        self._size += length

    def get_column(self, key: str) -> NDArray:
        """获取列数据（视图）| Get column data (view)"""
        if key not in self._columns:
            raise KeyError(f"Column {key} does not exist")
        return self._columns[key].values

    def __getitem__(self, key: Union[str, slice]) -> Union[NDArray, Dict[str, NDArray]]:
        """支持列访问和切片 | Support column access and slicing"""
        if isinstance(key, str):
            return self.get_column(key)
        return {k: v.values[key] for k, v in self._columns.items()}
    
    def iloc(self, key: int):
        """按行索引访问 | Access by row index"""
        return {k: v.values[key] for k, v in self._columns.items()}
    
    def loc(self, key: Union[str, slice]):
        """按列名访问 | Access by column name"""
        return self.get_column(key)
    
    def iterrows(self, reverse: bool = False):
        """按行迭代 | Iterate by row"""
        for i in range(self._size-1, -1, -1) if reverse else range(self._size):
            yield self.iloc(i)
        
    def __iter__(self):
        """迭代 | Iterate"""
        return self.iterrows()
    
    def itercols(self):
        """按列迭代 | Iterate by column"""
        for k, v in self._columns.items():
            yield k, v.values

    def __setitem__(self, key: str, value: Union[Any, Sequence[Any]]) -> None:
        """设置列数据 | Set column data"""
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) != self._size:
                raise ValueError(f"Length mismatch: got {len(value)}, expected {self._size}")
            if key not in self._columns:
                dtype = self._dtypes.get(key) or self._infer_dtype(value[0])
                self.ensure_column(key, dtype)
            self._columns[key].values[:] = value
        else:
            if key not in self._columns:
                dtype = self._dtypes.get(key) or self._infer_dtype(value)
                self.ensure_column(key, dtype)
            self._columns[key].values[:] = value

    def to_list(self, reverse: bool = False) -> List[Dict[str, Any]]:
        """转换为字典列表 | Convert to list of dictionaries"""
        if reverse:
            return [
                {key: col.values[i] for key, col in self._columns.items()}
                for i in range(self._size-1, -1, -1)
            ]
        else:
            return [
                {key: col.values[i] for key, col in self._columns.items()}
                for i in range(self._size)
            ]
    
    def to_dict(self, reverse: bool = False) -> Dict[str, NDArray]:
        """转换为字典 | Convert to dictionary"""
        return {k: v.values[::-1] if reverse else v.values for k, v in self._columns.items()}
    
    def to_polars(self, reverse: bool = False, infer_schema_length: Optional[int] = None):
        """转换为polars DataFrame | Convert to polars DataFrame"""
        import polars as pl
        return pl.DataFrame(self.to_list(reverse=reverse), infer_schema_length=infer_schema_length)
    
    def to_pandas(self, reverse: bool = False):
        """转换为pandas DataFrame | Convert to pandas DataFrame"""
        import pandas as pd
        return pd.DataFrame(self.to_list(reverse=reverse))
    
    def to_numpy(self, reverse: bool = False):
        """转换为NumPy数组 | Convert to NumPy array"""
        return np.array(self.to_list(reverse=reverse))
    
    def to_torch(self, reverse: bool = False):
        """转换为PyTorch张量 | Convert to PyTorch tensor"""
        import torch
        return torch.tensor(self.to_list(reverse=reverse))
    
    def clear(self) -> None:
        """清空数据 | Clear all data"""
        for vector in self._columns.values():
            vector.clear()
        self._size = 0

    def __len__(self) -> int:
        """返回行数 | Return number of rows"""
        return self._size
    
    @property
    def shape(self) -> Tuple[int, int]:
        """返回表格的形状 | Return the shape of the table"""
        return self._size, len(self._columns)

    @property
    def columns(self) -> List[str]:
        """返回列名列表 | Return list of column names"""
        return list(self._columns.keys())

    @property
    def dtypes(self) -> Dict[str, np.dtype]:
        """返回列数据类型 | Return column dtypes"""
        return {k: v.dtype for k, v in self._columns.items()}

    def __repr__(self) -> str:
        """字符串表示 | String representation"""
        return f"VectorTable(size={self._size}, columns={self.columns})"

    def resize(self, new_size: int, fill_value: Optional[Dict[str, Any]] = None) -> None:
        """调整向量大小 | Resize vector
        
        Args:
            new_size: 新的大小 | New size
            fill_value: 各列的填充值字典 | Fill value dictionary for columns
        """
        if new_size < 0:
            raise ValueError("new_size must be positive")
            
        fill_dict = fill_value or {}
        for key, vector in self._columns.items():
            fill_val = fill_dict.get(key, self._fill_values.get(key))
            vector.resize(new_size, fill_val)
            
        self._size = new_size

    def update(self, start: int, data: Dict[str, Sequence[Any]]) -> None:
        """更新指定位置开始的一段数据 | Update a range of data starting from specified position
        
        Args:
            start: 起始位置 | Start position
            data: 新的数据字典 | New data dictionary
        """
        if start < 0:
            start += self._size
        if not 0 <= start <= self._size:
            raise IndexError("start index out of range")
            
        max_length = max(len(v) for v in data.values())
        if start + max_length > self._size:
            raise ValueError("update would extend beyond current size")
            
        for key, values in data.items():
            if key not in self._columns:
                dtype = self._dtypes.get(key) or self._infer_dtype(values[0])
                self.ensure_column(key, dtype)
            self._columns[key].update(start, values)

    def is_fill_value(self, key: str, value: Any) -> bool:
        """判断值是否为填充值 | Check if value is fill value"""
        if key not in self._columns:
            raise KeyError(f"Column {key} does not exist")
            
        if key not in self._fill_values:
            raise ValueError(f"Column {key} has no fill value")
        
        fill_value = self._fill_values[key]
        dtype = self._columns[key].dtype
        
        if np.issubdtype(dtype, np.floating):
            return np.isnan(value)
        elif np.issubdtype(dtype, np.integer):
            return value == np.iinfo(dtype).min
        elif np.issubdtype(dtype, np.bool_):
            return value == False
        elif np.issubdtype(dtype, (np.datetime64, np.timedelta64)):
            return np.isnat(value)
        else:
            return value is None

    def get_fill_mask(self, key: str) -> np.ndarray:
        """获取列的填充值掩码 | Get fill value mask for column"""
        if key not in self._columns:
            raise KeyError(f"Column {key} does not exist")
            
        values = self._columns[key].values
        dtype = self._columns[key].dtype
        
        if np.issubdtype(dtype, np.floating):
            return np.isnan(values)
        elif np.issubdtype(dtype, np.integer):
            return values == np.iinfo(dtype).min
        elif np.issubdtype(dtype, np.bool_):
            return values == False
        elif np.issubdtype(dtype, (np.datetime64, np.timedelta64)):
            return np.isnat(values)
        else:
            return np.array([v is None for v in values])

    def insert(self, index: int, data: Dict[str, Any]) -> None:
        """在指定位置插入数据 | Insert data at specified position"""
        if index < 0:
            index += self._size
        if not 0 <= index <= self._size:
            raise IndexError("insert index out of range")
            
        # 确保所有列都存在 | Ensure all columns exist
        for key, value in data.items():
            if key not in self._columns:
                dtype = self._dtypes.get(key) or self._infer_dtype(value)
                self.ensure_column(key, dtype)
                
        # 在所有列中插入数据 | Insert data in all columns
        for key in self._columns:
            value = data.get(key, self._fill_values[key])
            self._columns[key].insert(index, value)
            
        self._size += 1