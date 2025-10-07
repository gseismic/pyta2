from typing import Optional, Union, Any, Sequence, Iterator, Dict
import numpy as np
from numpy.typing import NDArray, DTypeLike
from .base import BaseVector

class NumPyVector(BaseVector[Any]):
    """基于NumPy的向量实现 | NumPy-based vector implementation"""
    
    def __init__(
        self, 
        dtype: DTypeLike = np.float64,
        buffer_factor: float = 2.0
    ) -> None:
        """初始化NumPy向量 | Initialize NumPy vector"""
        super().__init__(buffer_factor=buffer_factor)
        self._dtype = dtype
        self._data: NDArray = np.empty(self._buffer_size, dtype=self._dtype)
        self._valid_data: Optional[NDArray] = None  # 缓存有效数据视图

    def _get_valid_data(self) -> NDArray:
        """获取有效数据视图 | Get valid data view"""
        if self._valid_data is None or self._valid_data.size != self._size:
            self._valid_data = self._data[:self._size]
        return self._valid_data

    def append(self, value: Any) -> None:
        """添加元素到末尾 | Append element to end"""
        self._check_resize()
        self._data[self._size] = value
        self._size += 1
        self._valid_data = None  # 使缓存失效

    def extend(self, values: Sequence[Any]) -> None:
        """批量添加元素 | Extend with multiple elements"""
        if isinstance(values, np.ndarray) and values.dtype == self._dtype:
            values_array = values
        else:
            values_array = np.array(values, dtype=self._dtype)
            
        needed_size = self._size + len(values_array)
        if needed_size > self._buffer_size:
            new_size = max(needed_size, int(self._buffer_size * self._buffer_factor))
            self._resize_buffer(new_size)
            self._buffer_size = new_size
            
        self._data[self._size:self._size + len(values_array)] = values_array
        self._size = needed_size
        self._valid_data = None  # 使缓存失效

    def _resize_buffer(self, new_size: int) -> None:
        """调整缓冲区大小 | Resize buffer"""
        new_data = np.empty(new_size, dtype=self._dtype)
        if self._size > 0:
            new_data[:self._size] = self._data[:self._size]
        self._data = new_data
        self._valid_data = None  # 使缓存失效

    def pop(self) -> Any:
        """弹出末尾元素 | Pop element from end"""
        if self._size == 0:
            raise IndexError("pop from empty vector")
        self._size -= 1
        return self._data[self._size]

    def _clear_buffer(self) -> None:
        """清理内部缓冲区 | Clear internal buffer"""
        self._data.fill(0)

    def __getitem__(self, key: Union[int, slice, NDArray]) -> Union[Any, NDArray]:
        """获取元素 | Get element
        
        完全匹配numpy的行为 | Fully match numpy behavior
        
        Args:
            key: 整数、切片或ndarray | Integer, slice or ndarray
            
        Returns:
            单个值或ndarray | Single value or ndarray
        """
        return self._get_valid_data()[key]

    def __iter__(self) -> Iterator[Any]:
        """迭代器 | Iterator"""
        return iter(self._data[:self._size])

    @property
    def values(self) -> NDArray:
        """当前向量中的所有值（视图）| Current values in the vector (view)"""
        return self._get_valid_data()

    def __array__(self, dtype: Optional[DTypeLike] = None) -> NDArray:
        """NumPy数组接口 | NumPy array interface"""
        arr = self.values
        return arr.astype(dtype) if dtype is not None else arr

    @property
    def __array_interface__(self) -> Dict[str, Any]:
        """NumPy数组接口字典 | NumPy array interface dictionary"""
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
        """返回数据类型 | Return data type"""
        return self._dtype

    @property
    def shape(self) -> tuple[int, ...]:
        """返回形状 | Return shape"""
        return (self._size,)

    def astype(self, dtype: DTypeLike) -> 'NumPyVector':
        """转换数据类型 | Convert data type"""
        new_vector = NumPyVector(dtype=dtype, buffer_factor=self._buffer_factor)
        if self._size > 0:
            new_vector._data = np.empty(self._buffer_size, dtype=dtype)
            new_vector._data[:self._size] = self.values.astype(dtype)
            new_vector._size = self._size
            new_vector._buffer_size = self._buffer_size
        return new_vector

    def copy(self) -> 'NumPyVector':
        """创建深拷贝 | Create deep copy"""
        new_vector = NumPyVector(dtype=self._dtype, buffer_factor=self._buffer_factor)
        if self._size > 0:
            new_vector._data = np.empty(self._buffer_size, dtype=self._dtype)
            new_vector._data[:self._size] = self.values.copy()
            new_vector._size = self._size
            new_vector._buffer_size = self._buffer_size
        return new_vector

    def __setitem__(self, key: Union[int, slice, NDArray], value: Any) -> None:
        """设置元素值 | Set element value
        
        完全匹配numpy的行为 | Fully match numpy behavior
        
        Args:
            key: 整数、切片或ndarray | Integer, slice or ndarray
            value: 要设置的值 | Value to set
        """
        self._get_valid_data()[key] = value

    def fill(self, value: Any) -> None:
        """填充所有元素为指定值 | Fill all elements with specified value"""
        self._data[:self._size] = value

    def update(self, start: int, values: Sequence[Any]) -> None:
        """更新指定位置开始的一段数据 | Update a range of data starting from specified position
        
        Args:
            start: 起始位置 | Start position
            values: 新的值序列 | New value sequence
        """
        if start < 0:
            start += self._size
        if not 0 <= start <= self._size:
            raise IndexError("start index out of range")
            
        end = start + len(values)
        if end > self._size:
            raise ValueError("update would extend beyond current size")
            
        self._data[start:end] = values

    def resize(self, new_size: int, fill_value: Any = None) -> None:
        """调整向量大小 | Resize vector
        
        Args:
            new_size: 新的大小 | New size
            fill_value: 填充值（用于扩展时） | Fill value (for expansion)
        """
        if new_size < 0:
            raise ValueError("negative size")
            
        if new_size > self._buffer_size:
            # 需要扩容 | Need to expand
            new_buffer_size = max(new_size, int(self._buffer_size * self._buffer_factor))
            self._resize_buffer(new_buffer_size)
            self._buffer_size = new_buffer_size
            
        if new_size > self._size and fill_value is not None:
            # 填充新增的元素 | Fill new elements
            self._data[self._size:new_size] = fill_value
            
        self._size = new_size

    def insert(self, index: int, value: Any) -> None:
        """在指定位置插入值 | Insert value at specified position
        
        Args:
            index: 插入位置 | Insert position
            value: 要插入的值 | Value to insert
        """
        if index < 0:
            index += self._size
        if not 0 <= index <= self._size:
            raise IndexError("insert index out of range")
            
        self._check_resize()
        
        # 移动数据 | Move data
        if index < self._size:
            self._data[index+1:self._size+1] = self._data[index:self._size]
        
        self._data[index] = value
        self._size += 1

    # 添加比较运算符支持
    def __lt__(self, other: Any) -> NDArray:
        """小于运算符 | Less than operator"""
        return self._get_valid_data() < other

    def __le__(self, other: Any) -> NDArray:
        """小于等于运算符 | Less than or equal operator"""
        return self._get_valid_data() <= other

    def __gt__(self, other: Any) -> NDArray:
        """大于运算符 | Greater than operator"""
        return self._get_valid_data() > other

    def __ge__(self, other: Any) -> NDArray:
        """大于等于运算符 | Greater than or equal operator"""
        return self._get_valid_data() >= other

    def __eq__(self, other: Any) -> NDArray:  # type: ignore
        """等于运算符 | Equal operator"""
        return self._get_valid_data() == other

    def __ne__(self, other: Any) -> NDArray:  # type: ignore
        """不等于运算符 | Not equal operator"""
        return self._get_valid_data() != other

# 别名 | Alias
NumpyVector = NumPyVector