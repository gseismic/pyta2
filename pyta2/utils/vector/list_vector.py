from typing import TypeVar, List, Iterator, Union, Optional, Sequence
from .base import BaseVector

T = TypeVar('T')

class ListVector(BaseVector[T]):
    """基于列表的向量实现 | List-based vector implementation"""
    
    def __init__(self, buffer_factor: float = 2.0) -> None:
        """初始化列表向量 | Initialize list vector"""
        super().__init__(buffer_factor=buffer_factor)
        self._values: List[T] = []  # 改用动态列表 | Use dynamic list
        self._values_extend([None] * self._buffer_size)  # type: ignore

    def append(self, value: T) -> None:
        """添加元素到末尾 | Append element to end"""
        if self._size >= self._buffer_size:
            new_size = int(self._buffer_size * self._buffer_factor)
            self._resize_buffer(new_size)
            self._buffer_size = new_size
        self._values[self._size] = value
        self._size += 1

    def _values_extend(self, values: Sequence[T]) -> None:
        """优化的列表扩展 | Optimized list extension"""
        self._values.extend(values)

    def _resize_buffer(self, new_size: int) -> None:
        """优化的缓冲区调整 | Optimized buffer resize"""
        extension = [None] * (new_size - len(self._values))  # type: ignore
        self._values_extend(extension)

    def pop(self) -> T:
        """弹出末尾元素 | Pop element from end"""
        if self._size == 0:
            raise IndexError("pop from empty vector")
        self._size -= 1
        return self._values[self._size]  # type: ignore

    def _clear_buffer(self) -> None:
        """清理内部缓冲区 | Clear internal buffer"""
        self._values = [None] * self._buffer_size

    def __getitem__(self, key: Union[int, slice]) -> Union[T, Sequence[T]]:
        """获取元素 | Get element"""
        if isinstance(key, slice):
            return self.data()[key]
        
        if key < 0:
            key += self._size
        if not 0 <= key < self._size:
            raise IndexError("vector index out of range")
        return self._values[key]  # type: ignore

    def __iter__(self) -> Iterator[T]:
        """迭代器 | Iterator"""
        for i in range(self._size):
            yield self._values[i]  # type: ignore

    def data(self) -> List[T]:
        """返回数据的拷贝 | Return a copy of data"""
        return self._values[:self._size]  # type: ignore
