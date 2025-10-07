from typing import TypeVar, Generic, Optional, Sequence

T = TypeVar('T')

class BaseVector(Generic[T]):
    """向量基类 | Base class for vector implementations"""
    _default_buffer_size = 1024 * 16
    _min_buffer_factor = 1.5
    def __init__(self, buffer_factor: float = 2.0) -> None:
        """初始化向量 | Initialize vector
        
        Args:
            buffer_factor: 缓冲区扩容因子 | Buffer expansion factor
        """
        self._size: int = 0
        self._buffer_factor = max(self._min_buffer_factor, buffer_factor)
        self._buffer_size: int = self._default_buffer_size

    def append(self, value: T) -> None:
        """添加元素到末尾 | Append element to end"""
        raise NotImplementedError

    def pop(self) -> T:
        """弹出末尾元素 | Pop element from end"""
        raise NotImplementedError

    def clear(self) -> None:
        """清空向量 | Clear vector"""
        self._size = 0 
        self._clear_buffer() 

    def _clear_buffer(self) -> None:
        """清理内部缓冲区 | Clear internal buffer"""
        raise NotImplementedError

    def _check_resize(self) -> None:
        """检查是否需要扩容 | Check if resize is needed"""
        if self._size >= self._buffer_size:
            new_size = int(self._buffer_size * self._buffer_factor)
            self._resize_buffer(new_size)
            self._buffer_size = new_size

    def _resize_buffer(self, new_size: int) -> None:
        """调整缓冲区大小 | Resize buffer"""
        raise NotImplementedError

    def extend(self, values: Sequence[T]) -> None:
        """批量添加元素 | Extend with multiple elements"""
        for value in values:
            self.append(value)

    def __len__(self) -> int:
        """返回向量长度 | Return vector length"""
        return self._size

    def __bool__(self) -> bool:
        """返回向量是否为空 | Return if vector is empty"""
        return self._size > 0
