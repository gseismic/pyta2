from typing import Dict, Any, Optional, List, Union, Sequence
import numpy as np
from numpy.typing import NDArray, DTypeLike
from .deque.deque_table import DequeTable
from .deque.numpy_deque import NumpyVector
from .deque.moving_vector import MovingVector

class VectorTable(DequeTable):
    """基于DequeTable的表格向量，固定为无限长度
    Table-structured vectors based on DequeTable, fixed as infinite length
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
        super().__init__(maxlen=None, buffer_factor=buffer_factor, dtypes=dtypes)

__all__ = ['VectorTable', 'NumpyVector', 'MovingVector']
