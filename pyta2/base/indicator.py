from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple
from pyta2.utils.deque import DequeTable
from pyta2.utils.space import Space
from pyta2.base.schema import Schema
from collections import OrderedDict

class rIndicator(ABC): 
    name = None 
    
    def __init__(self,
                 window: int,
                 schema: Union[List[Tuple[str, Space]], OrderedDict[str, Space], Schema],
                 *,
                 buffer_size: Optional[int] = None,
                 extra_window: int = 0,
                 buffer_factor: int = 2,
                 return_dict: bool = False):
        """
        Args:
            window: 窗口大小
            schema: 字段名和空间定义
            buffer_size: 输出缓存大小
            extra_window: 额外窗口大小, window+extra_window为有效输出所需要的数据长度
            buffer_factor: 缓冲区因子
            return_dict: 是否返回字典
        Example:
            schema = [
                ('rsi', Space.Scalar(high=-np.inf, low=0, dtype=np.float64))
                ('macd', Space.Scalar(high=-np.inf, low=0, dtype=np.float64))
            ]
            schema = Schema(schema)
        """
        assert window > 0, f'window must be greater than 0, got {window}'
        assert isinstance(schema, (list, dict, OrderedDict, Schema)), 'schema is required'
        assert extra_window >= 0, f'extra_window must be greater than or equal to 0, got {extra_window}'
        assert buffer_size is None or buffer_size > 0, f'buffer_size must be greater than 0, got {buffer_size}'
        self.set_window(window, extra_window)
        self.schema = Schema(schema) if isinstance(schema, (list, dict, OrderedDict)) else schema
        self.buffer_factor = buffer_factor
        self.output_keys = list(self.schema.keys())
        self.return_dict = return_dict
        self.g_index = -1
        
        self._outputs = None
        self.resize_buffer(buffer_size)

        # self.father_indicator = None 
        self._meta_info = None
        self.reset() 
    
    def resize_buffer(self, buffer_size: Optional[int]):
        self.buffer_size = buffer_size
        if self._outputs is None and (buffer_size is None or buffer_size > 0):
            self._outputs = DequeTable(maxlen=self.buffer_size,
                                       dtypes=self.schema.get_dtypes(),
                                       buffer_factor=self.buffer_factor)
        elif self._outputs is not None:
            self._outputs.resize(buffer_size)
        
    def set_window(self, window: Optional[int] = None, 
                   extra_window: Optional[int] = None):
        if window is not None:
            self.window = window
        if extra_window is not None:
            self.extra_window = extra_window
        # self.required_window = self.window + self.extra_window
    
    def reset(self):
        self.g_index = -1
        if self._outputs is not None:
            self._outputs.clear()
        self._meta_info = None
        self.reset_extras()
    
    def rolling(self, *args, **kwargs):
        """
        Args:
            args: 输入数据
            kwargs: 输入数据
        Notes:
            - 0-based index，rolling第一个点后: self.g_index == 0
        """
        # 不应通过 g_index是否小于required_window 来判断是否返回无效值
        self.g_index += 1 
        # 这里需要子类自行实现点数不够时的输出(因为不同的字段需要的点数可能不一样)
        output = self.forward(*args, **kwargs) 

        dict_output = None
        if self._outputs is not None:
            # 需要缓存
            dict_output = self.make_dict_output(output) 
            self._outputs.append(dict_output) 
           
        if self.return_dict:
            if dict_output is None: 
                return self.make_dict_output(output) 
            return dict_output
        return output
    
    @abstractmethod
    def reset_extras(self):
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    @property
    @abstractmethod
    def full_name(self):
        pass
    
    def make_dict_output(self, output: Any):
        if len(self.output_keys) == 1:
            return {self.output_keys[0]: output}
        else:
            return dict(zip(self.output_keys, output))
    
    @property
    def outputs(self):
        return self._outputs
    
    @property
    def required_window(self):
        return self.window + self.extra_window
        
    @property
    def meta_info(self):
        if self._meta_info is None:
            self._meta_info = {
                'name': self.name,
                'full_name': self.full_name,
                'schema': self.schema,
                'window': self.window,
                'extra_window': self.extra_window,
                'required_window': self.required_window,
                'buffer_size': self.buffer_size,
                'buffer_factor': self.buffer_factor,
                'g_index': self.g_index,
            }
        return self._meta_info
