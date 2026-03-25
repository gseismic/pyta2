from typing import List, Tuple, Union, Mapping
from collections import OrderedDict
from pyta2.utils.space import Space

class Schema:
    def __init__(self, 
                 schema: Union[List[Tuple[str, Space]], Mapping[str, Space]]
                 ):
        """ 
        Args:
            schema: 字段名和空间定义
        Example1:
            schema = [
                ('rsi', Space.Scalar(high=100, low=0, dtype=np.float64)),
                ('macd', Space.Scalar(high=100, low=0, dtype=np.float64)),
            ]
            schema = Schema(schema)
        Example2:
            schema = {
                'rsi': Space.Scalar(high=100, low=0, dtype=np.float64),
                'macd': Space.Scalar(high=100, low=0, dtype=np.float64)
            }
            schema = Schema(schema)
        Example3:
            schema = OrderedDict([
                ('rsi', Space.Scalar(high=100, low=0, dtype=np.float64)),
                ('macd', Space.Scalar(high=100, low=0, dtype=np.float64))
            ])
            schema = Schema(schema)
        """
        if isinstance(schema, (list, dict, OrderedDict)):
            self.schema = OrderedDict(schema)
        else:
            raise TypeError(f'schema must be a list[(str, Space)], dict, or OrderedDict, got {type(schema)}')
        
        # 内容校验：确保 key 是字符串，value 是 Space 类型
        for key, space in self.schema.items():
            if not isinstance(key, str):
                raise ValueError(f"Schema key must be str, got {type(key)} for {key}")
            if not isinstance(space, Space):
                raise ValueError(f"Schema value must be a Space object, got {type(space)} for key '{key}'")
    
    def keys(self):
        return self.schema.keys()
    
    def values(self):
        return self.schema.values()
    
    def items(self):
        return self.schema.items()
    
    def __getitem__(self, key):
        return self.schema[key]
    
    def __iter__(self):
        return iter(self.schema)
    
    def __len__(self):
        return len(self.schema)

    def get_dtypes(self):
        return {key: space.dtype for key, space in self.schema.items()}

