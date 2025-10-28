from typing import List, Tuple, Union
from ..utils.space import Space
from collections import OrderedDict

class Schema:
    def __init__(self, 
                 schema: Union[List[Tuple[str, Space]], OrderedDict[str, Space]]
                 ):
        """ 
        Args:
            schema: 字段名和空间定义
        Example1:
            schema = [
                ('rsi', Space.Scalar(high=100, low=0, dtype=np.float64))
                ('macd', Space.Scalar(high=100, low=0, dtype=np.float64))
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
        if isinstance(schema, list):
            self.schema = OrderedDict(schema)
        elif isinstance(schema, (OrderedDict, dict)):
            self.schema = OrderedDict(schema)
        else:
            raise ValueError(f'schema must be a list[(str, Space)] or OrderedDict[str, Space], got {type(schema)}')
    
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
    
    # def get_keys(self): 
    #     return list(self.schema.keys())
    
    # def get_spaces(self, as_list: bool = False):
    #     if as_list:
    #         return list(self.schema.values())
    #     else:
    #         return self.schema
    
    def get_dtypes(self):
        return {key: space.dtype for key, space in self.schema.items()}
