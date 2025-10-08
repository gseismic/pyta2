import numpy as np
from .base import Space


class Category(Space):
    """
    用于表示分类变量的空间，支持任意类型的离散值
    """
    def __init__(self, cats, seed=None):
        """
        :param categories: 可选项列表，可以是任意类型
        :param seed: 随机种子
        """
        # 确保类别唯一性
        self.categories = tuple(cats)
        if len(self.categories) != len(set(self.categories)):
            raise ValueError("Categories must be unique")
        
        self.n = len(self.categories)
        if self.n == 0:
            raise ValueError("At least one category is required")
        
        # 推断数据类型
        dtype = type(self.categories[0])
        for item in self.categories:
            if type(item) != dtype:
                dtype = object  # 混合类型则使用object
                break
        
        # 初始化基类
        super().__init__(shape=(), dtype=dtype, seed=seed)
        
        # 创建索引映射
        self._index_map = {item: idx for idx, item in enumerate(self.categories)}
        self._reverse_map = {idx: item for idx, item in enumerate(self.categories)}
    
    def sample(self, mask=None):
        """
        从类别中随机采样
        :param mask: 可选掩码，形状为(n,)的np.int8数组，0表示允许选择
        """
        if mask is not None:
            mask = np.asarray(mask, dtype=np.int8)
            if mask.shape != (self.n,):
                raise ValueError(f"Mask shape {mask.shape} != {(self.n,)}")
            
            valid_indices = np.where(mask == 0)[0]
            if len(valid_indices) == 0:
                raise ValueError("At least one choice must be unmasked")
            
            idx = self.np_random.choice(valid_indices)
        else:
            idx = self.np_random.randint(0, self.n)
        
        return self._reverse_map[idx]
    
    def contains(self, x):
        """检查值是否在类别中"""
        return x in self._index_map
    
    def to_index(self, value):
        """将值转换为整数索引"""
        return self._index_map[value]
    
    def from_index(self, idx):
        """将整数索引转换为原始值"""
        return self._reverse_map[idx]
    
    def __repr__(self):
        return f"Category(cats={self.cats}, dtype={self.dtype})"
    
    def __eq__(self, other):
        return isinstance(other, Category) and self.categories == other.categories
    
    def to_json(self):
        """序列化为JSON格式"""
        return {
            "type": "Cat",
            "categories": list(self.categories),
            "dtype": self.dtype,
        }
    
    @classmethod
    def from_json(cls, data):
        """从JSON反序列化"""
        return cls(categories=data["categories"], dtype=data["dtype"])
    
class Bool(Category):
    def __init__(self):
        super().__init__([True, False])
    
    def __repr__(self):
        return f"Bool()"

class Sign(Category):
    def __init__(self):
        super().__init__([-1, 0, 1])
    
    def __repr__(self):
        return f"Sign()"

__all__ = ['Category', 'Bool', 'Sign']