import numpy as np
import pytest
from pyta_dev.utils.vector import VectorTable


def test_dict_vector_basic():
    """测试基本操作 | Test basic operations"""
    dv = VectorTable()
    
    # 测试append | Test append
    dv.append({'a': 1, 'b': 1.0})
    dv.append({'a': 2, 'b': 2.0, 'c': 'text'})
    
    assert len(dv) == 2
    assert dv.columns == ['a', 'b', 'c']
    assert np.array_equal(dv['a'], [1, 2])
    assert np.array_equal(dv['b'], [1.0, 2.0])
    assert dv['c'][0] is None  # 第一行c列应该是填充值
    assert dv['c'][1] == 'text'


def test_dict_vector_dtype_inference():
    """测试数据类型推断 | Test dtype inference"""
    dv = VectorTable()
    
    # 测试不同类型的推断 | Test inference of different types
    dv.append({
        'int': 1,
        'float': 1.0,
        'bool': True,
        'str': 'text',
        'list': [1, 2, 3]
    })
    
    assert dv.dtypes['int'] == np.int64
    assert dv.dtypes['float'] == np.float64
    assert dv.dtypes['bool'] == np.bool_
    assert dv.dtypes['str'] == np.dtype('O')
    assert dv.dtypes['list'] == np.dtype('O')


def test_dict_vector_extend():
    """测试批量添加 | Test batch append"""
    dv = VectorTable()
    
    # 测试extend | Test extend
    dv.extend({
        'a': [1, 2, 3],
        'b': [1.0, 2.0, 3.0]
    })
    
    assert len(dv) == 3
    assert np.array_equal(dv['a'], [1, 2, 3])
    assert np.array_equal(dv['b'], [1.0, 2.0, 3.0])
    
    # 测试不同长度的extend | Test extend with different lengths
    with pytest.raises(ValueError):
        dv.extend({'a': [4], 'b': [4.0, 5.0]})


def test_dict_vector_fill_values():
    """测试填充值 | Test fill values"""
    dv = VectorTable()
    
    dv.append({'a': 1, 'b': 1.0})
    dv.append({'a': 2})  # b应该使用填充值
    
    assert np.isnan(dv['b'][1])  # 浮点数的填充值是nan
    
    # 测试整数的填充值
    dv.append({'c': 100})
    assert dv['a'][2] == np.iinfo(np.int64).min


def test_dict_vector_resize():
    """测试调整大小 | Test resize"""
    dv = VectorTable()
    dv.extend({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0]})
    
    # 测试扩大 | Test expand
    dv.resize(5, {'a': 0, 'b': np.nan})
    assert len(dv) == 5
    assert np.array_equal(dv['a'][:3], [1, 2, 3])
    assert np.array_equal(dv['a'][3:], [0, 0])
    
    # 测试缩小 | Test shrink
    dv.resize(2)
    assert len(dv) == 2
    assert np.array_equal(dv['a'], [1, 2])


def test_dict_vector_update():
    """测试更新操作 | Test update operations"""
    dv = VectorTable()
    dv.extend({'a': [1, 2, 3], 'b': [1.0, 2.0, 3.0]})
    
    # 测试更新部分数据 | Test partial update
    dv.update(1, {'a': [10, 20]})
    assert np.array_equal(dv['a'], [1, 10, 20])
    
    # 测试无效更新 | Test invalid update
    with pytest.raises(ValueError):
        dv.update(1, {'a': [10, 20, 30]})  # 超出范围


def test_dict_vector_insert():
    """测试插入操作 | Test insert operations"""
    dv = VectorTable()
    dv.extend({'a': [1, 2], 'b': [1.0, 2.0]})
    
    # 测试在中间插入 | Test insert in middle
    dv.insert(1, {'a': 10, 'b': 10.0})
    assert np.array_equal(dv['a'], [1, 10, 2])
    assert np.array_equal(dv['b'], [1.0, 10.0, 2.0])
    
    # 测试在末尾插入 | Test insert at end
    dv.insert(len(dv), {'a': 20, 'b': 20.0})
    assert np.array_equal(dv['a'], [1, 10, 2, 20])


def test_dict_vector_conversion():
    """测试转换方法 | Test conversion methods"""
    dv = VectorTable()
    dv.extend({'a': [1, 2], 'b': [1.0, 2.0]})
    
    # 测试to_list | Test to_list
    lst = dv.to_list()
    assert len(lst) == 2
    assert lst[0] == {'a': 1, 'b': 1.0}
    
    # 测试to_dict | Test to_dict
    d = dv.to_dict()
    assert np.array_equal(d['a'], [1, 2])
    assert np.array_equal(d['b'], [1.0, 2.0])


def test_dict_vector_fill_mask():
    """测试填充值掩码 | Test fill value mask"""
    dv = VectorTable()
    dv.append({'a': 1, 'b': 1.0})
    dv.append({'a': 2})
    
    # 测试is_fill_value | Test is_fill_value
    assert dv.is_fill_value('b', dv['b'][1])
    assert not dv.is_fill_value('b', dv['b'][0])
    
    # 测试get_fill_mask | Test get_fill_mask
    mask = dv.get_fill_mask('b')
    assert np.array_equal(mask, [False, True])


def test_dict_vector_edge_cases():
    """测试边界情况 | Test edge cases"""
    dv = VectorTable()
    
    # 测试空向量操作 | Test empty vector operations
    assert len(dv) == 0
    assert dv.columns == []
    
    # 测试无效列访问 | Test invalid column access
    with pytest.raises(KeyError):
        _ = dv['nonexistent']
    
    # 测试无效插入 | Test invalid insert
    with pytest.raises(IndexError):
        dv.insert(1, {'a': 1})
        
    # 测试负索引插入 | Test negative index insert
    dv.append({'a': 1})
    dv.insert(-1, {'a': 0})
    assert np.array_equal(dv['a'], [0, 1])


def test_dict_vector_custom_dtypes():
    """测试自定义数据类型 | Test custom dtypes"""
    dtypes = {'a': np.int32, 'b': np.float32}
    dv = VectorTable(dtypes=dtypes)
    
    dv.append({'a': 1, 'b': 1.0})
    
    assert dv.dtypes['a'] == np.int32
    assert dv.dtypes['b'] == np.float32


def test_dict_vector_setitem():
    """测试设置列数据 | Test setting column data"""
    dv = VectorTable()
    dv.extend({'a': [1, 2, 3]})
    
    # 测试设置整列 | Test setting entire column
    dv['b'] = [4.0, 5.0, 6.0]
    assert np.array_equal(dv['b'], [4.0, 5.0, 6.0])
    
    # 测试设置标量值 | Test setting scalar value
    dv['a'] = 10
    assert np.array_equal(dv['a'], [10, 10, 10])
    
    # 测试长度不匹配 | Test length mismatch
    with pytest.raises(ValueError):
        dv['a'] = [1, 2]


if __name__ == "__main__":
    pytest.main([__file__]) 