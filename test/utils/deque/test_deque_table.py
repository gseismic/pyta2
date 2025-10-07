import pytest
import numpy as np
from pyta2.utils.deque.deque_table import DequeTable
from pyta2.utils.deque.numpy_deque import NumpyDeque

def test_initialization():
    # 测试默认初始化
    dt = DequeTable(maxlen=None)
    assert dt.maxlen == NumpyDeque.default_maxlen
    assert len(dt) == 0
    assert dt.columns == []
    assert dt.dtypes == {}
    
    # 测试带maxlen初始化
    dt = DequeTable(maxlen=10)
    assert dt.maxlen == 10
    
    # 测试带dtypes初始化
    dt = DequeTable(maxlen=None, dtypes={'a': np.int32, 'b': np.float64})
    print(dt.dtypes)
    assert dt.dtypes == {'a': np.int32, 'b': np.float64}

def test_append_single_row():
    dt = DequeTable(maxlen=5)
    
    # 添加第一行
    dt.append({'a': 1, 'b': 2.0})
    assert len(dt) == 1
    assert dt.columns == ['a', 'b']
    assert dt.dtypes['a'] == np.int64
    assert dt.dtypes['b'] == np.float64
    
    # 检查数据
    assert dt['a'][0] == 1
    assert dt['b'][0] == 2.0
    assert dt[0] == {'a': 1, 'b': 2.0}
    
    # 添加第二行
    dt.append({'a': 3, 'b': 4.0, 'c': 'new'})
    assert len(dt) == 2
    assert dt.columns == ['a', 'b', 'c']
    assert dt.dtypes['c'] == np.dtype('O')
    
    # 检查第一行缺失列的值
    assert dt[0]['c'] is None
    assert dt[1] == {'a': 3, 'b': 4.0, 'c': 'new'}

def test_extend_rows():
    dt = DequeTable(maxlen=5)
    
    # 扩展多行数据
    dt.extend({
        'a': [1, 2, 3],
        'b': [1.1, 2.2, 3.3],
        'c': ['one', 'two', 'three']
    })
    
    assert len(dt) == 3
    assert dt.columns == ['a', 'b', 'c']
    assert dt.dtypes['a'] == np.int64
    assert dt.dtypes['b'] == np.float64
    assert dt.dtypes['c'] == np.dtype('O')
    
    # 检查数据
    assert dt[0] == {'a': 1, 'b': 1.1, 'c': 'one'}
    assert dt[1] == {'a': 2, 'b': 2.2, 'c': 'two'}
    assert dt[2] == {'a': 3, 'b': 3.3, 'c': 'three'}
    
    # 测试部分列扩展
    dt.extend({
        'a': [4, 5],
        'd': ['new1', 'new2']
    })
    
    print(dt)
    
    assert len(dt) == 5
    assert 'd' in dt.columns
    print(dt[3])
    print({'a': 4, 'b': np.nan, 'c': None, 'd': 'new1'})
    
    # 由于NaN比较的特殊性，需要特殊处理
    expected_row = {'a': 4, 'b': np.nan, 'c': None, 'd': 'new1'}
    actual_row = dt[3]
    
    # 比较非NaN字段
    assert actual_row['a'] == expected_row['a']
    assert actual_row['c'] == expected_row['c'] 
    assert actual_row['d'] == expected_row['d']
    
    # 比较NaN字段
    assert np.isnan(actual_row['b']) and np.isnan(expected_row['b'])
    
    # 同样处理第4行
    expected_row_4 = {'a': 5, 'b': np.nan, 'c': None, 'd': 'new2'}
    actual_row_4 = dt[4]
    assert actual_row_4['a'] == expected_row_4['a']
    assert actual_row_4['c'] == expected_row_4['c']
    assert actual_row_4['d'] == expected_row_4['d']
    assert np.isnan(actual_row_4['b']) and np.isnan(expected_row_4['b'])

def test_column_access():
    dt = DequeTable(maxlen=5)
    dt.append({'a': 1, 'b': 2.0})
    dt.append({'a': 3, 'b': 4.0, 'c': 'hello'})
    dt.append({'a': 5, 'b': 6.0, 'c': 'world'})
    
    # 测试列访问
    assert np.array_equal(dt['a'], np.array([1, 3, 5]))
    assert np.array_equal(dt['b'], np.array([2.0, 4.0, 6.0]))
    # 注意：第一行添加时c列不存在，所以被填充为None
    assert np.array_equal(dt['c'], np.array([None, 'hello', 'world'], dtype=object))
    
    # 测试不存在的列
    with pytest.raises(KeyError):
        dt['d']

def test_row_access():
    dt = DequeTable(maxlen=5)
    dt.append({'a': 1, 'b': 2.0})
    dt.append({'a': 3, 'b': 4.0, 'c': 'hello'})
    dt.append({'a': 5, 'b': 6.0, 'c': 'world'})
    
    # 测试行访问
    # 注意：第一行添加时c列不存在，所以被填充为None
    assert dt[0] == {'a': 1, 'b': 2.0, 'c': None}
    assert dt[1] == {'a': 3, 'b': 4.0, 'c': 'hello'}
    assert dt[2] == {'a': 5, 'b': 6.0, 'c': 'world'}
    
    # 测试负索引
    assert dt[-1] == dt[2]
    assert dt[-2] == dt[1]
    
    # 测试越界索引
    with pytest.raises(IndexError):
        dt[3]
    with pytest.raises(IndexError):
        dt[-4]

def test_iteration():
    dt = DequeTable(maxlen=5)
    dt.append({'a': 1, 'b': 2.0})
    dt.append({'a': 3, 'b': 4.0, 'c': 'hello'})
    dt.append({'a': 5, 'b': 6.0, 'c': 'world'})
    
    # 测试正向迭代
    rows = list(dt)
    assert len(rows) == 3
    assert rows[0] == dt[0]
    assert rows[1] == dt[1]
    assert rows[2] == dt[2]
    
    # 测试反向迭代
    reversed_rows = list(reversed(dt))
    assert len(reversed_rows) == 3
    assert reversed_rows[0] == dt[2]
    assert reversed_rows[1] == dt[1]
    assert reversed_rows[2] == dt[0]

def test_conversion():
    dt = DequeTable(maxlen=5)
    dt.append({'a': 1, 'b': 2.0})
    dt.append({'a': 3, 'b': 4.0, 'c': 'hello'})
    dt.append({'a': 5, 'b': 6.0, 'c': 'world'})
    
    # 测试转换为列表
    as_list = dt.to_list()
    assert len(as_list) == 3
    assert as_list[0] == dt[0]
    assert as_list[1] == dt[1]
    assert as_list[2] == dt[2]
    
    # 测试转换为字典
    as_dict = dt.to_dict()
    assert 'a' in as_dict
    assert 'b' in as_dict
    assert 'c' in as_dict
    assert np.array_equal(as_dict['a'], np.array([1, 3, 5]))
    assert np.array_equal(as_dict['b'], np.array([2.0, 4.0, 6.0]))
    # 注意：第一行添加时c列不存在，所以被填充为None
    assert np.array_equal(as_dict['c'], np.array([None, 'hello', 'world'], dtype=object))

def test_clear():
    dt = DequeTable(maxlen=5)
    
    # 填充数据
    dt.append({'a': 1})
    dt.append({'b': 2})
    assert len(dt) == 2
    assert len(dt.columns) == 2
    
    # 清空数据
    dt.clear()
    assert len(dt) == 0
    assert len(dt.columns) == 2  # 列仍然存在
    
    # 添加新数据
    dt.append({'a': 3})
    assert len(dt) == 1
    assert dt['a'][0] == 3

def test_resize():
    dt = DequeTable(maxlen=5)
    dt.append({'a': 1, 'b': 2.0})
    dt.append({'a': 3, 'b': 4.0, 'c': 'hello'})
    dt.append({'a': 5, 'b': 6.0, 'c': 'world'})
    
    # 缩小队列大小
    dt.resize(2)
    assert dt.maxlen == 2
    assert len(dt) == 2
    # 注意：resize后保留最后2行，第一行添加时c列不存在，所以被填充为None
    assert dt[0] == {'a': 3, 'b': 4.0, 'c': 'hello'}
    assert dt[1] == {'a': 5, 'b': 6.0, 'c': 'world'}
    
    # 扩大队列大小
    dt.resize(5)
    assert dt.maxlen == 5
    assert len(dt) == 2
    
    # 添加更多数据
    dt.append({'a': 7})
    dt.append({'a': 8})
    dt.append({'a': 9})
    assert len(dt) == 5
    # 注意：b列是float类型，缺失值用NaN填充
    expected_row = {'a': 9, 'b': np.nan, 'c': None}
    actual_row = dt[4]
    assert actual_row['a'] == expected_row['a']
    assert actual_row['c'] == expected_row['c']
    assert np.isnan(actual_row['b']) and np.isnan(expected_row['b'])
    
    # 测试无效大小
    with pytest.raises(ValueError):
        dt.resize(0)
    with pytest.raises(ValueError):
        dt.resize(-1)

def test_maxlen_behavior():
    dt = DequeTable(maxlen=3)
    
    # 测试队列满时丢弃旧数据
    dt.append({'a': 1})
    dt.append({'a': 2})
    dt.append({'a': 3})
    assert len(dt) == 3
    assert dt['a'].tolist() == [1, 2, 3]
    
    dt.append({'a': 4})
    assert len(dt) == 3
    assert dt['a'].tolist() == [2, 3, 4]
    
    # 测试扩展超过最大长度
    dt.extend({'a': [5, 6, 7]})
    assert len(dt) == 3
    assert dt['a'].tolist() == [5, 6, 7]
    
    # 测试扩展部分超过最大长度
    dt.extend({'a': [8, 9]})
    assert len(dt) == 3
    assert dt['a'].tolist() == [7, 8, 9]

def test_mixed_types():
    dt = DequeTable(maxlen=5)
    
    # 测试混合类型
    dt.append({
        'int': 42,
        'float': 3.14,
        'str': 'hello',
        'bool': True,
        'none': None,
        'list': [1, 2, 3],
        'dict': {'key': 'value'}
    })
    
    assert len(dt) == 1
    assert dt.dtypes['int'] == np.int64
    assert dt.dtypes['float'] == np.float64
    assert dt.dtypes['str'] == np.dtype('O')
    assert dt.dtypes['bool'] == np.bool_
    assert dt.dtypes['none'] == np.dtype('O')
    assert dt.dtypes['list'] == np.dtype('O')
    assert dt.dtypes['dict'] == np.dtype('O')
    
    # 检查值
    assert dt[0]['int'] == 42
    assert dt[0]['float'] == 3.14
    assert dt[0]['str'] == 'hello'
    # 注意：numpy布尔值比较需要特殊处理
    assert bool(dt[0]['bool']) is True
    assert dt[0]['none'] is None
    assert dt[0]['list'] == [1, 2, 3]
    assert dt[0]['dict'] == {'key': 'value'}

def test_dtypes_parameter():
    # 测试dtypes参数
    dt = DequeTable(maxlen=None, dtypes={
        'a': np.int16,
        'b': np.float32,
        'c': np.dtype('datetime64[D]')  # 指定具体的日期时间单位
    })
    
    # 添加数据
    dt.append({
        'a': 1000,  # 超出int16范围，但会被强制转换
        'b': 3.14,
        'c': np.datetime64('2023-01-01')  # 使用numpy datetime64对象
    })
    
    assert dt.dtypes['a'] == np.int16
    assert dt.dtypes['b'] == np.float32
    assert dt.dtypes['c'] == np.dtype('datetime64[D]')
    
    # 检查值
    assert dt['a'][0] == 1000  # 实际存储为int16，但取值时会转换为Python int
    assert dt['b'][0] == pytest.approx(3.14)
    assert dt['c'][0] == np.datetime64('2023-01-01')
    
    # 测试新列类型推断
    print('-'*50)
    dt.append({'d': 'new column'})
    assert dt.dtypes['d'] == np.dtype('O')
    
    print(dt)
    raise
    
if __name__ == '__main__':
    # pytest.main(__file__)
    if 1:
        test_initialization()
    if 1:
        test_append_single_row()
    if 1:
        test_extend_rows()
    if 1:
        test_column_access()
    if 1:
        test_row_access()
    if 1:
        test_iteration()
    if 1:
        test_conversion()
    if 1:
        test_clear()
    if 1:
        test_resize()
    if 1:
        test_maxlen_behavior()
    if 1:
        test_mixed_types()
    if 1:
        test_dtypes_parameter()