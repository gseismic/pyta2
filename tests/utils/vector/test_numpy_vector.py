import pytest
import numpy as np
from pyta_dev.utils.vector import NumPyVector

def test_init():
    """测试初始化 | Test initialization"""
    vector = NumPyVector(dtype=np.float64)
    assert len(vector) == 0
    assert bool(vector) is False
    assert vector.dtype == np.float64

def test_append_pop():
    """测试添加和弹出操作 | Test append and pop operations"""
    vector = NumPyVector(dtype=np.float64)
    
    # 测试append | Test append
    vector.append(1.0)
    vector.append(2.0)
    vector.append(3.0)
    assert len(vector) == 3
    assert bool(vector) is True
    
    # 测试pop | Test pop
    assert vector.pop() == 3.0
    assert vector.pop() == 2.0
    assert vector.pop() == 1.0
    assert len(vector) == 0
    
    # 测试空vector的pop | Test pop from empty vector
    with pytest.raises(IndexError):
        vector.pop()

def test_clear():
    """测试清空操作 | Test clear operation"""
    vector = NumPyVector(dtype=np.float64)
    vector.append(1.0)
    vector.append(2.0)
    vector.clear()
    assert len(vector) == 0
    assert bool(vector) is False

def test_extend():
    """测试批量添加 | Test extend operation"""
    vector = NumPyVector(dtype=np.float64)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector.extend(data)
    assert len(vector) == 5
    np.testing.assert_array_equal(vector.values, np.array(data))

def test_getitem():
    """测试索引访问 | Test index access"""
    vector = NumPyVector(dtype=np.float64)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector.extend(data)
    
    # 测试正向索引 | Test positive index
    assert vector[0] == 1.0
    assert vector[4] == 5.0
    
    # 测试负向索引 | Test negative index
    assert vector[-1] == 5.0
    assert vector[-5] == 1.0
    
    # 测试越界索引 | Test out of range index
    with pytest.raises(IndexError):
        _ = vector[5]
    with pytest.raises(IndexError):
        _ = vector[-6]
    
    # 测试切片 | Test slice
    np.testing.assert_array_equal(vector[1:4], np.array([2.0, 3.0, 4.0]))
    np.testing.assert_array_equal(vector[::2], np.array([1.0, 3.0, 5.0]))

def test_iteration():
    """测试迭代 | Test iteration"""
    vector = NumPyVector(dtype=np.float64)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector.extend(data)
    
    # 测试for循环迭代 | Test for loop iteration
    for i, v in enumerate(vector):
        assert v == data[i]

def test_auto_resize():
    """测试自动扩容 | Test auto resize"""
    vector = NumPyVector(dtype=np.float64, buffer_factor=2.0)
    
    # 添加超过初始容量的元素 | Add elements beyond initial capacity
    initial_size = 16
    test_data = np.arange(initial_size * 2, dtype=np.float64)
    
    for x in test_data:
        vector.append(x)
        
    assert len(vector) == len(test_data)
    np.testing.assert_array_equal(vector.values, test_data)

def test_numpy_interface():
    """测试NumPy接口 | Test NumPy interface"""
    vector = NumPyVector(dtype=np.float64)
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    vector.extend(data)
    
    # 测试转换为numpy数组 | Test conversion to numpy array
    arr = np.array(vector)
    np.testing.assert_array_equal(arr, np.array(data))
    
    # 测试类型转换 | Test type conversion
    int_arr = np.array(vector, dtype=np.int64)
    np.testing.assert_array_equal(int_arr, np.array(data, dtype=np.int64))

def test_astype():
    """测试类型转换 | Test type conversion"""
    vector = NumPyVector(dtype=np.float64)
    vector.extend([1.5, 2.5, 3.5])
    
    # 转换为整数类型 | Convert to integer type
    int_vector = vector.astype(np.int64)
    np.testing.assert_array_equal(int_vector.values, np.array([1, 2, 3], dtype=np.int64))
    
    # 原vector不变 | Original vector unchanged
    np.testing.assert_array_equal(vector.values, np.array([1.5, 2.5, 3.5]))

def test_copy():
    """测试深拷贝 | Test deep copy"""
    vector = NumPyVector(dtype=np.float64)
    vector.extend([1.0, 2.0, 3.0])
    
    # 创建深拷贝 | Create deep copy
    vector_copy = vector.copy()
    
    # 修改拷贝不影响原vector | Modifying copy doesn't affect original vector
    vector_copy._data[0] = 100.0
    assert vector[0] == 1.0
    assert vector_copy[0] == 100.0

def test_shape_property():
    """测试shape属性 | Test shape property"""
    vector = NumPyVector(dtype=np.float64)
    assert vector.shape == (0,)
    
    vector.extend([1.0, 2.0, 3.0])
    assert vector.shape == (3,)
