import pytest
from pyta_dev.utils.vector import ListVector

def test_init():
    """测试初始化 | Test initialization"""
    vector = ListVector()
    assert len(vector) == 0
    assert bool(vector) is False

def test_append_pop():
    """测试添加和弹出操作 | Test append and pop operations"""
    vector = ListVector()
    
    # 测试append | Test append
    vector.append(1)
    vector.append(2)
    vector.append(3)
    assert len(vector) == 3
    assert bool(vector) is True
    
    # 测试pop | Test pop
    assert vector.pop() == 3
    assert vector.pop() == 2
    assert vector.pop() == 1
    assert len(vector) == 0
    
    # 测试空vector的pop | Test pop from empty vector
    with pytest.raises(IndexError):
        vector.pop()

def test_clear():
    """测试清空操作 | Test clear operation"""
    vector = ListVector()
    vector.append(1)
    vector.append(2)
    vector.clear()
    assert len(vector) == 0
    assert bool(vector) is False

def test_extend():
    """测试批量添加 | Test extend operation"""
    vector = ListVector()
    vector.extend([1, 2, 3, 4, 5])
    assert len(vector) == 5
    assert list(vector) == [1, 2, 3, 4, 5]

def test_getitem():
    """测试索引访问 | Test index access"""
    vector = ListVector()
    vector.extend([1, 2, 3, 4, 5])
    
    # 测试正向索引 | Test positive index
    assert vector[0] == 1
    assert vector[4] == 5
    
    # 测试负向索引 | Test negative index
    assert vector[-1] == 5
    assert vector[-5] == 1
    
    # 测试越界索引 | Test out of range index
    with pytest.raises(IndexError):
        _ = vector[5]
    with pytest.raises(IndexError):
        _ = vector[-6]
    
    # 测试切片 | Test slice
    assert vector[1:4] == [2, 3, 4]
    assert vector[::2] == [1, 3, 5]

def test_iteration():
    """测试迭代 | Test iteration"""
    vector = ListVector()
    test_data = [1, 2, 3, 4, 5]
    vector.extend(test_data)
    
    # 测试for循环迭代 | Test for loop iteration
    for i, v in enumerate(vector):
        assert v == test_data[i]

def test_auto_resize():
    """测试自动扩容 | Test auto resize"""
    vector = ListVector(buffer_factor=2.0)
    
    # 添加超过初始容量的元素 | Add elements beyond initial capacity
    initial_size = 16
    test_data = list(range(initial_size * 2))
    
    for x in test_data:
        vector.append(x)
        
    assert len(vector) == len(test_data)
    assert list(vector) == test_data

def test_data():
    """测试数据拷贝 | Test data copy"""
    vector = ListVector()
    test_data = [1, 2, 3, 4, 5]
    vector.extend(test_data)
    
    # 获取数据拷贝 | Get data copy
    data = vector.data()
    assert data == test_data
    
    # 修改拷贝不影响原vector | Modifying copy doesn't affect original vector
    data[0] = 100
    assert vector[0] == 1
