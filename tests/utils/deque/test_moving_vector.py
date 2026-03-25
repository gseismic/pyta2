import pytest
import numpy as np
from pyta2.utils.deque.moving_vector import MovingVector

def test_moving_vector_basic():
    vec = MovingVector()
    vec.append(10)
    vec.append(11)
    vec.append(12)
    
    assert len(vec) == 3
    assert vec.notional_len == 3
    assert vec.kept_values == [10, 11, 12]
    assert vec[0] == 10
    assert vec[-1] == 12

def test_moving_vector_rekeep():
    vec = MovingVector()
    for i in range(10):
        vec.append(i)
    
    # 只保留最后 3 个
    vec.rekeep_n(3)
    assert len(vec) == 3
    assert vec.notional_len == 10
    assert vec.discard_end == 7
    assert vec.kept_values == [7, 8, 9]
    
    # 绝对索引访问
    assert vec[7] == 7
    assert vec[9] == 9
    
    # 访问已丢弃索引应报错
    with pytest.raises(IndexError, match="already discarded"):
        _ = vec[0]
        
    # 访问越界索引应报错
    with pytest.raises(IndexError, match="out of range"):
        _ = vec[10]

def test_moving_vector_rekeep_zero():
    # 测试 rekeep_n(0) 清空逻辑
    vec = MovingVector()
    vec.append(1)
    vec.append(2)
    vec.rekeep_n(0)
    
    assert len(vec) == 0
    assert vec.notional_len == 2
    assert vec.discard_end == 2
    assert vec.kept_values == []

def test_moving_vector_setitem():
    vec = MovingVector()
    vec.append(100)
    vec.append(200)
    vec.rekeep_n(1) # 只剩下 [200]，discard_end=1
    
    # 设置存量数据
    vec[1] = 201
    assert vec[1] == 201
    assert vec.kept_values == [201]
    
    # 设置负索引
    vec[-1] = 202
    assert vec[1] == 202

def test_moving_vector_numpy_conversion():
    vec = MovingVector()
    vec.extend([1, 2, 3])
    
    arr = np.array(vec)
    assert isinstance(arr, np.ndarray)
    assert np.array_equal(arr, np.array([1, 2, 3]))

def test_moving_vector_repr():
    vec = MovingVector()
    for i in range(20):
        vec.append(i)
    
    r = repr(vec)
    # 验证是否截断
    assert "..." in r
    assert "notional_len=20" in r
    assert "kept_len=20" in r
