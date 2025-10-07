import pytest
import numpy as np
from pyta2.utils.deque.numpy_deque import NumpyDeque, NumPyDeque


class TestNumpyDeque:
    """测试NumpyDeque基础功能"""
    
    def test_init_basic(self):
        """测试基本初始化"""
        q = NumpyDeque(5)
        assert q.maxlen == 5
        assert len(q) == 0
        assert q.dtype == np.float64
        
    def test_init_with_dtype(self):
        """测试不同数据类型的初始化"""
        q_int = NumpyDeque(5, dtype=np.int32)
        assert q_int.dtype == np.int32
        
        q_float = NumpyDeque(5, dtype=np.float32)
        assert q_float.dtype == np.float32
        
    def test_init_with_buffer_factor(self):
        """测试自定义缓冲区因子"""
        q = NumpyDeque(5, buffer_factor=2.0)
        assert q.maxlen == 5
        
    def test_push_and_len(self):
        """测试push操作和长度计算"""
        q = NumpyDeque(3)
        
        # 空队列
        assert len(q) == 0
        
        # 添加元素
        q.push(1)
        assert len(q) == 1
        assert q[0] == 1
        
        q.push(2)
        assert len(q) == 2
        assert q[0] == 1
        assert q[1] == 2
        
        q.push(3)
        assert len(q) == 3
        assert q[0] == 1
        assert q[1] == 2
        assert q[2] == 3
        
    def test_maxlen_overflow(self):
        """测试超过最大长度时的行为"""
        q = NumpyDeque(3)
        
        # 添加4个元素，应该只保留最后3个
        q.push(1)
        q.push(2)
        q.push(3)
        q.push(4)
        
        assert len(q) == 3
        assert q[0] == 2  # 第一个元素被移除
        assert q[1] == 3
        assert q[2] == 4
        
    def test_pop(self):
        """测试pop操作"""
        q = NumpyDeque(5)
        
        # 空队列pop应该抛出异常
        with pytest.raises(IndexError, match="pop from empty deque"):
            q.pop()
            
        # 添加元素后pop
        q.push(1)
        q.push(2)
        q.push(3)
        
        assert q.pop() == 1  # 从左侧pop
        assert len(q) == 2
        assert q[0] == 2
        
        assert q.pop() == 2
        assert len(q) == 1
        assert q[0] == 3
        
        assert q.pop() == 3
        assert len(q) == 0
        
    def test_clear(self):
        """测试清空队列"""
        q = NumpyDeque(5)
        q.push(1)
        q.push(2)
        q.push(3)
        
        assert len(q) == 3
        q.clear()
        assert len(q) == 0
        
        # 清空后应该能正常添加
        q.push(4)
        assert len(q) == 1
        assert q[0] == 4
        
    def test_indexing(self):
        """测试索引访问"""
        q = NumpyDeque(5)
        q.push(10)
        q.push(20)
        q.push(30)
        
        # 正向索引
        assert q[0] == 10
        assert q[1] == 20
        assert q[2] == 30
        
        # 负向索引
        assert q[-1] == 30
        assert q[-2] == 20
        assert q[-3] == 10
        
        # 索引越界
        with pytest.raises(IndexError):
            q[3]
        with pytest.raises(IndexError):
            q[-4]
            
    def test_slicing(self):
        """测试切片操作"""
        q = NumpyDeque(10)
        for i in range(5):
            q.push(i)
            
        # 基本切片
        assert np.array_equal(q[1:3], np.array([1, 2]))
        assert np.array_equal(q[:3], np.array([0, 1, 2]))
        assert np.array_equal(q[2:], np.array([2, 3, 4]))
        
        # 负索引切片
        assert np.array_equal(q[-3:], np.array([2, 3, 4]))
        assert np.array_equal(q[:-2], np.array([0, 1, 2]))
        
        # 步长切片
        assert np.array_equal(q[::2], np.array([0, 2, 4]))
        
    def test_setitem(self):
        """测试设置元素值"""
        q = NumpyDeque(5)
        q.push(1)
        q.push(2)
        q.push(3)
        
        # 设置单个元素
        q[1] = 99
        assert q[1] == 99
        assert q[0] == 1
        assert q[2] == 3
        
        # 设置切片
        q[0:2] = [100, 200]
        assert q[0] == 100
        assert q[1] == 200
        
    def test_extend(self):
        """测试批量添加"""
        q = NumpyDeque(5)
        q.extend([1, 2, 3])
        
        assert len(q) == 3
        assert q[0] == 1
        assert q[1] == 2
        assert q[2] == 3
        
        # 测试超过最大长度
        q.extend([4, 5, 6])
        assert len(q) == 5
        assert q[0] == 2  # 第一个元素被移除
        assert q[4] == 6
        
    def test_values_property(self):
        """测试values属性"""
        q = NumpyDeque(5)
        q.push(1)
        q.push(2)
        q.push(3)
        
        values = q.values
        assert isinstance(values, np.ndarray)
        assert np.array_equal(values, np.array([1, 2, 3]))
        
        # 修改values应该影响队列
        values[1] = 99
        assert q[1] == 99
        
    def test_numpy_compatibility(self):
        """测试NumPy兼容性"""
        q = NumpyDeque(5)
        q.push(1.5)
        q.push(2.5)
        q.push(3.5)
        
        # 转换为NumPy数组
        arr = np.array(q)
        assert isinstance(arr, np.ndarray)
        assert np.array_equal(arr, np.array([1.5, 2.5, 3.5]))
        
        # 测试数组接口
        interface = q.__array_interface__
        assert 'shape' in interface
        assert 'typestr' in interface
        assert 'data' in interface
        
    def test_comparison_operators(self):
        """测试比较运算符"""
        q1 = NumpyDeque(5)
        q1.push(1)
        q1.push(2)
        q1.push(3)
        
        q2 = NumpyDeque(5)
        q2.push(1)
        q2.push(2)
        q2.push(3)
        
        # 相等比较 - 使用np.array_equal处理数组比较
        assert np.array_equal(q1.values, q2.values)
        assert not np.array_equal(q1.values, q2.values) == False
        
        # 修改后不相等
        q1[1] = 99
        assert not np.array_equal(q1.values, q2.values)
        
        # 与标量比较
        assert np.array_equal(q1 > 0, np.array([True, True, True]))
        assert np.array_equal(q1 < 100, np.array([True, True, True]))
        
    def test_different_dtypes(self):
        """测试不同数据类型"""
        # 整数类型
        q_int = NumpyDeque(5, dtype=np.int32)
        q_int.push(1)
        q_int.push(2)
        assert q_int.dtype == np.int32
        assert q_int[0] == 1
        
        # 浮点类型
        q_float = NumpyDeque(5, dtype=np.float32)
        q_float.push(1.5)
        q_float.push(2.5)
        assert q_float.dtype == np.float32
        assert q_float[0] == 1.5
        
        # 布尔类型
        q_bool = NumpyDeque(5, dtype=np.bool_)
        q_bool.push(True)
        q_bool.push(False)
        assert q_bool.dtype == np.bool_
        assert q_bool[0] == True
        assert q_bool[1] == False
        
    def test_none_handling(self):
        """测试None值处理"""
        q = NumpyDeque(5, dtype=np.float64)
        q.push(None)  # 应该转换为np.nan
        
        assert np.isnan(q[0])
        
        # 整数类型处理None
        q_int = NumpyDeque(5, dtype=np.int32)
        q_int.push(None)  # 应该转换为0
        assert q_int[0] == 0
        
    def test_resize(self):
        """测试调整队列大小"""
        q = NumpyDeque(5)
        for i in range(3):
            q.push(i)
            
        # 缩小队列
        q.resize(2)
        assert q.maxlen == 2
        assert len(q) == 2
        assert q[0] == 1  # 保留最后2个元素
        assert q[1] == 2
        
        # 扩大队列
        q.resize(10)
        assert q.maxlen == 10
        assert len(q) == 2
        
    def test_repr(self):
        """测试字符串表示"""
        q = NumpyDeque(5)
        q.push(1)
        q.push(2)
        q.push(3)
        
        repr_str = repr(q)
        assert "NumPyDeque" in repr_str
        assert "[1.0, 2.0, 3.0]" in repr_str  # 浮点数表示
        
    def test_buffer_management(self):
        """测试缓冲区管理"""
        # 测试小缓冲区
        q = NumpyDeque(3, buffer_factor=1.5)
        for i in range(10):  # 超过缓冲区大小
            q.push(i)
            
        # 应该只保留最后3个元素
        assert len(q) == 3
        assert q[0] == 7
        assert q[1] == 8
        assert q[2] == 9
        
    def test_large_dataset(self):
        """测试大数据集"""
        q = NumpyDeque(1000)
        
        # 添加大量数据
        for i in range(2000):
            q.push(i)
            
        # 应该只保留最后1000个元素
        assert len(q) == 1000
        assert q[0] == 1000
        assert q[999] == 1999
        
    def test_performance_basic(self):
        """基础性能测试"""
        import time
        
        q = NumpyDeque(10000)
        
        # 测试push性能
        start = time.perf_counter()
        for i in range(10000):
            q.push(i)
        push_time = time.perf_counter() - start
        
        # 测试索引访问性能
        start = time.perf_counter()
        total = 0
        for i in range(10000):
            total += q[i]
        access_time = time.perf_counter() - start
        
        assert push_time < 1.0  # push操作应该在1秒内完成
        assert access_time < 1.0  # 索引访问应该在1秒内完成
        assert total == sum(range(10000))  # 验证数据正确性


class TestNumPyDequeAlias:
    """测试NumPyDeque别名"""
    
    def test_alias_works(self):
        """测试别名是否正常工作"""
        q1 = NumpyDeque(5)
        q2 = NumPyDeque(5)
        
        # 应该是同一个类
        assert q1.__class__ == q2.__class__
        
        # 功能应该相同
        q1.push(1)
        q2.push(1)
        assert q1[0] == q2[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
