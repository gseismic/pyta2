import numpy as np
import pytest
from pyta2.trend.ma.sma import rSMA
from pyta2.trend.ma._batch import SMA


class TestSMA:
    """测试SMA（简单移动平均）模块"""
    
    def test_sma_initialization(self):
        """测试SMA初始化"""
        sma = rSMA(5)
        assert sma.n == 5
        assert sma.window == 5
        assert sma.name == "SMA"
        assert sma.full_name == "SMA(5)"
    
    def test_sma_invalid_n(self):
        """测试无效的n参数"""
        with pytest.raises(AssertionError):
            rSMA(0)
        
        with pytest.raises(AssertionError):
            rSMA(-1)
    
    def test_sma_basic_calculation(self):
        """测试SMA基本计算"""
        sma = rSMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 前两个值应该返回NaN（窗口不足）
        result1 = sma.rolling(values[:1])
        print(result1)
        assert np.isnan(result1)
        
        result2 = sma.rolling(values[:2])
        assert np.isnan(result2)
        
        # 第三个值开始有结果
        result3 = sma.rolling(values[:3])
        assert result3 == 2.0  # (1+2+3)/3
        
        result4 = sma.rolling(values[:4])
        assert result4 == 3.0  # (2+3+4)/3
        
        result5 = sma.rolling(values[:5])
        assert result5 == 4.0  # (3+4+5)/3
    
    def test_sma_with_nan_values(self):
        """测试包含NaN值的情况"""
        sma = rSMA(3)
        values = np.array([1, np.nan, 3, 4, 5])
        
        # 包含NaN的窗口应该返回NaN
        result = sma.rolling(values[:3])
        assert np.isnan(result)
    
    def test_sma_reset(self):
        """测试SMA重置功能"""
        sma = rSMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 计算一次
        result1 = sma.rolling(values)
        assert result1 == 4.0
        
        # 重置后重新计算
        sma.reset()
        result2 = sma.rolling(values[:3])
        assert result2 == 2.0
    
    def test_sma_batch_function(self):
        """测试SMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = SMA(values, 3)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[4] == 4.0  # (3+4+5)/3
        assert result[5] == 5.0  # (4+5+6)/3
        assert result[6] == 6.0  # (5+6+7)/3
        assert result[7] == 7.0  # (6+7+8)/3
        assert result[8] == 8.0  # (7+8+9)/3
        assert result[9] == 9.0  # (8+9+10)/3
    
    def test_sma_batch_with_numpy_array(self):
        """测试SMA批量函数使用numpy数组"""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = SMA(values, 3)
        
        # 验证结果类型
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 验证具体值
        assert result[2] == 2.0
        assert result[9] == 9.0
    
    def test_sma_edge_cases(self):
        """测试SMA边界情况"""
        # 测试单个值
        sma = rSMA(1)
        values = np.array([5])
        result = sma.rolling(values)
        assert result == 5.0
        
        # 测试所有相同值
        sma = rSMA(3)
        values = np.array([2, 2, 2, 2, 2])
        result = sma.rolling(values)
        assert result == 2.0
    
    def test_sma_large_window(self):
        """测试大窗口SMA"""
        sma = rSMA(10)
        values = np.arange(1, 21)  # 1到20
        
        # 前9个值应该返回NaN
        for i in range(9):
            result = sma.rolling(values[:i+1])
            assert np.isnan(result)
        
        # 第10个值开始有结果
        result = sma.rolling(values[:10])
        assert result == 5.5  # (1+2+...+10)/10
        
        result = sma.rolling(values[:11])
        assert result == 6.5  # (2+3+...+11)/10 = 65/10 = 6.5

    def test_sma_window(self):
        """测试包含NaN值的情况"""
        ma = rSMA(4)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 包含NaN的窗口应该返回NaN
        for i in range(len(values)):
            if i < ma.window - 1:
                result = ma.rolling(values[:i+1])
                assert np.isnan(result)
            else:
                result = ma.rolling(values[:i+1])
                assert not np.isnan(result)
