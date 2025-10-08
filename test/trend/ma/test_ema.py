import numpy as np
import pytest
from pyta2.trend.ma.ema import rEMA
from pyta2.trend.ma._batch import EMA


class TestEMA:
    """测试EMA（指数移动平均）模块"""
    
    def test_ema_initialization(self):
        """测试EMA初始化"""
        ema = rEMA(5)
        assert ema.n == 5
        assert ema.window == 5
        assert ema.name == "EMA"
        assert ema.full_name == "EMA(5)"
        assert ema.alpha == 2.0 / (5 + 1)  # 2/6 = 1/3
    
    def test_ema_invalid_n(self):
        """测试无效的n参数"""
        with pytest.raises(AssertionError):
            rEMA(0)
        
        with pytest.raises(AssertionError):
            rEMA(-1)
    
    def test_ema_basic_calculation(self):
        """测试EMA基本计算"""
        ema = rEMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 前两个值应该返回NaN（窗口不足）
        result1 = ema.rolling(values[:1])
        assert np.isnan(result1)
        
        result2 = ema.rolling(values[:2])
        assert np.isnan(result2)
        
        # 第三个值开始有结果，初始值为前n个值的平均
        result3 = ema.rolling(values[:3])
        assert result3 == 2.0  # (1+2+3)/3
        
        # 后续值使用EMA公式
        result4 = ema.rolling(values[:4])
        expected4 = 2.0 + (4 - 2.0) * ema.alpha  # 2 + 2 * 0.5 = 3
        assert abs(result4 - expected4) < 1e-10
    
    def test_ema_manual_calculation(self):
        """测试EMA手动计算验证"""
        ema = rEMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 手动计算EMA值
        alpha = 2.0 / (3 + 1)  # 0.5
        
        # 第一个有效值：前3个值的平均
        ema_val = np.mean(values[:3])  # 2.0
        result3 = ema.rolling(values[:3])
        assert result3 == ema_val
        
        # 第二个值：EMA = alpha * new_value + (1-alpha) * previous_EMA
        new_ema = alpha * values[3] + (1 - alpha) * ema_val
        result4 = ema.rolling(values[:4])
        assert abs(result4 - new_ema) < 1e-10
        
        # 第三个值
        new_ema2 = alpha * values[4] + (1 - alpha) * new_ema
        result5 = ema.rolling(values[:5])
        assert abs(result5 - new_ema2) < 1e-10
    
    def test_ema_with_nan_values(self):
        """测试包含NaN值的情况"""
        ema = rEMA(3)
        values = np.array([1, np.nan, 3, 4, 5])
        
        # 包含NaN的窗口应该返回NaN
        result = ema.rolling(values[:3])
        assert np.isnan(result)
    
    def test_ema_reset(self):
        """测试EMA重置功能"""
        ema = rEMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 计算一次
        result1 = ema.rolling(values)
        
        # 重置后重新计算
        ema.reset()
        result2 = ema.rolling(values[:3])
        assert result2 == 2.0  # 重置后重新开始
    
    def test_ema_batch_function(self):
        """测试EMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = EMA(values, 3)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        assert result[2] == 2.0  # 前3个值的平均
        # 后续值使用EMA公式计算
    
    def test_ema_batch_with_numpy_array(self):
        """测试EMA批量函数使用numpy数组"""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = EMA(values, 3)
        
        # 验证结果类型
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 验证前两个值为NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 验证第三个值
        assert result[2] == 2.0
    
    def test_ema_edge_cases(self):
        """测试EMA边界情况"""
        # 测试单个值
        ema = rEMA(1)
        values = np.array([5])
        result = ema.rolling(values)
        assert result == 5.0
        
        # 测试所有相同值
        ema = rEMA(3)
        values = np.array([2, 2, 2, 2, 2])
        result = ema.rolling(values)
        assert result == 2.0  # 相同值EMA应该等于该值
    
    def test_ema_different_periods(self):
        """测试不同周期的EMA"""
        # 测试周期为1的EMA
        ema1 = rEMA(1)
        values = np.array([1, 2, 3, 4, 5])
        result = ema1.rolling(values)
        assert result == 5.0  # 周期为1时，EMA等于最新值
        
        # 测试大周期EMA
        ema10 = rEMA(10)
        values = np.arange(1, 21)  # 1到20
        
        # 前9个值应该返回NaN
        for i in range(9):
            result = ema10.rolling(values[:i+1])
            assert np.isnan(result)
        
        # 第10个值开始有结果
        result = ema10.rolling(values[:10])
        assert result == 5.5  # 前10个值的平均
    
    def test_ema_alpha_calculation(self):
        """测试EMA alpha值计算"""
        # 测试不同周期的alpha值
        ema3 = rEMA(3)
        assert ema3.alpha == 2.0 / (3 + 1)  # 0.5
        
        ema5 = rEMA(5)
        assert ema5.alpha == 2.0 / (5 + 1)  # 1/3
        
        ema20 = rEMA(20)
        assert ema20.alpha == 2.0 / (20 + 1)  # 2/21
    
    def test_ema_consistency(self):
        """测试EMA计算的一致性"""
        ema = rEMA(3)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 逐步计算
        results = []
        for i in range(3, len(values) + 1):
            result = ema.rolling(values[:i])
            results.append(result)
        
        # 验证结果递增（对于递增的输入值）
        for i in range(1, len(results)):
            assert results[i] >= results[i-1]


    def test_ema_window(self):
        """测试包含NaN值的情况"""
        ma = rEMA(4)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 包含NaN的窗口应该返回NaN
        for i in range(len(values)):
            if i < ma.window - 1:
                result = ma.rolling(values[:i+1])
                assert np.isnan(result)
            else:
                result = ma.rolling(values[:i+1])
                assert not np.isnan(result)
