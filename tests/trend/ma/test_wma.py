import numpy as np
import pytest
from pyta2.trend.ma.wma import rWMA
from pyta2.trend.ma._batch import WMA


class TestWMA:
    """测试WMA（加权移动平均）模块"""
    
    def test_wma_initialization(self):
        """测试WMA初始化"""
        wma = rWMA(3)
        assert wma.n == 3
        assert wma.window == 3
        assert wma.name == "WMA"
        assert wma.full_name == "WMA(3)"
        
        # 验证权重计算
        expected_weights = np.array([1/6, 2/6, 3/6])  # [1/6, 1/3, 1/2]
        np.testing.assert_array_almost_equal(wma._weights, expected_weights)
    
    def test_wma_invalid_n(self):
        """测试无效的n参数"""
        with pytest.raises(AssertionError):
            rWMA(0)
        
        with pytest.raises(AssertionError):
            rWMA(-1)
    
    def test_wma_weights_calculation(self):
        """测试WMA权重计算"""
        # 测试不同周期的权重
        wma3 = rWMA(3)
        expected_weights3 = np.array([1/6, 2/6, 3/6])
        np.testing.assert_array_almost_equal(wma3._weights, expected_weights3)
        
        wma5 = rWMA(5)
        expected_weights5 = np.array([1/15, 2/15, 3/15, 4/15, 5/15])
        np.testing.assert_array_almost_equal(wma5._weights, expected_weights5)
        
        # 验证权重和为1
        assert abs(np.sum(wma3._weights) - 1.0) < 1e-10
        assert abs(np.sum(wma5._weights) - 1.0) < 1e-10
    
    def test_wma_basic_calculation(self):
        """测试WMA基本计算"""
        wma = rWMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 前两个值应该返回NaN（窗口不足）
        result1 = wma.rolling(values[:1])
        assert np.isnan(result1)
        
        result2 = wma.rolling(values[:2])
        assert np.isnan(result2)
        
        # 第三个值开始有结果
        result3 = wma.rolling(values[:3])
        expected3 = 1 * (1/6) + 2 * (2/6) + 3 * (3/6)  # 1/6 + 4/6 + 9/6 = 14/6
        assert abs(result3 - expected3) < 1e-10
        
        result4 = wma.rolling(values[:4])
        expected4 = 2 * (1/6) + 3 * (2/6) + 4 * (3/6)  # 2/6 + 6/6 + 12/6 = 20/6
        assert abs(result4 - expected4) < 1e-10
        
        result5 = wma.rolling(values[:5])
        expected5 = 3 * (1/6) + 4 * (2/6) + 5 * (3/6)  # 3/6 + 8/6 + 15/6 = 26/6
        assert abs(result5 - expected5) < 1e-10
    
    def test_wma_manual_calculation(self):
        """测试WMA手动计算验证"""
        wma = rWMA(4)
        values = np.array([1, 2, 3, 4, 5, 6])
        
        # 手动计算WMA值
        weights = np.array([1/10, 2/10, 3/10, 4/10])  # 权重：1,2,3,4，和为10
        
        # 第一个有效值
        result4 = wma.rolling(values[:4])
        expected4 = np.dot(values[:4], weights)
        assert abs(result4 - expected4) < 1e-10
        
        # 第二个值
        result5 = wma.rolling(values[:5])
        expected5 = np.dot(values[1:5], weights)
        assert abs(result5 - expected5) < 1e-10
    
    def test_wma_with_nan_values(self):
        """测试包含NaN值的情况"""
        wma = rWMA(3)
        values = np.array([1, np.nan, 3, 4, 5])
        
        # 包含NaN的窗口应该返回NaN
        result = wma.rolling(values[:3])
        assert np.isnan(result)
    
    def test_wma_reset(self):
        """测试WMA重置功能"""
        wma = rWMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # 计算一次
        result1 = wma.rolling(values)
        
        # 重置后重新计算
        wma.reset()
        result2 = wma.rolling(values[:3])
        expected2 = 1 * (1/6) + 2 * (2/6) + 3 * (3/6)
        assert abs(result2 - expected2) < 1e-10
    
    def test_wma_batch_function(self):
        """测试WMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = WMA(values, 3)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        expected3 = 1 * (1/6) + 2 * (2/6) + 3 * (3/6)
        assert abs(result[2] - expected3) < 1e-10
    
    def test_wma_batch_with_numpy_array(self):
        """测试WMA批量函数使用numpy数组"""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = WMA(values, 3)
        
        # 验证结果类型
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 验证前两个值为NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
    
    def test_wma_edge_cases(self):
        """测试WMA边界情况"""
        # 测试单个值
        wma = rWMA(1)
        values = np.array([5])
        result = wma.rolling(values)
        assert result == 5.0  # 单个值的WMA就是该值
        
        # 测试所有相同值
        wma = rWMA(3)
        values = np.array([2, 2, 2, 2, 2])
        result = wma.rolling(values)
        assert result == 2.0  # 相同值的WMA应该等于该值
    
    def test_wma_different_periods(self):
        """测试不同周期的WMA"""
        # 测试周期为2的WMA
        wma2 = rWMA(2)
        values = np.array([1, 2, 3, 4, 5])
        
        # 第一个值应该返回NaN
        result1 = wma2.rolling(values[:1])
        assert np.isnan(result1)
        
        result2 = wma2.rolling(values[:2])
        expected2 = 1 * (1/3) + 2 * (2/3)  # 1/3 + 4/3 = 5/3
        assert abs(result2 - expected2) < 1e-10
        
        # 测试大周期WMA
        wma5 = rWMA(5)
        values = np.arange(1, 11)  # 1到10
        
        # 前4个值应该返回NaN
        for i in range(4):
            result = wma5.rolling(values[:i+1])
            assert np.isnan(result)
        
        # 第5个值开始有结果
        result = wma5.rolling(values[:5])
        weights = np.array([1/15, 2/15, 3/15, 4/15, 5/15])
        expected = np.dot(values[:5], weights)
        assert abs(result - expected) < 1e-10
    
    def test_wma_weights_properties(self):
        """测试WMA权重属性"""
        wma = rWMA(5)
        
        # 验证权重递增
        for i in range(1, len(wma._weights)):
            assert wma._weights[i] > wma._weights[i-1]
        
        # 验证权重和为1
        assert abs(np.sum(wma._weights) - 1.0) < 1e-10
        
        # 验证权重比例
        for i in range(1, len(wma._weights)):
            expected_ratio = (i + 1) / i
            actual_ratio = wma._weights[i] / wma._weights[i-1]
            assert abs(actual_ratio - expected_ratio) < 1e-10
    
    def test_wma_consistency(self):
        """测试WMA计算的一致性"""
        wma = rWMA(3)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 逐步计算
        results = []
        for i in range(3, len(values) + 1):
            result = wma.rolling(values[:i])
            results.append(result)
        
        # 验证结果递增（对于递增的输入值）
        for i in range(1, len(results)):
            assert results[i] >= results[i-1]
    
    def test_wma_vs_sma(self):
        """测试WMA与SMA的关系"""
        wma = rWMA(3)
        values = np.array([1, 2, 3, 4, 5])
        
        # WMA应该更重视最近的值
        wma_result = wma.rolling(values)
        sma_result = np.mean(values[-3:])  # 简单平均
        
        # 对于递增序列，WMA应该大于SMA
        assert wma_result > sma_result


    def test_wma_window(self):
        """测试包含NaN值的情况"""
        ma = rWMA(4)
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 包含NaN的窗口应该返回NaN
        for i in range(len(values)):
            if i < ma.window - 1:
                result = ma.rolling(values[:i+1])
                assert np.isnan(result)
            else:
                result = ma.rolling(values[:i+1])
                assert not np.isnan(result)
