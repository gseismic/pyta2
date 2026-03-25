import numpy as np
import pytest
from pyta2.trend.ma._batch import SMA, EMA, WMA, HMA, DEMA, TEMA


class TestBatchFunctions:
    """测试批量移动平均函数"""
    
    def test_sma_batch_function(self):
        """测试SMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = SMA(values, 3)
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        assert result[2] == 2.0  # (1+2+3)/3
        assert result[3] == 3.0  # (2+3+4)/3
        assert result[4] == 4.0  # (3+4+5)/3
        assert result[9] == 9.0  # (8+9+10)/3
    
    def test_ema_batch_function(self):
        """测试EMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = EMA(values, 3)
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        assert result[2] == 2.0  # 前3个值的平均
        assert not np.isnan(result[3])
        assert not np.isnan(result[9])
    
    def test_wma_batch_function(self):
        """测试WMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = WMA(values, 3)
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前两个值应该是NaN
        assert np.isnan(result[0])
        assert np.isnan(result[1])
        
        # 从第三个值开始有结果
        expected3 = 1 * (1/6) + 2 * (2/6) + 3 * (3/6)  # 14/6
        assert abs(result[2] - expected3) < 1e-10
        assert not np.isnan(result[9])
    
    def test_hma_batch_function(self):
        """测试HMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        result = HMA(values, 4)
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前几个值应该是NaN（窗口不足）
        for i in range(5):  # HMA(4)需要更多窗口
            print(i, result[i])
            assert np.isnan(result[i])
        
        # 窗口足够后开始有结果
        assert not np.isnan(result[5])
    
    def test_dema_batch_function(self):
        """测试DEMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
        result = DEMA(values, 3) 
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前几个值应该是NaN（窗口不足）
        for i in range(5):  # DEMA(3)需要2*3-1=5个窗口
            assert np.isnan(result[i])
        
        # 窗口足够后开始有结果
        assert not np.isnan(result[5])
    
    def test_tema_batch_function(self):
        """测试TEMA批量函数"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        result = TEMA(values, 3)
        
        # 验证结果类型和长度
        assert isinstance(result, np.ndarray)
        assert len(result) == len(values)
        
        # 前几个值应该是NaN（窗口不足）
        for i in range(7):  # TEMA(3)需要3*3-2=7个窗口
            assert np.isnan(result[i])
        
        # 窗口足够后开始有结果
        assert not np.isnan(result[7])
    
    def test_batch_functions_with_numpy_array(self):
        """测试批量函数使用numpy数组"""
        values = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        
        # 测试所有批量函数
        sma_result = SMA(values, 3)
        ema_result = EMA(values, 3)
        wma_result = WMA(values, 3)
        
        # 验证结果类型
        assert isinstance(sma_result, np.ndarray)
        assert isinstance(ema_result, np.ndarray)
        assert isinstance(wma_result, np.ndarray)
        
        # 验证长度
        assert len(sma_result) == len(values)
        assert len(ema_result) == len(values)
        assert len(wma_result) == len(values)
    
    def test_batch_functions_with_different_periods(self):
        """测试批量函数使用不同周期"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 测试不同周期的SMA
        sma1 = SMA(values, 1)
        sma3 = SMA(values, 3)
        sma5 = SMA(values, 5)
        
        # 周期为1时，第一个值就有结果
        assert not np.isnan(sma1[0])
        assert sma1[0] == 1.0
        
        # 周期为3时，前两个值为NaN
        assert np.isnan(sma3[0])
        assert np.isnan(sma3[1])
        assert not np.isnan(sma3[2])
        
        # 周期为5时，前四个值为NaN
        for i in range(4):
            assert np.isnan(sma5[i])
        assert not np.isnan(sma5[4])
    
    def test_batch_functions_with_nan_values(self):
        """测试批量函数处理NaN值"""
        values = [1, 2, np.nan, 4, 5, 6, 7, 8, 9, 10]
        
        # 测试SMA处理NaN
        sma_result = SMA(values, 3)
        assert np.isnan(sma_result[2])  # 包含NaN的窗口应该返回NaN
        
        # 测试EMA处理NaN
        ema_result = EMA(values, 3)
        assert np.isnan(ema_result[2])  # 包含NaN的窗口应该返回NaN
    
    def test_batch_functions_consistency(self):
        """测试批量函数的一致性"""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        
        # 测试相同输入产生相同结果
        result1 = SMA(values, 3)
        result2 = SMA(values, 3)
        np.testing.assert_array_equal(result1, result2)
        
        # 测试不同周期产生不同结果
        sma3 = SMA(values, 3)
        sma5 = SMA(values, 5)
        
        # 结果应该不同
        assert not np.array_equal(sma3, sma5)
    
    def test_batch_functions_performance(self):
        """测试批量函数性能"""
        values = np.random.randn(1000)  # 生成1000个随机数
        
        # 测试计算时间
        import time
        
        start_time = time.time()
        sma_result = SMA(values, 10)
        end_time = time.time()
        
        # 验证结果
        assert isinstance(sma_result, np.ndarray)
        assert len(sma_result) == len(values)
        
        # 验证计算时间合理
        assert (end_time - start_time) < 1.0  # 应该在1秒内完成
    
    def test_batch_functions_edge_cases(self):
        """测试批量函数边界情况"""
        # 测试空数组（会抛出异常）
        with pytest.raises(ValueError):
            SMA([], 3)
        
        # 测试单个值
        single_result = SMA([5], 1)
        assert len(single_result) == 1
        assert not np.isnan(single_result[0])
        assert single_result[0] == 5.0
        
        # 测试所有相同值
        same_values = [2, 2, 2, 2, 2]
        sma_result = SMA(same_values, 3)
        assert not np.isnan(sma_result[2])
        assert sma_result[2] == 2.0
    
    def test_batch_functions_parameter_validation(self):
        """测试批量函数参数验证"""
        values = [1, 2, 3, 4, 5]
        
        # 测试无效周期
        with pytest.raises(AssertionError):
            SMA(values, 0)
        
        with pytest.raises(AssertionError):
            SMA(values, -1)
        
        # 测试有效周期
        result = SMA(values, 1)
        assert isinstance(result, np.ndarray)
