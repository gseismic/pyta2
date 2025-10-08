import numpy as np
import pytest
from pyta2.trend.ma.api import get_ma_class, get_ma_function, get_ma_func, get_ma_window


class TestAPIFunctions:
    """测试移动平均API函数"""
    
    def test_get_ma_class(self):
        """测试获取移动平均类"""
        # 测试所有支持的移动平均类型
        ma_types = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA']
        
        for ma_type in ma_types:
            ma_class = get_ma_class(ma_type)
            assert ma_class is not None
            assert hasattr(ma_class, 'name')
            assert ma_class.name == ma_type
    
    def test_get_ma_class_invalid_type(self):
        """测试获取无效的移动平均类"""
        with pytest.raises(KeyError):
            get_ma_class('INVALID')
        
        with pytest.raises(KeyError):
            get_ma_class('')
    
    def test_get_ma_function(self):
        """测试获取移动平均函数"""
        # 测试所有支持的移动平均类型
        ma_types = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA']
        
        for ma_type in ma_types:
            ma_function = get_ma_function(ma_type)
            assert ma_function is not None
            assert callable(ma_function)
    
    def test_get_ma_function_invalid_type(self):
        """测试获取无效的移动平均函数"""
        with pytest.raises(KeyError):
            get_ma_function('INVALID')
        
        with pytest.raises(KeyError):
            get_ma_function('')
    
    def test_get_ma_func_alias(self):
        """测试get_ma_func别名"""
        # get_ma_func应该是get_ma_function的别名
        assert get_ma_func is get_ma_function
        
        # 测试功能相同
        sma_func1 = get_ma_function('SMA')
        sma_func2 = get_ma_func('SMA')
        assert sma_func1 is sma_func2
    
    def test_get_ma_window(self):
        """测试获取移动平均窗口大小"""
        # 测试不同移动平均类型的窗口计算
        test_cases = [
            ('SMA', 5, 5),   # SMA窗口等于周期
            ('EMA', 5, 5),   # EMA窗口等于周期
            ('WMA', 5, 5),   # WMA窗口等于周期
            ('HMA', 4, 5),   # HMA窗口 = max(2-1, 4-1) + 2 = 5
            ('DEMA', 5, 9),  # DEMA窗口 = 2*5-1 = 9
            ('TEMA', 5, 13), # TEMA窗口 = 3*5-2 = 13
        ]
        
        for ma_type, n, expected_window in test_cases:
            window = get_ma_window(ma_type, n)
            assert window == expected_window
    
    def test_get_ma_window_different_periods(self):
        """测试不同周期的窗口计算"""
        # 测试SMA不同周期
        assert get_ma_window('SMA', 1) == 1
        assert get_ma_window('SMA', 10) == 10
        assert get_ma_window('SMA', 100) == 100
        
        # 测试EMA不同周期
        assert get_ma_window('EMA', 1) == 1
        assert get_ma_window('EMA', 10) == 10
        assert get_ma_window('EMA', 100) == 100
        
        # 测试WMA不同周期
        assert get_ma_window('WMA', 1) == 1
        assert get_ma_window('WMA', 10) == 10
        assert get_ma_window('WMA', 100) == 100
        
        # 测试HMA不同周期（HMA(1)会失败，因为n//2=0）
        # with pytest.raises(AssertionError):
        #     get_ma_window('HMA', 1)  # HMA(1)会失败
        assert get_ma_window('HMA', 4) == 5  # max(1, 3) + 2 = 5
        assert get_ma_window('HMA', 16) == 20  # max(7, 15) + 4 = 20
        
        # 测试DEMA不同周期
        assert get_ma_window('DEMA', 1) == 1  # 2*1-1 = 1
        assert get_ma_window('DEMA', 5) == 9  # 2*5-1 = 9
        assert get_ma_window('DEMA', 10) == 19  # 2*10-1 = 19
        
        # 测试TEMA不同周期
        assert get_ma_window('TEMA', 1) == 1  # 3*1-2 = 1
        assert get_ma_window('TEMA', 5) == 13  # 3*5-2 = 13
        assert get_ma_window('TEMA', 10) == 28  # 3*10-2 = 28
    
    def test_get_ma_window_invalid_type(self):
        """测试获取无效类型的窗口"""
        with pytest.raises(KeyError):
            get_ma_window('INVALID', 5)
        
        with pytest.raises(KeyError):
            get_ma_window('', 5)
    
    def test_get_ma_window_invalid_period(self):
        """测试获取无效周期的窗口"""
        # 测试无效周期
        with pytest.raises(AssertionError):
            get_ma_window('SMA', 0)
        
        with pytest.raises(AssertionError):
            get_ma_window('SMA', -1)
    
    def test_api_functions_consistency(self):
        """测试API函数的一致性"""
        # 测试所有移动平均类型的一致性
        ma_types = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA']
        
        for ma_type in ma_types:
            # 获取类和函数
            ma_class = get_ma_class(ma_type)
            ma_function = get_ma_function(ma_type)
            
            # 验证类名
            assert ma_class.name == ma_type
            
            # 验证函数可调用
            assert callable(ma_function)
            
            # 验证窗口计算
            window = get_ma_window(ma_type, 5)
            assert window > 0
    
    def test_api_functions_with_different_parameters(self):
        """测试API函数使用不同参数"""
        # 测试不同周期
        periods = [1, 3, 5, 10, 20]
        ma_types = ['SMA', 'EMA', 'WMA', 'HMA', 'DEMA', 'TEMA']
        
        for ma_type in ma_types:
            for period in periods:
                # 获取类和窗口
                ma_class = get_ma_class(ma_type)
                ma_function = get_ma_function(ma_type)
                window = get_ma_window(ma_type, period)
                
                # 验证结果
                assert ma_class is not None
                assert ma_function is not None
                assert window > 0
                
                # 验证窗口大小合理
                if ma_type in ['SMA', 'EMA', 'WMA']:
                    assert window == period
                elif ma_type == 'HMA':
                    assert window >= period
                elif ma_type == 'DEMA':
                    assert window == 2 * period - 1
                elif ma_type == 'TEMA':
                    assert window == 3 * period - 2
    
    def test_api_functions_performance(self):
        """测试API函数性能"""
        import time
        
        # 测试大量调用API函数的性能
        start_time = time.time()
        
        for _ in range(1000):
            get_ma_class('SMA')
            get_ma_function('SMA')
            get_ma_window('SMA', 5)
        
        end_time = time.time()
        
        # 验证性能合理（应该在1秒内完成1000次调用）
        assert (end_time - start_time) < 1.0
    
    def test_api_functions_edge_cases(self):
        """测试API函数边界情况"""
        # 测试最小周期
        assert get_ma_window('SMA', 1) == 1
        assert get_ma_window('EMA', 1) == 1
        assert get_ma_window('WMA', 1) == 1
        assert get_ma_window('HMA', 1) == 1
        assert get_ma_window('DEMA', 1) == 1
        assert get_ma_window('TEMA', 1) == 1
        
        # 测试大周期
        large_period = 1000
        assert get_ma_window('SMA', large_period) == large_period
        assert get_ma_window('EMA', large_period) == large_period
        assert get_ma_window('WMA', large_period) == large_period
        assert get_ma_window('HMA', large_period) > large_period
        assert get_ma_window('DEMA', large_period) == 2 * large_period - 1
        assert get_ma_window('TEMA', large_period) == 3 * large_period - 2
    
    def test_api_functions_type_validation(self):
        """测试API函数类型验证"""
        # 测试字符串类型参数
        assert get_ma_class('SMA').name == 'SMA'
        assert callable(get_ma_function('SMA'))
        assert isinstance(get_ma_window('SMA', 5), int)
        
        # 测试整数类型参数
        assert get_ma_window('SMA', 5) == 5
        assert get_ma_window('DEMA', 10) == 19
        assert get_ma_window('TEMA', 20) == 58
