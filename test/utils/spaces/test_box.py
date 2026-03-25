import pytest
import numpy as np
from pyta2.utils.space import Box, Scalar, PositiveScalar, NegativeScalar, PositiveBox, NegativeBox


class TestBox:
    """测试Box类"""

    def test_init_basic(self):
        """测试基本初始化"""
        b = Box(low=0, high=10)
        assert b.low == 0
        assert b.high == 10
        assert b.shape == ()
        assert b.dtype == np.float64

    def test_init_integer_dtype(self):
        """测试整数类型初始化"""
        b = Box(low=0, high=10, dtype=np.int32)
        assert b.dtype == np.int32
        assert b.null_value == -999999

    def test_init_custom_null_value(self):
        """测试自定义null值"""
        b = Box(low=0, high=10, null_value=-1)
        assert b.null_value == -1

    def test_init_validation_error(self):
        """测试边界验证错误"""
        with pytest.raises(ValueError, match="All low values must be <= high values"):
            Box(low=10, high=5)

    def test_sample_float(self):
        """测试浮点数采样"""
        b = Box(low=0.0, high=1.0)
        samples = [b.sample() for _ in range(100)]
        assert all(0.0 <= s <= 1.0 for s in samples)

    def test_sample_integer(self):
        """测试整数采样"""
        b = Box(low=0, high=10, dtype=np.int32)
        samples = [b.sample() for _ in range(100)]
        assert all(0 <= s <= 10 for s in samples)
        assert all(hasattr(s, 'dtype') and np.issubdtype(s.dtype, np.integer) for s in samples)

    def test_sample_with_seed(self):
        """测试种子可重复性"""
        b1 = Box(low=0, high=100, seed=42)
        b2 = Box(low=0, high=100, seed=42)
        assert b1.sample() == b2.sample()
        assert b1.sample() == b2.sample()

    def test_sample_shape(self):
        """测试采样形状"""
        b = Box(low=[0, 0], high=[10, 10])
        sample = b.sample()
        assert sample.shape == (2,)

        b_with_shape = Box(low=0, high=10, shape=(3, 4))
        sample = b_with_shape.sample()
        assert sample.shape == (3, 4)

    def test_contains_valid(self):
        """测试有效值包含检查"""
        b = Box(low=0, high=10)
        assert b.contains(5)
        assert b.contains(0)
        assert b.contains(10)

    def test_contains_invalid(self):
        """测试无效值包含检查"""
        b = Box(low=0, high=10)
        assert not b.contains(-1)
        assert not b.contains(11)

    def test_contains_wrong_shape(self):
        """测试形状不匹配"""
        b = Box(low=0, high=10, shape=(2,))
        assert not b.contains(5)

    def test_is_null(self):
        """测试null值检测"""
        b = Box(low=0, high=10, dtype=np.float64)
        assert b.is_null(np.nan)

        b_int = Box(low=0, high=10, dtype=np.int32)
        assert b_int.is_null(-999999)

    def test_set_get_null_value(self):
        """测试null值设置和获取"""
        b = Box(low=0, high=10)
        b.set_null_value(-1)
        assert b.get_null_value() == -1

    def test_eq_same(self):
        """测试相等Box"""
        b1 = Box(low=0, high=10)
        b2 = Box(low=0, high=10)
        assert b1 == b2

    def test_eq_different(self):
        """测试不相等Box"""
        b1 = Box(low=0, high=10)
        b2 = Box(low=0, high=11)
        assert b1 != b2

    def test_eq_with_null(self):
        """测试带null值的相等性"""
        b1 = Box(low=0, high=10, null_value=np.nan)
        b2 = Box(low=0, high=10, null_value=np.nan)
        assert b1 == b2

        b3 = Box(low=0, high=10, null_value=999)
        assert b1 != b3

    def test_repr(self):
        """测试字符串表示"""
        b = Box(low=0, high=10)
        r = repr(b)
        assert "Box" in r
        assert "0" in r
        assert "10" in r

    def test_to_json(self):
        """测试JSON序列化"""
        b = Box(low=0, high=10)
        json_data = b.to_json()
        assert json_data["type"] == "Box"
        assert json_data["low"] == 0
        assert json_data["high"] == 10
        assert json_data["shape"] == ()
        assert json_data["dtype"] == np.float64

    def test_from_json(self):
        """测试JSON反序列化"""
        json_data = {
            "type": "Box",
            "low": 0,
            "high": 10,
            "shape": (),
            "dtype": np.float64
        }
        b = Box.from_json(json_data)
        assert b.low == 0
        assert b.high == 10
        assert b.shape == ()


class TestPositiveBox:
    """测试PositiveBox类"""

    def test_init_default(self):
        """测试默认初始化"""
        b = PositiveBox()
        assert b.low == 0
        assert np.isinf(b.high)

    def test_init_with_high(self):
        """测试带high初始化"""
        b = PositiveBox(high=10)
        assert b.low == 0
        assert b.high == 10

    def test_init_with_shape(self):
        """测试带shape初始化"""
        b = PositiveBox(high=10, shape=(3,))
        assert b.shape == (3,)


class TestNegativeBox:
    """测试NegativeBox类"""

    def test_init_default(self):
        """测试默认初始化"""
        b = NegativeBox()
        assert np.isinf(b.low)
        assert b.low < 0
        assert b.high == 0

    def test_init_with_low(self):
        """测试带low初始化"""
        b = NegativeBox(low=-10)
        assert b.low == -10
        assert b.high == 0


class TestScalar:
    """测试Scalar类"""

    def test_init_default(self):
        """测试默认初始化"""
        s = Scalar()
        assert s.shape == ()

    def test_init_with_bounds(self):
        """测试带边界初始化"""
        s = Scalar(low=0, high=100)
        assert s.low == 0
        assert s.high == 100
        assert s.shape == ()


class TestPositiveScalar:
    """测试PositiveScalar类"""

    def test_init(self):
        """测试初始化"""
        s = PositiveScalar(high=10)
        assert s.low == 0
        assert s.high == 10
        assert s.shape == ()
        assert s.null_value is None or np.isnan(s.null_value)


class TestNegativeScalar:
    """测试NegativeScalar类"""

    def test_init(self):
        """测试初始化"""
        s = NegativeScalar(low=-10)
        assert s.low == -10
        assert s.high == 0
        assert s.shape == ()
        assert s.null_value is None or np.isnan(s.null_value)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
