import pytest
import numpy as np
from pyta2.utils.space import Discrete


class TestDiscrete:
    """测试Discrete类"""

    def test_init(self):
        """测试基本初始化"""
        d = Discrete(5)
        assert d.n == 5
        assert d.dtype == np.int64

    def test_init_invalid_n_zero(self):
        """测试n=0时抛出错误"""
        with pytest.raises(AssertionError, match="n must be positive"):
            Discrete(0)

    def test_init_invalid_n_negative(self):
        """测试n<0时抛出错误"""
        with pytest.raises(AssertionError, match="n must be positive"):
            Discrete(-1)

    def test_sample(self):
        """测试基本采样"""
        d = Discrete(5)
        samples = [d.sample() for _ in range(100)]
        assert all(0 <= s < 5 for s in samples)
        assert all(isinstance(s, (int, np.integer)) for s in samples)

    def test_sample_with_seed(self):
        """测试种子可重复性"""
        d1 = Discrete(10, seed=42)
        d2 = Discrete(10, seed=42)
        assert d1.sample() == d2.sample()
        assert d1.sample() == d2.sample()

    def test_sample_with_mask(self):
        """测试带掩码采样 - 1表示允许,0表示排除"""
        d = Discrete(5)
        mask = np.array([1, 0, 1, 0, 1], dtype=np.int8)
        samples = set()
        for _ in range(100):
            s = d.sample(mask=mask)
            samples.add(s)
        assert samples == {0, 2, 4}
        assert 1 not in samples
        assert 3 not in samples

    def test_sample_mask_wrong_shape_error(self):
        """测试掩码形状错误"""
        d = Discrete(5)
        mask = np.array([1, 1, 1], dtype=np.int8)
        with pytest.raises(AssertionError):
            d.sample(mask=mask)

    def test_sample_mask_all_invalid_error(self):
        """测试全掩码时抛出错误"""
        d = Discrete(5)
        mask = np.array([0, 0, 0, 0, 0], dtype=np.int8)
        with pytest.raises(ValueError, match="At least one valid choice must be provided"):
            d.sample(mask=mask)

    def test_contains_valid_integers(self):
        """测试有效整数包含"""
        d = Discrete(5)
        assert d.contains(0)
        assert d.contains(1)
        assert d.contains(2)
        assert d.contains(3)
        assert d.contains(4)

    def test_contains_boundary(self):
        """测试边界值"""
        d = Discrete(5)
        assert not d.contains(-1)
        assert not d.contains(5)

    def test_contains_float(self):
        """测试浮点数被拒绝"""
        d = Discrete(5)
        assert not d.contains(0.0)
        assert not d.contains(2.5)
        assert not d.contains(4.9)

    def test_contains_numpy_integer(self):
        """测试numpy整数类型"""
        d = Discrete(5)
        assert d.contains(np.int64(2))
        assert d.contains(np.int32(3))
        assert not d.contains(np.int64(5))

    def test_eq_same(self):
        """测试相等"""
        d1 = Discrete(5)
        d2 = Discrete(5)
        assert d1 == d2

    def test_eq_different(self):
        """测试不等"""
        d1 = Discrete(5)
        d2 = Discrete(10)
        assert d1 != d2

    def test_repr(self):
        """测试字符串表示"""
        d = Discrete(5)
        r = repr(d)
        assert "Discrete" in r
        assert "5" in r

    def test_to_json(self):
        """测试JSON序列化"""
        d = Discrete(5)
        json_data = d.to_json()
        assert json_data["type"] == "Discrete"
        assert json_data["n"] == 5

    def test_from_json(self):
        """测试JSON反序列化"""
        json_data = {
            "type": "Discrete",
            "n": 5
        }
        d = Discrete.from_json(json_data)
        assert d.n == 5

    def test_round_trip(self):
        """测试序列化往返"""
        d1 = Discrete(10)
        json_data = d1.to_json()
        d2 = Discrete.from_json(json_data)
        assert d1 == d2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
