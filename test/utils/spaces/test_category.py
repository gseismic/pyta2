import pytest
import numpy as np
from pyta2.utils.space import Category, Bool, Sign


class TestCategory:
    """测试Category类"""

    def test_init(self):
        """测试基本初始化"""
        c = Category(['a', 'b', 'c'])
        assert c.categories == ('a', 'b', 'c')
        assert c.n == 3
        assert c.dtype == str

    def test_init_string_type(self):
        """测试字符串类型推断"""
        c = Category(['apple', 'banana', 'cherry'])
        assert c.dtype == str

    def test_init_int_type(self):
        """测试整数类型推断"""
        c = Category([1, 2, 3])
        assert c.dtype == int

    def test_init_mixed_types(self):
        """测试混合类型推断为object"""
        c = Category([1, 'two', 3.0])
        assert c.dtype == object

    def test_init_duplicate_error(self):
        """测试重复类别抛出错误"""
        with pytest.raises(ValueError, match="Categories must be unique"):
            Category(['a', 'b', 'a'])

    def test_init_empty_error(self):
        """测试空类别抛出错误"""
        with pytest.raises(ValueError, match="At least one category is required"):
            Category([])

    def test_sample(self):
        """测试基本采样"""
        c = Category(['a', 'b', 'c'])
        samples = [c.sample() for _ in range(100)]
        assert all(s in ('a', 'b', 'c') for s in samples)

    def test_sample_with_seed(self):
        """测试种子可重复性"""
        c1 = Category(['a', 'b', 'c'], seed=42)
        c2 = Category(['a', 'b', 'c'], seed=42)
        assert c1.sample() == c2.sample()
        assert c1.sample() == c2.sample()

    def test_sample_with_mask(self):
        """测试带掩码采样 - 1表示允许,0表示排除"""
        c = Category(['a', 'b', 'c'])
        mask = np.array([1, 0, 1], dtype=np.int8)
        samples = set()
        for _ in range(100):
            s = c.sample(mask=mask)
            samples.add(s)
        assert samples == {'a', 'c'}
        assert 'b' not in samples

    def test_sample_mask_all_invalid_error(self):
        """测试全掩码时抛出错误"""
        c = Category(['a', 'b', 'c'])
        mask = np.array([0, 0, 0], dtype=np.int8)
        with pytest.raises(ValueError, match="At least one valid choice must be provided"):
            c.sample(mask=mask)

    def test_sample_mask_wrong_shape_error(self):
        """测试掩码形状错误"""
        c = Category(['a', 'b', 'c'])
        mask = np.array([1, 1], dtype=np.int8)
        with pytest.raises(ValueError):
            c.sample(mask=mask)

    def test_contains(self):
        """测试包含检查"""
        c = Category(['a', 'b', 'c'])
        assert c.contains('a')
        assert c.contains('b')
        assert c.contains('c')
        assert not c.contains('d')

    def test_to_index(self):
        """测试值转索引"""
        c = Category(['a', 'b', 'c'])
        assert c.to_index('a') == 0
        assert c.to_index('b') == 1
        assert c.to_index('c') == 2

    def test_from_index(self):
        """测试索引转值"""
        c = Category(['a', 'b', 'c'])
        assert c.from_index(0) == 'a'
        assert c.from_index(1) == 'b'
        assert c.from_index(2) == 'c'

    def test_eq_same(self):
        """测试相等"""
        c1 = Category(['a', 'b', 'c'])
        c2 = Category(['a', 'b', 'c'])
        assert c1 == c2

    def test_eq_different(self):
        """测试不等"""
        c1 = Category(['a', 'b', 'c'])
        c2 = Category(['a', 'b', 'd'])
        assert c1 != c2

    def test_repr(self):
        """测试字符串表示"""
        c = Category(['a', 'b', 'c'])
        r = repr(c)
        assert "Category" in r
        assert "a" in r
        assert "b" in r
        assert "c" in r

    def test_to_json(self):
        """测试JSON序列化"""
        c = Category(['a', 'b', 'c'])
        json_data = c.to_json()
        assert json_data["type"] == "Cat"
        assert json_data["categories"] == ['a', 'b', 'c']

    def test_from_json(self):
        """测试JSON反序列化"""
        json_data = {
            "type": "Cat",
            "categories": ['a', 'b', 'c'],
            "dtype": object
        }
        c = Category.from_json(json_data)
        assert c.categories == ('a', 'b', 'c')


class TestBool:
    """测试Bool类"""

    def test_init(self):
        """测试初始化"""
        b = Bool()
        assert b.categories == (True, False)
        assert b.n == 2

    def test_contains(self):
        """测试包含检查"""
        b = Bool()
        assert b.contains(True)
        assert b.contains(False)

    def test_repr(self):
        """测试字符串表示"""
        b = Bool()
        assert repr(b) == "Bool()"

    def test_sample(self):
        """测试采样"""
        b = Bool()
        samples = [b.sample() for _ in range(100)]
        assert all(s in (True, False) for s in samples)


class TestSign:
    """测试Sign类"""

    def test_init(self):
        """测试初始化"""
        s = Sign()
        assert s.categories == (-1, 0, 1)
        assert s.n == 3

    def test_contains(self):
        """测试包含检查"""
        s = Sign()
        assert s.contains(-1)
        assert s.contains(0)
        assert s.contains(1)
        assert not s.contains(2)
        assert not s.contains(-2)

    def test_repr(self):
        """测试字符串表示"""
        s = Sign()
        assert repr(s) == "Sign()"

    def test_sample(self):
        """测试采样"""
        s = Sign()
        samples = [s.sample() for _ in range(100)]
        assert all(samp in (-1, 0, 1) for samp in samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
