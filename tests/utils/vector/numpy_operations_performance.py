import time
import numpy as np
from typing import Any, Callable, Dict, List
from pyta2.utils.vector import NumPyVector

class Timer:
    """计时器类 | Timer class"""
    def __init__(self) -> None:
        self.times: Dict[str, float] = {}
        
    def __call__(self, func: Callable) -> Callable:
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            self.times[func.__name__] = end - start
            print(f"{func.__name__}: {end - start:.4f} seconds")
            return result
        return wrapper

timer = Timer()

# 基础运算测试 | Basic operation tests
@timer
def test_list_sum(data: List[float]) -> float:
    """测试Python列表sum性能 | Test Python list sum performance"""
    return sum(data)

@timer
def test_numpy_array_sum(data: np.ndarray) -> float:
    """测试numpy数组sum性能 | Test numpy array sum performance"""
    return float(np.sum(data))

@timer
def test_numpy_vector_sum(vector: NumPyVector) -> float:
    """测试NumPyVector sum性能 | Test NumPyVector sum performance"""
    return float(np.sum(vector))  # 直接使用NumPy接口

# 统计运算测试 | Statistical operation tests
@timer
def test_list_mean(data: List[float]) -> float:
    """测试Python列表mean性能 | Test Python list mean performance"""
    return sum(data) / len(data)

@timer
def test_numpy_array_mean(data: np.ndarray) -> float:
    """测试numpy数组mean性能 | Test numpy array mean performance"""
    return float(np.mean(data))

@timer
def test_numpy_vector_mean(vector: NumPyVector) -> float:
    """测试NumPyVector mean性能 | Test NumPyVector mean performance"""
    return float(np.mean(vector))  # 直接使用NumPy接口

# 数学运算测试 | Mathematical operation tests
@timer
def test_list_sqrt(data: List[float]) -> List[float]:
    """测试Python列表sqrt性能 | Test Python list sqrt performance"""
    return [np.sqrt(x) for x in data]

@timer
def test_numpy_array_sqrt(data: np.ndarray) -> np.ndarray:
    """测试numpy数组sqrt性能 | Test numpy array sqrt performance"""
    return np.sqrt(data)

@timer
def test_numpy_vector_sqrt(vector: NumPyVector) -> np.ndarray:
    """测试NumPyVector sqrt性能 | Test NumPyVector sqrt performance"""
    return np.sqrt(vector)  # 直接使用NumPy接口

# 复杂运算测试 | Complex operation tests
@timer
def test_list_complex_ops(data: List[float]) -> List[float]:
    """测试Python列表复杂运算性能 | Test Python list complex operations performance"""
    return [(np.sin(x) + np.cos(x)) * np.sqrt(x) for x in data if x > 0]

@timer
def test_numpy_array_complex_ops(data: np.ndarray) -> np.ndarray:
    """测试numpy数组复杂运算性能 | Test numpy array complex operations performance"""
    mask = data > 0
    return (np.sin(data[mask]) + np.cos(data[mask])) * np.sqrt(data[mask])

@timer
def test_numpy_vector_complex_ops(vector: NumPyVector) -> np.ndarray:
    """测试NumPyVector复杂运算性能 | Test NumPyVector complex operations performance"""
    mask = vector > 0  # 直接使用NumPy接口
    return (np.sin(vector[mask]) + np.cos(vector[mask])) * np.sqrt(vector[mask])

def print_speedup_comparison(operation: str, baseline: float, times: List[float], names: List[str]) -> None:
    """打印性能对比信息 | Print performance comparison information"""
    speedups = [baseline / t if t > 0 else 0.0 for t in times]
    print(f"\n=== {operation} 性能对比 | Performance Comparison ===")
    print(f"基准时间 (Base time) [{names[0]}]: {baseline:.4f} seconds")
    for name, time, speedup in zip(names[1:], times[1:], speedups[1:]):
        print(f"{name}:")
        print(f"  时间 (Time): {time:.4f} seconds")
        print(f"  加速比 (Speedup): {speedup:.2f}x")

def main() -> None:
    """主测试函数 | Main test function"""
    # 测试参数 | Test parameters
    n = 10_000_000  # 1千万数据点
    
    # 准备数据 | Prepare data
    data_list = [float(i) for i in range(n)]
    data_array = np.arange(n, dtype=np.float64)
    vector = NumPyVector(dtype=np.float64)
    vector.extend(data_array)
    
    print("\n=== 测试sum性能 | Testing sum performance ===")
    list_sum = test_list_sum(data_list)
    numpy_array_sum = test_numpy_array_sum(data_array)
    numpy_vector_sum = test_numpy_vector_sum(vector)
    
    print_speedup_comparison(
        "Sum",
        timer.times["test_list_sum"],
        [timer.times["test_list_sum"],
         timer.times["test_numpy_array_sum"],
         timer.times["test_numpy_vector_sum"]],
        ["List", "NumPy Array", "NumPyVector"]
    )
    
    print("\n=== 测试mean性能 | Testing mean performance ===")
    list_mean = test_list_mean(data_list)
    numpy_array_mean = test_numpy_array_mean(data_array)
    numpy_vector_mean = test_numpy_vector_mean(vector)
    
    print_speedup_comparison(
        "Mean",
        timer.times["test_list_mean"],
        [timer.times["test_list_mean"],
         timer.times["test_numpy_array_mean"],
         timer.times["test_numpy_vector_mean"]],
        ["List", "NumPy Array", "NumPyVector"]
    )
    
    print("\n=== 测试sqrt性能 | Testing sqrt performance ===")
    list_sqrt = test_list_sqrt(data_list)
    numpy_array_sqrt = test_numpy_array_sqrt(data_array)
    numpy_vector_sqrt = test_numpy_vector_sqrt(vector)
    
    print_speedup_comparison(
        "Sqrt",
        timer.times["test_list_sqrt"],
        [timer.times["test_list_sqrt"],
         timer.times["test_numpy_array_sqrt"],
         timer.times["test_numpy_vector_sqrt"]],
        ["List", "NumPy Array", "NumPyVector"]
    )
    
    print("\n=== 测试复杂运算性能 | Testing complex operations performance ===")
    list_complex = test_list_complex_ops(data_list)
    numpy_array_complex = test_numpy_array_complex_ops(data_array)
    numpy_vector_complex = test_numpy_vector_complex_ops(vector)
    
    print_speedup_comparison(
        "Complex Operations",
        timer.times["test_list_complex_ops"],
        [timer.times["test_list_complex_ops"],
         timer.times["test_numpy_array_complex_ops"],
         timer.times["test_numpy_vector_complex_ops"]],
        ["List", "NumPy Array", "NumPyVector"]
    )

if __name__ == "__main__":
    main() 