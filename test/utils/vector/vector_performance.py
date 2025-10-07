import time
import numpy as np
from typing import List, Any, Callable, Dict, Tuple
from pyta2.utils.vector import ListVector, NumPyVector

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

def calculate_speedup(baseline_time: float, times: List[float]) -> List[float]:
    """计算加速比 | Calculate speedup"""
    return [baseline_time / t if t > 0 else 0.0 for t in times]

timer = Timer()

@timer
def test_list_append(n: int) -> List[float]:
    """测试Python列表append性能 | Test Python list append performance"""
    lst: List[float] = []
    for i in range(n):
        lst.append(float(i))
    return lst

@timer
def test_list_vector_append(n: int) -> ListVector[float]:
    """测试ListVector append性能 | Test ListVector append performance"""
    vector = ListVector[float]()
    for i in range(n):
        vector.append(float(i))
    return vector

@timer
def test_numpy_vector_append(n: int) -> NumPyVector:
    """测试NumPyVector append性能 | Test NumPyVector append performance"""
    vector = NumPyVector(dtype=np.float64)
    for i in range(n):
        vector.append(float(i))
    return vector

@timer
def test_list_extend(n: int) -> List[float]:
    """测试Python列表extend性能 | Test Python list extend performance"""
    lst: List[float] = []
    chunk_size = 10000  # 增大chunk大小
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        lst.extend(float(j) for j in range(i, end))
    return lst

@timer
def test_list_vector_extend(n: int) -> ListVector[float]:
    """测试ListVector extend性能 | Test ListVector extend performance"""
    vector = ListVector[float]()
    chunk_size = 10000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        vector.extend(float(j) for j in range(i, end))
    return vector

@timer
def test_numpy_vector_extend(n: int) -> NumPyVector:
    """测试NumPyVector extend性能 | Test NumPyVector extend performance"""
    vector = NumPyVector(dtype=np.float64)
    chunk_size = 10000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        values = np.arange(i, end, dtype=np.float64)
        vector.extend(values)
    return vector

@timer
def test_numpy_array_extend(n: int) -> np.ndarray:
    """测试原生NumPy extend性能 | Test native NumPy extend performance"""
    arr = np.empty(n, dtype=np.float64)
    chunk_size = 10000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        arr[i:end] = np.arange(i, end, dtype=np.float64)
    return arr

@timer
def test_list_iteration(lst: List[float]) -> float:
    """测试Python列表迭代性能 | Test Python list iteration performance"""
    total = 0.0
    for x in lst:
        total += x
    return total

@timer
def test_list_vector_iteration(vector: ListVector[float]) -> float:
    """测试ListVector迭代性能 | Test ListVector iteration performance"""
    total = 0.0
    for x in vector:
        total += x
    return total

@timer
def test_numpy_vector_iteration(vector: NumPyVector) -> float:
    """测试NumPyVector迭代性能 | Test NumPyVector iteration performance"""
    total = 0.0
    for x in vector:
        total += x
    return total

@timer
def test_numpy_vector_sum(vector: NumPyVector) -> float:
    """测试NumPyVector原生求和性能 | Test NumPyVector native sum performance"""
    return float(np.sum(vector.values))

@timer
def test_list_indexing(lst: List[float], indices: List[int]) -> List[float]:
    """测试Python列表索引访问性能 | Test Python list indexing performance"""
    return [lst[i] for i in indices]

@timer
def test_list_vector_indexing(vector: ListVector[float], indices: List[int]) -> List[float]:
    """测试ListVector索引访问性能 | Test ListVector indexing performance"""
    return [vector[i] for i in indices]

@timer
def test_numpy_vector_indexing(vector: NumPyVector, indices: List[int]) -> np.ndarray:
    """测试NumPyVector索引访问性能 | Test NumPyVector indexing performance"""
    return vector.values[indices]

@timer
def test_numpy_array_append(n: int) -> np.ndarray:
    """测试原生NumPy append性能 | Test native NumPy append performance"""
    arr = np.empty(n, dtype=np.float64)
    for i in range(n):
        arr[i] = float(i)
    return arr

@timer
def test_numpy_array_extend(n: int) -> np.ndarray:
    """测试原生NumPy extend性能 | Test native NumPy extend performance"""
    arr = np.empty(n, dtype=np.float64)
    chunk_size = 10000
    for i in range(0, n, chunk_size):
        end = min(i + chunk_size, n)
        arr[i:end] = np.arange(i, end, dtype=np.float64)
    return arr

def print_speedup_comparison(operation: str, baseline: float, times: List[float], names: List[str]) -> None:
    """打印性能对比信息 | Print performance comparison information"""
    speedups = calculate_speedup(baseline, times)
    print(f"\n=== {operation} 性能对比 | Performance Comparison ===")
    print(f"基准时间 (Base time) [{names[0]}]: {baseline:.4f} seconds")
    for name, time, speedup in zip(names[1:], times[1:], speedups[1:]):
        print(f"{name}:")
        print(f"  时间 (Time): {time:.4f} seconds")
        print(f"  加速比 (Speedup): {speedup:.2f}x")

def main() -> None:
    """主测试函数 | Main test function"""
    # 测试参数 | Test parameters
    n = 50_000_000  # 增加到5千万
    n_indices = 5_000_000  # 增加到500万
    
    print("\n=== 预热 | Warm up ===")
    # 预热JIT编译器 | Warm up JIT compiler
    _ = test_numpy_vector_append(1000)
    _ = test_numpy_vector_extend(1000)
    
    print("\n=== 测试append性能 | Testing append performance ===")
    lst = test_list_append(n)
    list_vector = test_list_vector_append(n)
    numpy_vector = test_numpy_vector_append(n)
    numpy_array = test_numpy_array_append(n)
    
    print_speedup_comparison(
        "Append",
        timer.times["test_list_append"],
        [timer.times["test_list_append"], 
         timer.times["test_list_vector_append"],
         timer.times["test_numpy_vector_append"],
         timer.times["test_numpy_array_append"]],
        ["List", "ListVector", "NumPyVector", "NumPy Array"]
    )
    
    print("\n=== 测试extend性能 | Testing extend performance ===")
    lst_extend = test_list_extend(n)
    list_vector_extend = test_list_vector_extend(n)
    numpy_vector_extend = test_numpy_vector_extend(n)
    numpy_array_extend = test_numpy_array_extend(n)
    
    print_speedup_comparison(
        "Extend",
        timer.times["test_list_extend"],
        [timer.times["test_list_extend"], 
         timer.times["test_list_vector_extend"],
         timer.times["test_numpy_vector_extend"],
         timer.times["test_numpy_array_extend"]],
        ["List", "ListVector", "NumPyVector", "NumPy Array"]
    )
    
    print("\n=== 测试迭代性能 | Testing iteration performance ===")
    list_sum = test_list_iteration(lst)
    list_vector_sum = test_list_vector_iteration(list_vector)
    numpy_vector_sum = test_numpy_vector_iteration(numpy_vector)
    numpy_native_sum = test_numpy_vector_sum(numpy_vector)
    
    print_speedup_comparison(
        "Iteration",
        timer.times["test_list_iteration"],
        [timer.times["test_list_iteration"], 
         timer.times["test_list_vector_iteration"],
         timer.times["test_numpy_vector_iteration"],
         timer.times["test_numpy_vector_sum"]],
        ["List", "ListVector", "NumPyVector", "NumPyVector(native)"]
    )
    
    print("\n=== 测试索引访问性能 | Testing indexing performance ===")
    indices = np.random.randint(0, n, n_indices).tolist()
    list_indexed = test_list_indexing(lst, indices)
    list_vector_indexed = test_list_vector_indexing(list_vector, indices)
    numpy_vector_indexed = test_numpy_vector_indexing(numpy_vector, indices)
    
    print_speedup_comparison(
        "Indexing",
        timer.times["test_list_indexing"],
        [timer.times["test_list_indexing"], 
         timer.times["test_list_vector_indexing"],
         timer.times["test_numpy_vector_indexing"]],
        ["List", "ListVector", "NumPyVector"]
    )

if __name__ == "__main__":
    main()