import numpy as np
from pyta2.utils.deque.numpy_deque import NumpyDeque

def demo_numpy_deque():
    print("=== 01: NumpyDeque Usage Example ===\n")
    
    # 1. 初始化 (必须显式传入 maxlen)
    # 固定长度模式
    q = NumpyDeque(maxlen=5, dtype=np.float64)
    print(f"Created fixed-size deque: {q}")
    
    # 2. 基础操作: append 和 popleft
    print("\n-- Base operations --")
    for i in range(7):
        q.append(float(i))
        print(f"Append {i}: {q}")
    
    val = q.popleft()
    print(f"Popleft returned: {val}")
    print(f"After popleft: {q}")
    
    # 3. 向量化扩展: extend
    print("\n-- Vectorized extend --")
    q.extend([10, 20, 30])
    print(f"After extend [10, 20, 30]: {q}")
    
    # 4. 转换与索引
    print("\n-- Conversion & Indexing --")
    print(f"As numpy array: {np.array(q)}")
    print(f"Index access q[-1]: {q[-1]}")
    print(f"Slice access q[1:3]: {q[1:3]}")
    
    # 5. 无限长度模式 (maxlen=None)
    print("\n-- Unlimited mode (maxlen=None) --")
    unlimited_q = NumpyDeque(maxlen=None)
    print(f"Initial: {unlimited_q}")
    
    # 触发自动扩容
    unlimited_q.extend(np.arange(2000))
    print(f"After adding 2000 elements: len={len(unlimited_q)}, cache_size={unlimited_q._cache_size}")
    print(f"First element: {unlimited_q[0]}, Last element: {unlimited_q[-1]}")

if __name__ == "__main__":
    demo_numpy_deque()
