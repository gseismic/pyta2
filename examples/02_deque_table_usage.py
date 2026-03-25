import numpy as np
from pyta2.utils.deque.deque_table import DequeTable

def demo_deque_table():
    print("=== 02: DequeTable Usage Example ===\n")
    
    # 1. 初始化
    # DequeTable 是基于列式存储的字典队列
    dt = DequeTable(maxlen=5)
    print(f"Created fixed-size DequeTable: {dt}")
    
    # 2. 基础操作: append 一行字典
    print("\n-- Base operations --")
    dt.append({'a': 1, 'b': 100})
    dt.append({'a': 2, 'b': 200, 'c': 'Hello', 'd': 1000}) # 动态新增列
    print(f"Added columns {dt.columns}: {dt}")

    print(f'to_polars: \n{dt.to_polars()}')
    print(f'to_pandas: \n{dt.to_pandas()}')
    
    # 3. 列式访问与行式访问
    print("\n-- Column & Row Access --")
    print(f"Column 'a' values: {dt['a']}")
    print(f"Column 'c' values (padded with None): {dt['c']}")
    print(f"Row access dt[0]: {dt[0]}")
    print(f"Row access dt[-1]: {dt[-1]}")
    
    # 4. 向量化扩展: extend
    print("\n-- Batch extending --")
    dt.extend({
        'a': [10, 20, 30],
        'd': [True, False, True] # 运行时动态新增 d 列
    })
    print(f"After extension: {dt}")
    
    # 5. 遍历
    print("\n-- Row Iteration --")
    for i, row in enumerate(dt):
        print(f"Row {i}: {row}")
        if i >= 1: break # 只打印前两行
        
    # 6. 无限长度模式 (maxlen=None)
    print("\n-- Unlimited mode (maxlen=None) --")
    unlimited_dt = DequeTable(maxlen=None)
    unlimited_dt.extend({'idx': list(range(2000)), 'val': [v*1.0 for v in range(2000)]})
    print(f"Unlimited total rows: {len(unlimited_dt)}")
    print(f"First row: {unlimited_dt[0]}, Last row: {unlimited_dt[-1]}")
    
    # 7. 转换与操作
    print("\n-- Formatting & Analysis --")
    print(f"As column dictionary: {unlimited_dt.to_dict().keys()}")
    
    # 转换为主流数据框架进行分析
    try:
        df_pd = unlimited_dt.to_pandas()
        print(f"Convert to Pandas (head):\n{df_pd.head()}")
        
        df_pl = unlimited_dt.to_polars()
        print(f"Convert to Polars (head):\n{df_pl.head()}")
    except ImportError:
        print("Pandas or Polars not installed, skipping dataframe conversion demo.")
    
    # 验证清理
    unlimited_dt.clear()
    print(f"After clear: len={len(unlimited_dt)}")

if __name__ == "__main__":
    demo_deque_table()
