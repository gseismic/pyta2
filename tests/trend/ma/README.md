# 移动平均模块测试

本目录包含pyta2库中移动平均模块的完整测试套件。

## 测试文件结构

```
test/trend/ma/
├── __init__.py                    # 测试模块初始化
├── README.md                      # 本说明文件
├── run_tests.py                   # 测试运行脚本
├── test_sma.py                    # SMA（简单移动平均）测试
├── test_ema.py                    # EMA（指数移动平均）测试
├── test_wma.py                    # WMA（加权移动平均）测试
├── test_hma.py                    # HMA（Hull移动平均）测试
├── test_dema.py                   # DEMA（双指数移动平均）测试
├── test_tema.py                   # TEMA（三指数移动平均）测试
├── test_batch_functions.py        # 批量函数测试
└── test_api_functions.py         # API函数测试
```

## 测试覆盖范围

### 1. 基础移动平均测试
- **SMA (Simple Moving Average)**: 简单移动平均
- **EMA (Exponential Moving Average)**: 指数移动平均  
- **WMA (Weighted Moving Average)**: 加权移动平均
- **HMA (Hull Moving Average)**: Hull移动平均
- **DEMA (Double Exponential Moving Average)**: 双指数移动平均
- **TEMA (Triple Exponential Moving Average)**: 三指数移动平均

### 2. 功能测试
- **批量函数测试**: 测试_batch.py中的批量计算函数
- **API函数测试**: 测试api.py中的工具函数

### 3. 测试类型
每个移动平均模块都包含以下测试类型：

#### 基础功能测试
- 初始化测试
- 参数验证测试
- 基本计算测试
- 边界情况测试

#### 数据处理测试
- NaN值处理测试
- 重置功能测试
- 不同周期测试
- 性能测试

#### 批量处理测试
- 批量函数测试
- numpy数组支持测试
- 一致性测试

## 运行测试

### 运行所有测试
```bash
python run_tests.py
```

### 运行特定测试
```bash
# 运行SMA测试
pytest test_sma.py -v

# 运行EMA测试  
pytest test_ema.py -v

# 运行所有移动平均测试
pytest test_*.py -v
```

### 运行详细测试
```bash
pytest test_sma.py -vvv
```

## 测试用例说明

### SMA测试 (test_sma.py)
- 测试简单移动平均的基本计算
- 验证窗口不足时返回NaN
- 测试批量计算功能
- 验证大窗口和小窗口情况

### EMA测试 (test_ema.py)  
- 测试指数移动平均的递归计算
- 验证alpha值计算正确性
- 测试EMA与SMA的区别
- 验证EMA的平滑特性

### WMA测试 (test_wma.py)
- 测试加权移动平均的权重计算
- 验证权重和为1
- 测试权重递增特性
- 验证WMA与SMA的关系

### HMA测试 (test_hma.py)
- 测试Hull移动平均的复杂计算
- 验证内部WMA组件
- 测试窗口计算
- 验证HMA的平滑特性

### DEMA测试 (test_dema.py)
- 测试双指数移动平均
- 验证EMA of EMA计算
- 测试窗口大小计算
- 验证DEMA公式

### TEMA测试 (test_tema.py)
- 测试三指数移动平均
- 验证三重EMA计算
- 测试复杂窗口计算
- 验证TEMA公式

### 批量函数测试 (test_batch_functions.py)
- 测试所有移动平均的批量计算
- 验证批量函数的一致性
- 测试性能
- 验证边界情况处理

### API函数测试 (test_api_functions.py)
- 测试get_ma_class函数
- 测试get_ma_function函数  
- 测试get_ma_window函数
- 验证API函数的一致性

## 测试数据

测试使用以下类型的数据：
- 简单递增序列: [1, 2, 3, 4, 5, ...]
- 随机数据: np.random.randn(1000)
- 相同值序列: [2, 2, 2, 2, ...]
- 包含NaN的序列: [1, 2, np.nan, 4, 5, ...]

## 性能要求

- 单个移动平均计算应在1秒内完成
- 批量计算1000个数据点应在1秒内完成
- 内存使用应合理，不应出现内存泄漏

## 注意事项

1. 某些移动平均（如HMA、DEMA、TEMA）需要更大的窗口才能产生有效结果
2. 窗口不足时，大多数移动平均返回NaN而不是抛出异常
3. 测试验证了数学公式的正确性，包括手动计算验证
4. 所有测试都包含边界情况和错误处理测试
