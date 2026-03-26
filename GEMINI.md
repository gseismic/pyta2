# pyta2 项目指南

## 项目概览
`pyta2` 是一个用于金融流数据的标准化、高性能技术指标库。它基于 NumPy、Polars 和 Matplotlib 构建，旨在提供极简且无冗余的指标计算实现。

### 核心技术栈
- **语言:** Python 3.8+
- **计算引擎:** NumPy (主要用于数组操作), Polars (用于高性能数据处理)
- **可视化:** Matplotlib, chineseize_matplotlib
- **基础库:** pandas, arrow

### 核心设计原则
- **性能优先:** 利用 NumPy 的矢量化操作和 Polars 的高性能特性。
- **极简设计:** 保持代码结构清晰，减少冗余，提供统一的 API 接口。
- **流式与批量支持:** 每个指标通常包含 `_rolling.py` (流式/滚动计算) 和 `_batch.py` (批量计算) 两个版本。

---

## 项目结构
- `pyta2/`: 核心代码库
    - `base/`: 包含指标基类 `rIndicator` 和模式定义 `Schema`。
    - `trend/`, `momentum/`, `stats/`, `structure/`, `volume/`, `perf/`: 按类别划分的各项指标实现。
    - `utils/`: 包含 `DequeTable`, `Space`, `Vector` 等工具类。
- `tests/`: 项目测试套件，按功能模块组织。
- `examples/`: 示例代码，展示指标的使用方法。

---

## 开发规范
### 指标实现
1. **基类继承:** 所有的流式指标应继承自 `pyta2.base.indicator.rIndicator`。
2. **实现接口:**
    - `reset_extras()`: 重置指标内部状态。
    - `forward(*args, **kwargs)`: 实现具体计算逻辑，返回当前步的计算结果。
    - `full_name` 属性: 返回指标的全名（包含参数）。
3. **文件名约定:**
    - 流式计算: `_rolling.py` 或以 `r` 开头的类名。
    - 批量计算: `_batch.py`。
4. **统一 API:** 每个子模块（如 `trend/ma/`）应通过 `api.py` 导出工厂函数或统一接口。

### 测试要求
- 新增指标必须在 `tests/` 目录下添加相应的测试用例。
- 推荐使用 `pytest` 运行测试。

---

## 常用命令
### 安装依赖
```bash
pip install -e .
```

### 运行测试
```bash
pytest tests/
```

### 运行示例
```bash
python examples/01_numpy_deque_usage.py
```
