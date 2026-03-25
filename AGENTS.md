# AGENTS.md - pyta2 Codebase Guide

pyta2 is a technical indicator library for financial data streaming, built on numpy, polars, and matplotlib.

## Project Structure

```
pyta2/
├── base/           # Core base classes (rIndicator, Schema)
├── trend/ma/       # Moving averages (SMA, EMA, WMA, HMA, DEMA, TEMA)
├── momentum/      # Momentum indicators (MACD, RSI)
├── stats/         # Statistical indicators (ATR, zscore)
├── volume/        # Volume indicators (VWAP)
├── structure/     # Structure indicators (Bollinger, ZigZag, Ichimoku)
├── relation/      # Relation indicators (cross, correlation)
├── cycle/         # Cycle indicators (aroon, PSY)
├── perf/          # Performance calculators
└── utils/         # Utilities (vector, deque, plot, space)
```

## Build/Lint/Test Commands

### Running Tests
```bash
# Run all tests
python -m pytest

# Run single test file
python -m pytest test/trend/ma/test_sma.py -v

# Run single test function
python -m pytest test/trend/ma/test_sma.py::TestSMA::test_sma_initialization -v

# Run with coverage
python -m pytest --cov=pyta2

# Run tests matching pattern
python -m pytest -k "sma" -v
```

### Test Discovery
Tests are in `test/` directory, mirroring the `pyta2/` structure. Use pytest's `::Class::method` syntax for specific tests.

### Installation
```bash
pip install -e .  # Development install
```

## Code Style Guidelines

### Naming Conventions

| Pattern | Example | Usage |
|---------|---------|-------|
| `r` prefix | `rSMA`, `rEMA`, `rATR` | Rolling indicator classes |
| lowercase | `sma`, `ema` | Batch functions, modules |
| PascalCase | `Schema`, `Box`, `NumPyDeque` | Classes |
| snake_case | `forward_rolling_apply` | Functions, methods |
| CAPS | `SMA`, `EMA` | Constants, batch function names |
| `_` prefix | `_cache_data`, `_resize_buffer` | Private methods |
| `__` prefix | `__init__` | Dunder methods |

### Imports

Standard library imports first, then third-party, then local:
```python
from abc import ABC, abstractmethod
from typing import Any, Optional, Union, List, Tuple, Dict
from collections import OrderedDict

import numpy as np
from numpy.typing import NDArray, DTypeLike

from ...base.indicator import rIndicator
from ...utils.space.box import Box
```

### Type Annotations

Use typing module comprehensively:
```python
from typing import Optional, Union, List, Tuple, Dict, Any, Sequence, Iterator
from numpy.typing import NDArray, DTypeLike

def forward(self, values: np.ndarray) -> Union[float, np.floating]:
    ...

def __init__(self, dtype: DTypeLike = np.float64) -> None:
    ...
```

### Class Structure

Indicator classes follow this pattern:
```python
class rSMA(rBaseMA):
    name = "SMA"  # Class-level attribute

    def __init__(self, n: int, **kwargs) -> None:
        assert n > 0, f'{self.name} n must be greater than 0, got {n}'
        super(rSMA, self).__init__(n=n, window=n, **kwargs)
        self.n = n

    def reset_extras(self) -> None:
        pass

    def forward(self, values: np.ndarray) -> np.floating:
        if len(values) < self.n:
            return np.nan
        return np.mean(values[-self.n:])

    @property
    def full_name(self) -> str:
        return f'{self.name}({self.n})'
```

### Error Handling

Use assertions for parameter validation:
```python
assert n > 0, f'{self.name} n must be greater than 0, got {n}'
assert isinstance(schema, (list, dict, OrderedDict, Schema)), 'schema is required'
```

Raise exceptions for runtime errors:
```python
if index < 0:
    raise IndexError("insert index out of range")
```

### Docstrings

Use Google-style docstrings with Parameters and Returns sections:
```python
class rATR(rIndicator):
    """Average True Range (ATR) indicator
    
    Parameters
    ----------
    n : int
        Length of ATR calculation period
    ma_type : str, default 'EMA'
        Type of moving average to use ('SMA', 'EMA', etc.)
        
    Returns
    -------
    atr : float
        ATR value
    """
```

### Schema Definition

Define output schemas using Box/Space classes:
```python
schema = [
    ('ma', Box(low=-np.inf, high=np.inf, shape=(), dtype=np.float64))
]

schema = {
    'acc_pnl': Scalar(low=-np.inf, high=np.inf),
    'acc_fee': Scalar(low=0, high=np.inf)
}
```

### Abstract Base Classes

For new indicator categories, inherit from `rIndicator`:
```python
from ...base.indicator import rIndicator

class rNewIndicator(rIndicator):
    name = "NewIndicator"
    
    def __init__(self, param: int, **kwargs):
        super().__init__(
            window=param,
            schema=[('output', Box(low=-np.inf, high=np.inf, dtype=np.float64))],
            **kwargs
        )
        self.param = param
    
    def reset_extras(self) -> None:
        pass
    
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @property
    @abstractmethod
    def full_name(self) -> str:
        pass
```

### Rolling vs Batch Functions

- **Rolling classes** (`rSMA`): Stateful, use `@property full_name`, implement `forward()` and `reset_extras()`
- **Batch functions** (`SMA`): Stateless, operate on full arrays, defined in `_batch.py`

### Space/Box Classes

Use provided space classes for schema definitions:
```python
from ...utils.space.box import Box, Scalar, PositiveScalar
from ...utils.space import Space

# Scalar (0D box)
Scalar(low=-np.inf, high=np.inf, dtype=np.float64)

# Positive values only
PositiveScalar(high=np.inf)

# Multi-dimensional box
Box(low=[0, 0], high=[1, 1], dtype=np.float64)
```

### Key Patterns

1. **Deque for buffers**: Use `NumPyDeque` for rolling data storage
2. **OrderedDict for schema**: Maintains field ordering
3. **g_index tracking**: Indicators track global index via `self.g_index`
4. **Window calculation**: Use `self.required_window = self.window + self.extra_window`

### Performance Notes

- numpy is preferred over polars for core calculations
- Use vectorized operations where possible
- Deque-based buffers prevent memory issues with streaming data
- Buffer factor controls pre-allocation growth

### Common Pitfalls

1. Don't modify `g_index` manually except in `rolling()` method
2. Schema keys must match output field names
3. When inheriting, always call `super().__init__()` with window and schema
4. Use `np.nan` for invalid values, not `None`
