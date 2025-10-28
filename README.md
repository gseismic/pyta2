# pyta2 
常见技术指标滚动计算 基于numpy,polars,matplotlib
定位：规范化无冗余的pyta库 

## 指标列表
- trend
    - ma
        - SMA
        - EMA
        - WMA
        - HMA
        - DEMA
        - TEMA
- perf
    - PositionSizePerf
    - OrderVolumePerf


## 设计原则
**性能优先**
**极简设计**

## 指标列表
## 🧭 技术指标分类总览

我们可以将所有技术指标划分为以下七大类：

1. **趋势类（Trend Indicators）**
2. **动量类（Momentum Indicators）**
3. **波动率类（Volatility Indicators）**
4. **成交量类（Volume Indicators）**
5. **价格通道类（Price Channel Indicators）**
6. **市场强弱与广度类（Market Strength & Breadth）**
7. **综合/混合类（Composite / Hybrid Indicators）**

## ① 趋势类指标（Trend Indicators）

> 主要用于判断市场趋势的方向与强度。

| 指标名称      | 简写（代码用）    | 说明                                    |
| --------- | ---------- | ------------------------------------- |
| 移动平均线     | `MA`       | 简单移动平均（Simple Moving Average）         |
| 指数移动平均线   | `EMA`      | Exponential Moving Average            |
| 加权移动平均线   | `WMA`      | Weighted Moving Average               |
| Hull移动平均线 | `HMA`      | Hull Moving Average                   |
| 平滑移动平均线   | `SMMA`     | Smoothed Moving Average               |
| 移动平均收敛散度  | `MACD`     | Moving Average Convergence Divergence |
| 平均方向性指数   | `ADX`      | Average Directional Index             |
| 抛物线转向指标   | `SAR`      | Parabolic Stop and Reverse            |
| 一目均衡表     | `ICHIMOKU` | Ichimoku Kinko Hyo                    |
| 三重指数平均线   | `TEMA`     | Triple Exponential Moving Average     |

---

## ② 动量类指标（Momentum Indicators）

> 衡量价格变动速度，用于识别超买/超卖与趋势反转。

| 指标名称       | 简写           | 说明                                    |
| ---------- | ------------ | ------------------------------------- |
| 相对强弱指数     | `RSI`        | Relative Strength Index               |
| 随机震荡指标     | `STOCH`      | Stochastic Oscillator                 |
| 动量指标       | `MOM`        | Momentum                              |
| 变化率指标      | `ROC`        | Rate of Change                        |
| 威廉指标       | `WILLIAMS_R` | Williams %R                           |
| 商品通道指数     | `CCI`        | Commodity Channel Index               |
| 终极震荡指标     | `ULTOSC`     | Ultimate Oscillator                   |
| 平均动量指数     | `AMI`        | Average Momentum Index（较少使用）          |
| TRIX三重指数动量 | `TRIX`       | Triple Exponential Average Oscillator |

---

## ③ 波动率类指标（Volatility Indicators）

> 衡量价格的波动性，用于风险管理、突破判断。

| 指标名称   | 简写         | 说明                         |
| ------ | ---------- | -------------------------- |
| 布林带    | `BBANDS`   | Bollinger Bands            |
| 平均真实波幅 | `ATR`      | Average True Range         |
| 标准差通道  | `STDDEV`   | Standard Deviation Channel |
| 唐奇安通道  | `DONCHIAN` | Donchian Channel           |
| 压缩波动率  | `KC`       | Keltner Channel            |
| 价格变异指数 | `CV`       | Coefficient of Variation   |

---

## ④ 成交量类指标（Volume Indicators）

> 分析成交量与价格之间的关系，判断趋势的有效性。

| 指标名称     | 简写        | 说明                             |
| -------- | --------- | ------------------------------ |
| 能量潮      | `OBV`     | On Balance Volume              |
| 累积/派发线   | `ADL`     | Accumulation/Distribution Line |
| 成交量变化率   | `VROC`    | Volume Rate of Change          |
| 资金流量指标   | `MFI`     | Money Flow Index               |
| 平均成交量    | `VMA`     | Volume Moving Average          |
| 平衡成交量变动率 | `PVI/NVI` | Positive/Negative Volume Index |
| 易变性指标    | `EOM`     | Ease of Movement               |

---

## ⑤ 价格通道类指标（Price Channel Indicators）

> 用于寻找突破、支撑和阻力。

| 指标名称   | 简写           | 说明                     |
| ------ | ------------ | ---------------------- |
| 唐奇安通道  | `DONCHIAN`   | Donchian Channel       |
| 布林带    | `BBANDS`     | Bollinger Bands（兼波动率）  |
| 肯特纳通道  | `KC`         | Keltner Channel        |
| 价格通道突破 | `PCBREAK`    | Price Channel Breakout |
| 线性回归通道 | `REGCHANNEL` | Regression Channel     |

---

## ⑥ 市场强弱与广度类指标（Market Strength / Breadth）

> 反映整体市场的内部强弱关系，常用于指数级别分析。

| 指标名称         | 简写       | 说明                            |
| ------------ | -------- | ----------------------------- |
| 涨跌比率         | `ADR`    | Advance/Decline Ratio         |
| 涨跌差额         | `ADD`    | Advance/Decline Difference    |
| 涨跌线          | `ADLINE` | Advance/Decline Line          |
| 麦克莱伦振荡指标     | `MCO`    | McClellan Oscillator          |
| 麦克莱伦累积指标     | `MCSUM`  | McClellan Summation Index     |
| ARMS指数（TRIN） | `TRIN`   | Trading Index                 |
| 上涨下跌成交量比     | `UVDR`   | Up Volume / Down Volume Ratio |

---

## ⑦ 综合/混合类指标（Composite / Hybrid）

> 将趋势、动量、成交量等信号结合，生成交易信号。

| 指标名称        | 简写         | 说明                                     |
| ----------- | ---------- | -------------------------------------- |
| 相对强度比较      | `RS_COMP`  | Relative Strength Comparison（个股 vs 指数） |
| 平衡点分析       | `PIVOT`    | Pivot Points                           |
| DMI系统       | `DMI`      | Directional Movement Index（含ADX）       |
| 移动平均震荡      | `MAOSC`    | MA Oscillator                          |
| 波动突破信号      | `BREAKOUT` | Volatility Breakout Signal             |
| TTM Squeeze | `TTM_SQZ`  | TTM Squeeze（布林带 + KC 通道）               |
| 综合评分指标      | `SCORE`    | Multi-Factor Score (自定义因子综合)           |

---

## 📘 建议的代码结构

如果你在设计财务分析库，可按如下方式组织指标模块结构：

```
indicators/
│
├── trend.py          # 趋势类
├── momentum.py       # 动量类
├── volatility.py     # 波动率类
├── volume.py         # 成交量类
├── channel.py        # 价格通道类
├── breadth.py        # 市场强弱类
└── composite.py      # 综合类
```

每个文件导出一个统一接口，例如：

```python
def MA(close: np.ndarray, period: int = 20) -> np.ndarray:
    """移动平均线 | Moving Average"""
    return np.convolve(close, np.ones(period)/period, mode='valid')
```

---

是否希望我接着为你生成一个**标准指标注册表（registry）**，例如：

```python
INDICATOR_REGISTRY = {
    "MA": ("trend", "移动平均线"),
    "RSI": ("momentum", "相对强弱指数"),
    ...
}
```

这样可支持自动加载和动态调用指标（如 `get_indicator("RSI")(data)`）。是否继续生成？
